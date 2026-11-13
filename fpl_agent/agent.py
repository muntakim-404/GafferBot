"""Step 3: the hourly "tick".

Each run (GitHub Actions starts one every hour):
  1. reads commands you sent the bot (/ft, /plan, /status, /help);
  2. scores finished gameweeks: what the bot's plan would have scored vs what
     you actually scored (the shadow log, logs/shadow.csv);
  3. if the next deadline is close, builds a plan and sends it to Telegram:
     an early message (<= 6.5h before) and a final one (<= 2h before), once each.

It only reads from FPL. It never changes your team.
State lives in files so it survives between runs (the workflow commits them):
  state/agent_state.json, logs/plans/gwNN-<window>.json, logs/shadow.csv
"""
from __future__ import annotations

import csv
import html
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .chips import chip_outlook, format_chips, remaining_chips
from .config import DEF, FWD, GK, MID, POS_NAME, ModelParams, SolverParams, captain_score
from .data import load_live, team_state_public
from .predictor import predict, xpts_matrix
from .solver import solve, validate
from .telegram import TelegramError

# GitHub's "hourly" schedule can in practice run every 4-8 hours, so the windows
# are wide enough that at least one run lands in each: the early plan is sent by
# the first run within 24h of the deadline, the final by the first within 4h.
EARLY_HOURS = 24.0
FINAL_HOURS = 4.0
CLOSE_CAPTAIN = 1.0      # captain options closer than this are within the model's error
CHIP_NAMES = {"bboost": "Bench Boost", "3xc": "Triple Captain", "freehit": "Free Hit", "wildcard": "Wildcard"}
SHADOW_COLS = ["gw", "window", "bot_points", "your_points", "diff", "bot_captain", "your_captain",
               "bot_transfers", "your_chip", "auto_subs"]


@dataclass
class AgentConfig:
    team_id: int
    tz: str = "UTC"
    horizon: int = 5
    root: Path = Path(".")
    early_hours: float = EARLY_HOURS
    final_hours: float = FINAL_HOURS
    chips: bool = True

    @property
    def state_file(self) -> Path: return self.root / "state" / "agent_state.json"
    @property
    def plans_dir(self) -> Path: return self.root / "logs" / "plans"
    @property
    def shadow_file(self) -> Path: return self.root / "logs" / "shadow.csv"


# ------------------------------------------------------------------------ state
def load_state(cfg: AgentConfig) -> dict:
    st = {"telegram_offset": 0, "ft": None, "sent": {}, "scored": [], "errors": []}
    if cfg.state_file.exists():
        st.update(json.loads(cfg.state_file.read_text(encoding="utf-8")))
    return st


def save_state(cfg: AgentConfig, st: dict) -> None:
    cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.state_file.write_text(json.dumps(st, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- formatting
def _e(s) -> str:
    return html.escape(str(s), quote=False)


def _local(ts: pd.Timestamp, tz: str) -> str:
    return pd.Timestamp(ts).tz_convert(ZoneInfo(tz)).strftime("%a %d %b %H:%M")


def _until(deadline: pd.Timestamp, now: pd.Timestamp) -> str:
    mins = int((deadline - now).total_seconds() // 60)
    if mins <= 0:
        return "passed"
    return f"{mins // 60}h {mins % 60:02d}m" if mins >= 60 else f"{mins}m"


def format_plan(p: dict, tz: str, now: pd.Timestamp) -> str:
    nm = lambda i: _e(p["names"][str(i)])
    xp = lambda i: p["xpts"][str(i)]
    dl = pd.Timestamp(p["deadline_utc"])
    title = {"early": "early look", "final": "final check", "manual": "on request"}[p["window"]]
    lines = [f"<b>FPL plan · GW{p['gw']}</b> ({title})",
             f"Deadline: {_local(dl, tz)} ({_e(tz)}) · in {_until(dl, now)}", ""]
    ft = f"{p['free_transfers']} (assumed: reply /ft N if different)" if p["ft_assumed"] else str(p["free_transfers"])
    lines += [f"Free transfers: {ft} · Bank £{p['bank']:.1f}m", ""]

    lines.append("<b>Transfers</b>")
    if p["transfers_in"]:
        for o, i in zip(p["transfers_out"], p["transfers_in"]):
            lines.append(f"OUT {nm(o)} (£{p['sell'][str(o)]:.1f}m) → IN {nm(i)} (£{p['price'][str(i)]:.1f}m)")
        lines.append(f"Hits: {p['hits']}{' (−' + str(4 * p['hits']) + ' pts)' if p['hits'] else ''} · "
                     f"Bank after £{p['bank_after']:.1f}m")
    else:
        lines.append("None: save the free transfer.")
    lines.append("")

    cap, vice = p["captain"], p["vice"]
    lines.append(f"<b>Captain</b>: {nm(cap)} · <b>Vice</b>: {nm(vice)}")
    cs = {int(k): v for k, v in p.get("cap_score", {}).items()}
    if cs:
        ranked = sorted(cs, key=lambda i: -cs[i])
        lines.append("Captain scores (bias-corrected): " + " · ".join(f"{nm(i)} {cs[i]:.1f}" for i in ranked[:3]))
        if len(ranked) > 1 and cs[ranked[0]] - cs[ranked[1]] < CLOSE_CAPTAIN:
            lines.append("⚠️ Close call: the top two are within the model's error; your choice.")
    lines.append("")

    lines.append(f"<b>XI</b> (expected {p['expected_points']:.1f} incl. captain)")
    for pos in (GK, DEF, MID, FWD):
        ids = [i for i in p["lineup"] if p["pos"][str(i)] == pos]
        lines.append(f"{POS_NAME[pos]} " + " · ".join(f"{nm(i)} {xp(i):.1f}" for i in ids))
    lines.append("<b>Bench</b>: " + ", ".join(nm(i) for i in p["bench"]))

    if p["injuries"]:
        lines += ["", "<b>Injury news in your squad</b>"]
        lines += [f"• {_e(n)}: {_e(news)}" for n, news in p["injuries"]]
    if p.get("chip_lines"):
        lines += [""] + p["chip_lines"]
    if p["later"]:
        lines += ["", "Next weeks: " + " · ".join(_e(s) for s in p["later"])]
    if p["notes"]:
        lines += [""] + [f"<i>{_e(n)}</i>" for n in p["notes"]]
    lines += ["", "/ft N · /plan · /status · /help"]
    return "\n".join(lines)


HELP = ("<b>Commands</b> (read at the bot's next run: usually within an hour, sometimes a few)\n"
        "/ft N: set your free transfers for the coming gameweek (0–5)\n"
        "/plan: send a plan at the next run, whatever the time\n"
        "/status: next deadline, settings, season comparison\n"
        "/help: this message")


# ------------------------------------------------------------------ plan building
def build_plan(cfg: AgentConfig, client, ft_override: int | None, window: str, now: pd.Timestamp) -> dict:
    ds = load_live(client, verbose=False)
    if ds.next_gw is None:
        raise RuntimeError("No upcoming gameweek.")
    state = team_state_public(client, cfg.team_id, ds, ft_override)
    if ft_override is not None:
        state.notes.insert(0, f"Free transfers set to {ft_override} by your /ft command.")

    chips_left = {}
    if cfg.chips:
        try:
            used = client.entry_history(cfg.team_id).get("chips", [])
            chips_left = remaining_chips(ds.next_gw, used, client.bootstrap().get("chips"))
        except Exception as e:   # never let the chip outlook block the transfer plan
            print(f"chip lookup failed: {e}")
    plan_gws = list(range(ds.next_gw, min(ds.next_gw + cfg.horizon, 39)))
    last = max([plan_gws[-1]] + list(chips_left.values()))
    pred = predict(ds, list(range(ds.next_gw, min(last, 38) + 1)), ModelParams())
    xp_all = xpts_matrix(pred)
    plan = solve(xp_all[plan_gws], ds.players, state, SolverParams())
    problems = validate(plan, ds.players, state)
    if problems:
        raise RuntimeError("Plan failed validation: " + "; ".join(problems))

    info = ds.players.set_index("id")
    chip_lines = []
    if chips_left:
        try:
            budget = sum(state.selling_price.get(p, info.at[p, "price"]) for p in state.squad) + state.bank
            outlook = chip_outlook(xp_all, ds.players, {w.gw: w.squad for w in plan.weeks}, budget,
                                   ds.next_gw, chips_left)
            chip_lines = format_chips(outlook, info["web_name"].to_dict(), _e)
        except Exception as e:
            print(traceback.format_exc())
            chip_lines = [f"<i>Chip outlook unavailable this time ({_e(type(e).__name__)}).</i>"]
    wk = plan.weeks[0]
    gw = wk.gw
    xp = pred[pred["gw"] == gw].set_index("id")["xpts"]
    ids = set(wk.squad) | set(wk.transfers_out)
    injuries = [(info.at[i, "web_name"], info.at[i, "news"]) for i in wk.squad
                if info.at[i, "status"] != "a" and info.at[i, "news"]]
    later = []
    for w in plan.weeks[1:3]:
        if w.transfers_in:
            later.append(f"GW{w.gw} " + ", ".join(f"{info.at[o, 'web_name']} → {info.at[i, 'web_name']}"
                                                   for o, i in zip(w.transfers_out, w.transfers_in)))
        else:
            later.append(f"GW{w.gw} no transfers")
    return {
        "gw": gw, "window": window, "generated_at": now.isoformat(),
        "deadline_utc": str(ds.deadlines.get(gw)),
        "free_transfers": state.free_transfers, "ft_assumed": state.ft_assumed,
        "bank": state.bank, "bank_after": wk.bank_after, "hits": wk.hits,
        "transfers_in": wk.transfers_in, "transfers_out": wk.transfers_out,
        "lineup": wk.lineup, "bench": wk.bench, "captain": wk.captain, "vice": wk.vice,
        "expected_points": wk.xpts,
        "squad": wk.squad,
        "xpts": {str(i): round(float(xp.get(i, 0.0)), 2) for i in ids},
        "pos": {str(i): int(info.at[i, "pos"]) for i in ids},
        "names": {str(i): info.at[i, "web_name"] for i in ids},
        "price": {str(i): float(info.at[i, "price"]) for i in ids},
        "sell": {str(i): float(state.selling_price.get(i, info.at[i, "price"])) for i in ids},
        "injuries": injuries, "later": later, "chip_lines": chip_lines,
        "cap_score": {str(i): round(captain_score(int(info.at[i, "pos"]), float(xp.get(i, 0.0))), 2)
                      for i in wk.lineup if captain_score(int(info.at[i, "pos"]), 0.0) is not None},
        "notes": [n.replace("re-run with --ft N", "reply /ft N") for n in state.notes],
    }


# -------------------------------------------------------------------- scoring
def _formation_ok(xi: list[int], pos: dict) -> bool:
    c = {k: sum(pos[i] == k for i in xi) for k in (GK, DEF, MID, FWD)}
    return c[GK] == 1 and c[DEF] >= 3 and c[MID] >= 2 and c[FWD] >= 1


def score_lineup(lineup, bench, captain, vice, pos, pts, mins):
    """Points the plan would have scored, with FPL's auto-sub and vice-captain rules.

    Returns (points, [(out, in) auto-subs], player who got the captain's double or None).
    """
    xi, used, subs = list(lineup), set(), []
    for k, p in enumerate(list(xi)):
        if mins.get(p, 0) > 0:
            continue
        for b in bench:
            if b in used or mins.get(b, 0) <= 0 or (pos[b] == GK) != (pos[p] == GK):
                continue
            trial = xi[:k] + [b] + xi[k + 1:]
            if _formation_ok(trial, pos):
                xi = trial
                used.add(b)
                subs.append((p, b))
                break
    total = sum(pts.get(i, 0) for i in xi)
    doubled = captain if mins.get(captain, 0) > 0 else vice if mins.get(vice, 0) > 0 else None
    if doubled is not None:
        total += pts.get(doubled, 0)
    return total, subs, doubled


def _best_plan_file(cfg: AgentConfig, gw: int) -> Path | None:
    for w in ("final", "early", "manual"):
        f = cfg.plans_dir / f"gw{gw:02d}-{w}.json"
        if f.exists():
            return f
    return None


def score_gameweek(cfg: AgentConfig, client, gw: int) -> dict | None:
    f = _best_plan_file(cfg, gw)
    if f is None:
        return None
    p = json.loads(f.read_text(encoding="utf-8"))
    live = client.live(gw)
    pts = {e["id"]: e["stats"]["total_points"] for e in live["elements"]}
    mins = {e["id"]: e["stats"]["minutes"] for e in live["elements"]}
    pos = {int(k): v for k, v in p["pos"].items()}
    bot, subs, doubled = score_lineup(p["lineup"], p["bench"], p["captain"], p["vice"], pos, pts, mins)
    bot -= 4 * p["hits"]
    picks = client.picks(cfg.team_id, gw)
    eh = picks["entry_history"]
    yours = eh["points"] - eh.get("event_transfers_cost", 0)
    your_cap = next((x["element"] for x in picks["picks"] if x.get("is_captain")), None)
    names = {int(k): v for k, v in p["names"].items()}
    return {
        "gw": gw, "window": p["window"], "bot_points": bot, "your_points": yours, "diff": bot - yours,
        "bot_captain": names.get(doubled, "none played"),
        "your_captain": names.get(your_cap, str(your_cap)),
        "bot_transfers": "; ".join(f"{names[o]}->{names[i]}" for o, i in zip(p["transfers_out"], p["transfers_in"])) or "none",
        "your_chip": CHIP_NAMES.get(picks.get("active_chip"), picks.get("active_chip") or ""),
        "auto_subs": "; ".join(f"{names.get(o, o)}->{names.get(i, i)}" for o, i in subs),
    }


def append_shadow(cfg: AgentConfig, row: dict) -> pd.DataFrame:
    cfg.shadow_file.parent.mkdir(parents=True, exist_ok=True)
    new = not cfg.shadow_file.exists()
    with open(cfg.shadow_file, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHADOW_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    return pd.read_csv(cfg.shadow_file)


def shadow_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No gameweeks scored yet."
    chip = df["your_chip"].astype(object).where(df["your_chip"].notna(), "")   # safe on all-empty columns
    fair = df[chip == ""]
    tot = int(fair["diff"].sum())
    return (f"Season so far: plan {'+' if tot >= 0 else ''}{tot} pts vs you over {len(fair)} GW"
            f"{'s' if len(fair) != 1 else ''} (plan better in {(fair['diff'] > 0).sum()}, you in "
            f"{(fair['diff'] < 0).sum()}, level {(fair['diff'] == 0).sum()})"
            + (f"; {len(df) - len(fair)} chip week(s) left out" if len(df) > len(fair) else "") + ".")


def format_score(row: dict, df: pd.DataFrame) -> str:
    d = row["diff"]
    lines = [f"<b>GW{row['gw']} result</b>",
             f"Plan: {row['bot_points']} pts (captain {_e(row['bot_captain'])}"
             + (f"; auto-subs {_e(row['auto_subs'])}" if row["auto_subs"] else "") + ")",
             f"You: {row['your_points']} pts (captain {_e(row['your_captain'])})",
             f"{'Plan' if d > 0 else 'You'} {'+' + str(abs(d)) if d else 'level'}" + (" this week" if d else ""),
             ]
    if row["your_chip"]:
        lines.append(f"<i>You played {_e(row['your_chip'])}, so this week isn't like-for-like.</i>")
    lines += ["", _e(shadow_summary(df))]
    return "\n".join(lines)


# ----------------------------------------------------------------------- commands
def handle_command(text: str, st: dict, next_gw: int | None, deadline, cfg: AgentConfig) -> tuple[str, bool]:
    """Returns (reply, plan_now)."""
    parts = text.strip().split()
    if not parts:
        return "", False
    if not parts[0].startswith("/"):
        return HELP, False                     # plain messages ("hi") get the command list
    cmd = parts[0].lower().split("@")[0]
    if cmd == "/ft":
        if len(parts) == 2 and parts[1].isdigit() and 0 <= int(parts[1]) <= 5 and next_gw:
            st["ft"] = {"gw": next_gw, "value": int(parts[1])}
            return f"OK: GW{next_gw} will be planned with {parts[1]} free transfer(s).", False
        return "Usage: /ft N (N from 0 to 5)", False
    if cmd == "/plan":
        return "OK: building a plan now.", True
    if cmd == "/status":
        ft = st.get("ft")
        ft_txt = f"{ft['value']} (set by you)" if ft and ft["gw"] == next_gw else "1 assumed (send /ft N to set)"
        df = pd.read_csv(cfg.shadow_file) if cfg.shadow_file.exists() else pd.DataFrame(columns=SHADOW_COLS)
        sent = ", ".join(st["sent"].get(str(next_gw), [])) or "none yet"
        dl = _local(deadline, cfg.tz) + f" ({cfg.tz})" if deadline is not None else "unknown"
        return (f"<b>Status</b>\nNext: GW{next_gw}, deadline {_e(dl)}\nFree transfers: {ft_txt}\n"
                f"Plans sent for GW{next_gw}: {sent}\n{_e(shadow_summary(df))}"), False
    if cmd in ("/help", "/start"):
        return HELP, False
    return "Unknown command. " + HELP, False


# --------------------------------------------------------------------------- tick
FAIL_ALERT_AFTER = 3      # failed runs in a row before you're alerted (~1.5h at a 30-min cadence)
FAIL_REMIND_EVERY = 24    # while still failing, remind every N runs (~12h)
QUIET_AFTER_DEADLINE = 4  # hours after a deadline when FPL is often down for updates


def tick(cfg: AgentConfig, client, tg, now: pd.Timestamp | None = None, force: bool = False) -> int:
    """One scheduled run. Returns a process exit code (1 makes GitHub email you).

    One-off glitches (a blip reaching FPL or Telegram) are simply retried at the
    next run. You're alerted, on Telegram if it works and by GitHub's failure
    email, when runs fail FAIL_ALERT_AFTER times in a row, or straight away if
    Telegram rejects the bot (wrong token or chat id), which can't fix itself.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    st = load_state(cfg)
    stage = ["starting"]
    try:
        _run(cfg, client, tg, now, force, st, stage)
    except Exception as e:
        print(traceback.format_exc())
        return _on_failure(st, tg, e, stage[0], now)
    else:
        if st.get("fail_streak"):
            print(f"Recovered after {st['fail_streak']} failed run(s).")
            st["fail_streak"] = 0
        return 0
    finally:
        save_state(cfg, st)


def _on_failure(st: dict, tg, e: Exception, stage: str, now: pd.Timestamp) -> int:
    streak = st["fail_streak"] = st.get("fail_streak", 0) + 1
    msg = f"{type(e).__name__}: {str(e)[:300]}"
    st["last_error"] = {"at": now.isoformat(), "stage": stage, "error": msg, "streak": streak}
    permanent = isinstance(e, TelegramError) and e.permanent
    last_dl = st.get("last_deadline_utc")
    after_deadline = (last_dl is not None and
                      pd.Timestamp(last_dl) <= now <= pd.Timestamp(last_dl) + pd.Timedelta(hours=QUIET_AFTER_DEADLINE))
    due = streak == FAIL_ALERT_AFTER or (streak > FAIL_ALERT_AFTER and streak % FAIL_REMIND_EVERY == 0)
    if not permanent and (after_deadline or not due):
        why = "FPL is usually updating after a deadline" if after_deadline else f"{streak} in a row so far"
        print(f"Run failed while {stage} ({why}); retrying next run without alerting.")
        return 0
    if not isinstance(e, TelegramError):
        try:
            tg.send(f"⚠️ <b>The bot has failed {streak} runs in a row</b> while {_e(stage)}.\n"
                    f"{_e(msg)}\nIt keeps retrying every run. If a deadline is close, make your moves "
                    f"in the FPL app. If FPL can't be reached for hours outside a deadline, FPL may be "
                    f"blocking GitHub's servers.")
        except Exception as e2:
            print(f"Couldn't alert on Telegram either: {e2}")
    return 1


def _run(cfg: AgentConfig, client, tg, now: pd.Timestamp, force: bool, st: dict, stage: list) -> None:
    stage[0] = "reading FPL's gameweek list"
    bs = client.bootstrap(max_age=None)
    ev = pd.DataFrame(bs["events"])
    ev["deadline"] = pd.to_datetime(ev["deadline_time"], utc=True)
    nxt = ev[ev["is_next"].astype(bool)]
    next_gw = int(nxt["id"].iloc[0]) if len(nxt) else None
    deadline = nxt["deadline"].iloc[0] if len(nxt) else None
    if deadline is not None:
        st["last_deadline_utc"] = deadline.isoformat()

    # 1. commands you sent the bot
    stage[0] = "reading your Telegram messages"
    plan_now = force
    for upd in tg.updates(st["telegram_offset"] + 1 if st["telegram_offset"] else 0):
        st["telegram_offset"] = max(st["telegram_offset"], upd["update_id"])
        msg = upd.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != str(tg.chat_id):
            continue                         # ignore anyone else who finds the bot
        reply, now_flag = handle_command(msg.get("text") or "", st, next_gw, deadline, cfg)
        plan_now |= now_flag
        if reply:
            tg.send(reply)
    save_state(cfg, st)

    # 2. deadline plans: time-critical, so before anything optional
    if next_gw is not None:
        hours = (deadline - now).total_seconds() / 3600
        sent = st["sent"].setdefault(str(next_gw), [])
        window = None
        if 0 < hours <= cfg.final_hours and "final" not in sent:
            window = "final"
        elif 0 < hours <= cfg.early_hours and not sent:
            window = "early"
        if window is None and plan_now:
            window = "manual"
        if window is not None:
            ft = st.get("ft")
            ft_override = ft["value"] if ft and ft["gw"] == next_gw else None
            stage[0] = f"building the GW{next_gw} {window} plan"
            plan = build_plan(cfg, client, ft_override, window, now)
            cfg.plans_dir.mkdir(parents=True, exist_ok=True)
            (cfg.plans_dir / f"gw{plan['gw']:02d}-{window}.json").write_text(
                json.dumps(plan, indent=2, default=str), encoding="utf-8")
            stage[0] = "sending the plan to Telegram"
            tg.send(format_plan(plan, cfg.tz, now))
            if window != "manual":
                sent.append(window)          # only once it's really been sent
            save_state(cfg, st)

    # 3. shadow log: optional, so a problem here never blocks a plan
    stage[0] = "scoring finished gameweeks"
    _score_finished(cfg, client, tg, ev, st, now)


def _score_finished(cfg: AgentConfig, client, tg, ev: pd.DataFrame, st: dict, now: pd.Timestamp) -> None:
    done = ev[ev["finished"].astype(bool) & ev.get("data_checked", ev["finished"]).astype(bool)]
    fails = st.setdefault("score_fails", {})
    for gw in sorted(int(g) for g in done["id"]):
        if gw in st["scored"] or _best_plan_file(cfg, gw) is None:
            continue
        try:
            row = score_gameweek(cfg, client, gw)
        except Exception as e:
            print(traceback.format_exc())
            n = fails[str(gw)] = fails.get(str(gw), 0) + 1
            st["last_error"] = {"at": now.isoformat(), "stage": f"scoring GW{gw}",
                                "error": f"{type(e).__name__}: {str(e)[:300]}", "streak": n}
            if n == FAIL_ALERT_AFTER:
                tg.send(f"<i>Couldn't score GW{gw} for the shadow log after {n} tries "
                        f"({_e(type(e).__name__)}). Plans aren't affected; it keeps trying.</i>")
            continue
        if row:
            df = append_shadow(cfg, row)
            st["scored"].append(gw)          # before sending, so a send failure can't cause a duplicate row
            fails.pop(str(gw), None)
            save_state(cfg, st)
            tg.send(format_score(row, df))
