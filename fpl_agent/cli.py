"""Command line entry point: `python -m fpl_agent <command> ...`"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from .backtest import run_backtest
from .config import POS_NAME, ModelParams, SolverParams
from .data import (Dataset, FPLAuthError, FPLClient, TeamState, load_archive, load_live, resolve_players,
                   team_state_from_my_team, team_state_public)
from .predictor import predict, xpts_matrix
from .solver import Plan, SolverError, most_owned_squad, solve, validate


# ------------------------------------------------------------------ input helpers
def _names(arg: str | None, players: pd.DataFrame) -> list[int]:
    return resolve_players(arg.split(","), players) if arg else []


def _load_dataset(args) -> tuple[Dataset, FPLClient | None]:
    if args.archive:
        return load_archive(args.archive), None
    client = FPLClient(cache_dir=args.cache_dir, access_token=args.access_token or os.environ.get("FPL_ACCESS_TOKEN"))
    print("Loading FPL data (first run fetches ~500 player histories, then cached)...")
    return load_live(client, refresh=args.refresh), client


def _load_state(args, ds: Dataset, client: FPLClient | None) -> TeamState | None:
    if args.from_scratch:
        return None
    if args.squad_file:
        # utf-8-sig: names like "Groß" break under Windows' default cp1252, and Notepad may add a BOM.
        with open(args.squad_file, encoding="utf-8-sig") as fh:
            js = json.load(fh)
        squad = resolve_players([str(x) for x in js["squad"]], ds.players)
        price = ds.players.set_index("id")["price"]
        sp = {int(k): float(v) for k, v in js.get("selling_prices", {}).items()}
        sp = {p: sp.get(p, float(price[p])) for p in squad}
        ft = args.ft if args.ft is not None else int(js.get("free_transfers", 1))
        return TeamState(squad, sp, float(js.get("bank", 0.0)), ft, ds.next_gw,
                         ["Squad from file; selling prices default to current prices unless given."])
    if args.template:
        squad = most_owned_squad(ds.players)
        price = ds.players.set_index("id")["price"]
        bank = round(100.0 - sum(price[p] for p in squad), 1)
        return TeamState(squad, {p: float(price[p]) for p in squad}, bank,
                         args.ft if args.ft is not None else 1, ds.next_gw,
                         ["Using the most-owned legal squad as a stand-in for your team."])
    if not args.team_id:
        sys.exit("Give --team-id (live), or --squad-file / --template / --from-scratch.")
    if client is None:
        sys.exit("--team-id needs live data (drop --archive).")
    if client.access_token:
        try:
            st = team_state_from_my_team(client.my_team(args.team_id), ds.next_gw)
            if args.ft is not None:
                st.free_transfers = args.ft
            return st
        except FPLAuthError as e:
            print(f"! {e} Falling back to public data.")
    return team_state_public(client, args.team_id, ds, args.ft)


# ---------------------------------------------------------------------- printing
def _local(ts) -> str:
    if ts is None or pd.isna(ts):
        return "unknown"
    return pd.Timestamp(ts).tz_convert(datetime.now().astimezone().tzinfo).strftime("%a %d %b %H:%M (local)")


def print_plan(plan: Plan, ds: Dataset, pred: pd.DataFrame, state: TeamState | None) -> None:
    info = ds.players.set_index("id")
    club = ds.teams.set_index("id")["short_name"]
    xp = pred.set_index(["id", "gw"])["xpts"]
    nm = lambda p: info.at[p, "web_name"]
    first = plan.weeks[0]
    print(f"\n=== Plan for GW{first.gw}  |  deadline {_local(ds.deadlines.get(first.gw))} ===")
    print(f"Data: {ds.source}, last finished GW{ds.last_finished_gw}")
    if state:
        ft_txt = f"{state.free_transfers} (assumed; pass --ft to set)" if state.ft_assumed else str(state.free_transfers)
        print(f"Free transfers: {ft_txt}  |  Bank: £{state.bank:.1f}m")
        for n in state.notes:
            print(f"  note: {n}")
    for n in plan.notes:
        print(f"  note: {n}")

    if plan.from_scratch:
        print(f"\nBest squad from scratch (cost £{sum(info.at[p, 'price'] for p in first.squad):.1f}m):")
    elif first.transfers_in:
        outs = ", ".join(f"{nm(p)} (sell £{(state.selling_price.get(p) if state else info.at[p, 'price']):.1f}m)" for p in first.transfers_out)
        ins = ", ".join(f"{nm(p)} (£{info.at[p, 'price']:.1f}m)" for p in first.transfers_in)
        print(f"\nTransfers: OUT {outs}\n           IN  {ins}")
        print(f"           hits: {first.hits} (-{4 * first.hits} pts)  |  bank after: £{first.bank_after:.1f}m")
    else:
        print("\nTransfers: none (roll the free transfer)")

    print(f"\nStarting XI (expected points GW{first.gw}):")
    for p in sorted(first.lineup, key=lambda p: (info.at[p, "pos"], -xp[(p, first.gw)])):
        tag = " (C)" if p == first.captain else " (V)" if p == first.vice else ""
        flag = ""
        if ds.has_availability and info.at[p, "news"]:
            flag = f"   ! {info.at[p, 'news']}"
        print(f"  {POS_NAME[info.at[p, 'pos']]:<3} {nm(p) + tag:<22} {club.get(info.at[p, 'team'], ''):<4} "
              f"£{info.at[p, 'price']:>4.1f}m  {xp[(p, first.gw)]:>4.1f}{flag}")
    print("Bench:  " + ", ".join(f"{i + 1}. {nm(p)} ({xp[(p, first.gw)]:.1f})" for i, p in enumerate(first.bench)))
    print(f"XI expected points incl. captain: {first.xpts:.1f}")

    if len(plan.weeks) > 1:
        print("\nLater weeks (provisional, re-planned every week):")
        for wk in plan.weeks[1:]:
            moves = (f"OUT {', '.join(nm(p) for p in wk.transfers_out)} -> IN {', '.join(nm(p) for p in wk.transfers_in)}"
                     if wk.transfers_in else "no transfers")
            print(f"  GW{wk.gw}: {moves}; hits {wk.hits}; captain {nm(wk.captain)}; XI xPts {wk.xpts:.1f}")
    print(f"\nObjective (discounted horizon value): {plan.objective:.1f}  [{plan.status}]")


def _plan_json(plan: Plan, ds: Dataset, state: TeamState | None, sp: SolverParams) -> dict:
    names = ds.players.set_index("id")["web_name"]
    d = plan.to_dict()
    ids = {p for wk in plan.weeks for p in wk.squad + wk.transfers_out}
    d["names"] = {int(p): names.get(p, str(p)) for p in ids}
    d["generated_at"] = datetime.now(timezone.utc).isoformat()
    d["data_source"] = ds.source
    d["deadline_utc"] = str(ds.deadlines.get(plan.weeks[0].gw))
    d["start_state"] = None if state is None else {
        "squad": state.squad, "bank": state.bank, "free_transfers": state.free_transfers, "notes": state.notes}
    d["solver_params"] = {k: v for k, v in vars(sp).items()}
    return d


# ---------------------------------------------------------------------- commands
def cmd_plan(args) -> None:
    ds, client = _load_dataset(args)
    if ds.next_gw is None:
        sys.exit("No upcoming gameweek: the season is over.")
    state = _load_state(args, ds, client)
    gws = list(range(ds.next_gw, min(ds.next_gw + args.horizon, 39)))
    pred = predict(ds, gws, ModelParams(), starters=_names(args.starter, ds.players), out=_names(args.out, ds.players))
    sp = SolverParams(allow_hits=not args.no_hits, max_transfers_first_gw=args.max_transfers,
                      lock=_names(args.lock, ds.players), ban=_names(args.ban, ds.players),
                      horizon_decay=args.decay, time_limit=args.time_limit)
    try:
        plan = solve(xpts_matrix(pred), ds.players, state, sp)
    except SolverError as e:
        sys.exit(f"Could not build a plan: {e}")
    problems = validate(plan, ds.players, state)
    if problems:   # never hand out a plan that breaks the rules
        sys.exit("Plan failed validation:\n  " + "\n  ".join(problems))
    print_plan(plan, ds, pred, state)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(_plan_json(plan, ds, state, sp), f, indent=2, default=str)
        print(f"Saved plan to {args.json}")


def cmd_predict(args) -> None:
    ds, _ = _load_dataset(args)
    if ds.next_gw is None:
        sys.exit("No upcoming gameweek: the season is over.")
    gws = list(range(ds.next_gw, min(ds.next_gw + args.horizon, 39)))
    pred = predict(ds, gws, ModelParams(), starters=_names(args.starter, ds.players), out=_names(args.out, ds.players))
    wide = pred.pivot_table(index="id", columns="gw", values="xpts")
    info = pred.drop_duplicates("id").set_index("id")
    wide["total"] = wide[gws].sum(axis=1)
    club = ds.teams.set_index("id")["short_name"]
    for pos, label in POS_NAME.items():
        top = wide[info.loc[wide.index, "pos"] == pos].sort_values("total", ascending=False).head(args.top)
        print(f"\n{label}  " + " ".join(f"GW{g:<4}" for g in gws) + " total")
        for pid, r in top.iterrows():
            avail = info.at[pid, "avail_next"]
            flag = f"  avail {avail:.0%}" if avail < 1 else ""
            print(f"  {info.at[pid, 'web_name']:<18}{club.get(info.at[pid, 'team'], ''):<4} £{info.at[pid, 'price']:>4.1f}  "
                  + " ".join(f"{r[g]:>6.1f}" for g in gws) + f" {r['total']:>6.1f}{flag}")
    if args.csv:
        pred.to_csv(args.csv, index=False)
        print(f"\nSaved all predictions to {args.csv}")


def cmd_backtest(args) -> None:
    ds = load_archive(args.archive)
    summary, preds = run_backtest(ds, start_gw=args.start_gw, end_gw=args.end_gw, verbose=True)
    print("\nAverage per gameweek (top11 = actual points of the 11 highest-predicted players):")
    print(summary.round(3).to_string())
    if args.csv:
        preds.to_csv(args.csv, index=False)
        print(f"Saved per-player predictions to {args.csv}")


def cmd_download(args) -> None:
    import pathlib
    import requests
    base = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{args.season}"
    dest = pathlib.Path(args.dest) / args.season
    (dest / "gws").mkdir(parents=True, exist_ok=True)
    for f in ["players_raw.csv", "teams.csv", "fixtures.csv", "gws/merged_gw.csv"]:
        r = requests.get(f"{base}/{f}", timeout=60)
        r.raise_for_status()
        (dest / f).write_bytes(r.content)
        print(f"  {dest / f}  ({len(r.content) // 1024} KB)")
    print(f"Done. Use --archive {dest}")


def cmd_tick(args) -> None:
    from pathlib import Path
    from zoneinfo import ZoneInfo

    from .agent import AgentConfig, tick
    from .telegram import DryRunTelegram, Telegram, TelegramError

    team = args.team_id or os.environ.get("FPL_TEAM_ID")
    if not team or not str(team).strip().isdigit():
        sys.exit("Set FPL_TEAM_ID (or pass --team-id) to your FPL team id.")
    tz = args.tz or os.environ.get("FPL_TIMEZONE") or "UTC"
    try:
        ZoneInfo(tz)
    except Exception:
        sys.exit(f"Unknown time zone '{tz}'. Use a name like Asia/Dhaka or Europe/London.")
    root = Path(args.root)
    if args.dry_run:
        root = root / ".dryrun"   # keep test runs out of the real logs and state
    cfg = AgentConfig(int(str(team).strip()), tz, args.horizon, root)
    if not args.dry_run:
        missing = [v for v in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not os.environ.get(v)]
        if missing:
            sys.exit(f"Missing {', '.join(missing)}. Set them (GitHub: repository secrets), or use --dry-run.")
    try:
        tg = DryRunTelegram() if args.dry_run else Telegram(os.environ.get("TELEGRAM_BOT_TOKEN", ""),
                                                             os.environ.get("TELEGRAM_CHAT_ID"))
        now = None
        if args.now:
            now = pd.Timestamp(args.now)
            now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        code = tick(cfg, FPLClient(cache_dir=args.cache_dir), tg, now, force=args.force)
    except TelegramError as e:
        sys.exit(f"Telegram problem: {e}")
    sys.exit(code)


def cmd_telegram_setup(args) -> None:
    from .telegram import Telegram, TelegramError
    token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("Set TELEGRAM_BOT_TOKEN to the token from @BotFather (or pass --token).")
    try:
        tg = Telegram(token, None)
        chats = {}
        for u in tg.updates(0):
            c = (u.get("message") or {}).get("chat") or {}
            if c.get("type") == "private":
                chats[c["id"]] = c.get("first_name") or c.get("username") or ""
        if not chats:
            sys.exit("No messages found. Open your bot in Telegram, send it any message (e.g. 'hi'), "
                     "then run this again.")
        cid = list(chats)[-1]
        tg.send("✅ <b>fpl-agent is connected.</b> Plans will arrive here before each FPL deadline.",
                chat_id=str(cid))
    except TelegramError as e:
        sys.exit(f"Telegram problem: {e}")
    print(f"Found your chat: {chats[cid]}")
    print(f"TELEGRAM_CHAT_ID = {cid}")
    print("A test message was sent to your Telegram. Save this chat id as a GitHub secret.")


def main(argv=None) -> None:
    # On Windows, redirected output uses cp1252; don't crash on names like "Mitrović".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        _main(argv)
    except ValueError as e:      # e.g. an unknown player name
        sys.exit(f"Error: {e}")


def _main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fpl_agent", description="FPL predictor + transfer planner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data_args(p):
        p.add_argument("--archive", help="season folder in vaastav layout (offline/testing) instead of the live API")
        p.add_argument("--cache-dir", default=".fpl_cache")
        p.add_argument("--refresh", action="store_true", help="ignore cached API responses")
        p.add_argument("--access-token", help="FPL access token (or env FPL_ACCESS_TOKEN) for exact team state")
        p.add_argument("--horizon", type=int, default=5, help="gameweeks to plan over")
        p.add_argument("--out", help="comma-separated players you know will miss the horizon")
        p.add_argument("--starter", help="comma-separated players you know will start (e.g. back from injury)")

    p = sub.add_parser("plan", help="recommend transfers, XI and captain")
    data_args(p)
    p.add_argument("--team-id", type=int, help="your FPL team id (from the URL of your Points page)")
    p.add_argument("--squad-file", help="JSON file with squad/bank/free_transfers (offline use)")
    p.add_argument("--template", action="store_true", help="use the most-owned squad (testing)")
    p.add_argument("--from-scratch", action="store_true", help="ideal squad with £100m (wildcard view)")
    p.add_argument("--ft", type=int, help="override free transfers available")
    p.add_argument("--no-hits", action="store_true", help="never pay -4 for extra transfers")
    p.add_argument("--max-transfers", type=int, help="limit transfers this gameweek")
    p.add_argument("--lock", help="comma-separated players to keep/buy")
    p.add_argument("--ban", help="comma-separated players to avoid/sell")
    p.add_argument("--decay", type=float, default=0.85, help="weight of each later gameweek")
    p.add_argument("--time-limit", type=float, default=120.0)
    p.add_argument("--json", help="also save the plan as JSON (for the agent layer)")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("predict", help="show top players by expected points")
    data_args(p)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--csv", help="save all predictions to CSV")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("backtest", help="walk-forward backtest on an archived season")
    p.add_argument("--archive", required=True)
    p.add_argument("--start-gw", type=int, default=5)
    p.add_argument("--end-gw", type=int)
    p.add_argument("--csv")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("tick", help="hourly agent run: Telegram plans before deadlines + shadow log")
    p.add_argument("--team-id", help="FPL team id (or env FPL_TEAM_ID)")
    p.add_argument("--tz", help="time zone for messages, e.g. Asia/Dhaka (or env FPL_TIMEZONE)")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--root", default=".", help="folder that holds state/ and logs/")
    p.add_argument("--cache-dir", default=".fpl_cache")
    p.add_argument("--force", action="store_true", help="send a plan now, whatever the time")
    p.add_argument("--dry-run", action="store_true", help="print messages instead of sending (no Telegram needed)")
    p.add_argument("--now", help="pretend it's this UTC time (testing), e.g. 2026-11-06T12:00")
    p.set_defaults(func=cmd_tick)

    p = sub.add_parser("telegram-setup", help="find your Telegram chat id and send a test message")
    p.add_argument("--token", help="bot token (or env TELEGRAM_BOT_TOKEN)")
    p.set_defaults(func=cmd_telegram_setup)

    p = sub.add_parser("download-archive", help="fetch a past season (vaastav layout) for backtests/tests")
    p.add_argument("--season", default="2025-26")
    p.add_argument("--dest", default="data")
    p.set_defaults(func=cmd_download)

    args = ap.parse_args(argv)
    args.func(args)
