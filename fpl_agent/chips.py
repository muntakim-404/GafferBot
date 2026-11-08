"""Chip planner: when to play Triple Captain, Bench Boost and Free Hit.

For every remaining gameweek in the current chip window (first half: up to
GW19, second half: up to GW38) it estimates, from the model's predictions:

  * Triple Captain: the captain's bias-corrected score (the extra third);
  * Bench Boost:    expected points of your 4 bench players;
  * Free Hit:       best one-week squad you could afford, minus your own XI;

using the squad the transfer plan expects you to own that week (beyond the
plan's horizon, the last planned squad). It then picks one week per remaining
chip, never two chips in one week (FPL allows one chip per gameweek), to
maximise the total.

Treat the values as a guide: they inherit the model's weaknesses (captaincy is
the weakest call) and weeks far ahead are re-estimated every week as fixtures,
injuries and doubles change.
"""
from __future__ import annotations

from itertools import permutations

import pandas as pd

from .config import DEF, FWD, GK, MID, SolverParams, captain_score
from .solver import solve

CHIP_LABEL = {"3xc": "Triple Captain", "bboost": "Bench Boost", "freehit": "Free Hit"}
VALUE_COL = {"3xc": "tc", "bboost": "bb", "freehit": "fh"}
# A week "stands out" only if it beats every other usable week by this much
# (expected points). Below that, the ranking is within the model's error, and
# waiting keeps the option of a double gameweek or better news. (guesses)
STANDOUT_MARGIN = {"3xc": 1.5, "bboost": 2.0, "freehit": 2.0}
FORMATIONS = [(d, m, f) for d in (3, 4, 5) for m in (2, 3, 4, 5) for f in (1, 2, 3) if d + m + f == 10]


# ------------------------------------------------------------------ chip windows
def _fallback_window(name: str, gw: int) -> tuple[int, int]:
    if gw <= 19:
        return (2 if name == "freehit" else 1), 19     # no Free Hit in GW1
    return 20, 38


def remaining_chips(next_gw: int, used: list[dict], bootstrap_chips: list[dict] | None = None) -> dict:
    """{chip: last gameweek it can be played} for chips not yet used in the current window.

    Windows come from FPL's bootstrap 'chips' list (start_event/stop_event) when
    present, otherwise the 2026/27 rule: first set up to GW19, second from GW20.
    `used` is the 'chips' list from entry/{id}/history ([{"name", "event"}, ...]).
    """
    out = {}
    for name in CHIP_LABEL:
        window = None
        for c in bootstrap_chips or []:
            s, e = c.get("start_event"), c.get("stop_event")
            if c.get("name") == name and s and e and s <= next_gw <= e:
                window = (int(s), int(e))
                break
        if window is None:
            window = _fallback_window(name, next_gw)
        start, stop = window
        if any(u.get("name") == name and start <= int(u.get("event") or 0) <= stop for u in used):
            continue
        if next_gw <= stop:
            out[name] = stop
    return out


# --------------------------------------------------------------------- lineups
def best_xi(squad: list[int], pos: dict, xp: dict) -> tuple[list[int], int, float]:
    """Best legal XI for one gameweek. Returns (xi, captain, points incl. captain double)."""
    by = {k: sorted((p for p in squad if pos[p] == k), key=lambda p: -xp.get(p, 0.0)) for k in (GK, DEF, MID, FWD)}
    best = None
    for d, m, f in FORMATIONS:
        if len(by[GK]) < 1 or len(by[DEF]) < d or len(by[MID]) < m or len(by[FWD]) < f:
            continue
        xi = by[GK][:1] + by[DEF][:d] + by[MID][:m] + by[FWD][:f]
        s = sum(xp.get(p, 0.0) for p in xi)
        if best is None or s > best[0]:
            best = (s, xi)
    if best is None:
        raise ValueError("Squad can't field a legal XI.")
    xi = best[1]
    eligible = [p for p in xi if captain_score(pos[p], 0.0) is not None]   # MID/FWD only
    cap = max(eligible, key=lambda p: captain_score(pos[p], xp.get(p, 0.0)))
    return xi, cap, best[0] + xp.get(cap, 0.0)


# ---------------------------------------------------------------------- values
def chip_values(xp: pd.DataFrame, players: pd.DataFrame, squads: dict[int, list[int]], budget: float,
                with_free_hit: bool = True, time_limit: float = 30.0) -> pd.DataFrame:
    """One row per gameweek: captain, tc, bb, fh (expected extra points from each chip).

    xp: index = player id, columns = gameweeks. squads: gameweek -> your 15 that week.
    budget: what a Free Hit squad may cost (your squad's selling value + bank).
    """
    pos = players.set_index("id")["pos"].to_dict()
    rows = []
    for gw in sorted(squads):
        col = xp[gw].to_dict()
        squad = squads[gw]
        xi, cap, xi_pts = best_xi(squad, pos, col)
        row = {"gw": gw, "captain": cap, "tc": captain_score(pos[cap], col.get(cap, 0.0)),
               "bb": sum(col.get(p, 0.0) for p in squad if p not in xi), "fh": 0.0, "xi_pts": xi_pts}
        if with_free_hit:
            fh = solve(xp[[gw]], players, None,
                       SolverParams(budget=budget, bench_weight=0.0, time_limit=time_limit, pool_per_position=40))
            row["fh"] = max(0.0, fh.weeks[0].xpts - xi_pts)
        rows.append(row)
    return pd.DataFrame(rows).set_index("gw")


def assign_chips(values: pd.DataFrame, chips: dict) -> dict:
    """Best week per chip, one chip per week. chips: {name: last usable gw}. Returns {name: gw}."""
    col = {"3xc": "tc", "bboost": "bb", "freehit": "fh"}
    names = [c for c in CHIP_LABEL if c in chips]
    weeks = list(values.index)
    best, best_total = {}, None
    for combo in permutations(weeks, len(names)):
        if any(gw > chips[n] for n, gw in zip(names, combo)):
            continue
        total = sum(values.at[gw, col[n]] for n, gw in zip(names, combo))
        if best_total is None or total > best_total + 1e-9:
            best, best_total = dict(zip(names, combo)), total
    return best


def chip_outlook(xp: pd.DataFrame, players: pd.DataFrame, plan_squads: dict[int, list[int]],
                 budget: float, next_gw: int, chips: dict, with_free_hit: bool = True) -> dict | None:
    """Everything the message and CLI need. None when no chips are left."""
    if not chips:
        return None
    last = max(chips.values())
    gws = [g for g in sorted(xp.columns) if next_gw <= g <= last]
    planned = sorted(plan_squads)
    squads = {}
    for g in gws:   # squad you'd own that week: planned squad, or the last planned one beyond the horizon
        known = [p for p in planned if p <= g]
        squads[g] = plan_squads[known[-1] if known else planned[0]]
    vals = chip_values(xp, players, squads, budget, with_free_hit and "freehit" in chips)
    best = assign_chips(vals, chips)
    detail = {}
    for c, gw in best.items():
        taken = {g for o, g in best.items() if o != c}      # weeks the other chips are planned for
        usable = vals.loc[[g for g in vals.index if g <= chips[c] and g not in taken], VALUE_COL[c]]
        others = usable.drop(gw)
        runner = float(others.max()) if len(others) else 0.0
        margin = float(usable[gw]) - runner
        detail[c] = {"gw": gw, "value": float(usable[gw]), "runner_up": runner, "margin": margin,
                     "standout": margin >= STANDOUT_MARGIN[c]}
    forced = len(gws) <= len(chips)     # every remaining week needs a chip
    play_now = [c for c, d in detail.items() if d["gw"] == next_gw and (d["standout"] or forced)]
    return {"chips": chips, "values": vals, "best": best, "detail": detail, "next_gw": next_gw,
            "last_gw": last, "weeks_left": len(gws), "forced": forced, "play_now": play_now,
            "free_hit_checked": with_free_hit and "freehit" in chips}


# ------------------------------------------------------------------- display
def format_chips(o: dict, names: dict, html_escape=lambda s: s) -> list[str]:
    """Lines for the Telegram message."""
    if not o:
        return []
    e = html_escape
    v, nxt = o["values"], o["next_gw"]
    left = len(o["chips"])
    lines = [f"<b>Chips</b>: {left} left, use by GW{o['last_gw']} ({o['weeks_left']} GWs, one chip per GW)"]
    if o["forced"]:
        lines.append(f"⚠️ Only {o['weeks_left']} gameweek(s) left for {left} chip(s): one must be played every week now.")
    elif o["weeks_left"] <= left + 2:
        lines.append(f"⚠️ Only {o['weeks_left']} gameweeks left for {left} chip(s). Don't let them expire.")
    for c in CHIP_LABEL:
        d = o["detail"].get(c)
        if d is None:
            continue
        if c == "freehit" and not o.get("free_hit_checked", True):
            continue
        who = f" with {e(names.get(v.at[d['gw'], 'captain'], v.at[d['gw'], 'captain']))}" if c == "3xc" else ""
        if d["standout"]:
            lines.append(f"• {CHIP_LABEL[c]}: GW{d['gw']} stands out (+{d['value']:.1f}{who}, vs +{d['runner_up']:.1f} next best)")
        else:
            lines.append(f"• {CHIP_LABEL[c]}: hold, no week stands out yet (best GW{d['gw']} +{d['value']:.1f}{who}, "
                         f"next +{d['runner_up']:.1f})")
    if o["play_now"]:   # at most one: assign_chips never puts two chips in the same week
        why = "time is running out" if o["forced"] and not o["detail"][o["play_now"][0]]["standout"] else "this week stands out"
        lines.append(f"▶ Consider playing <b>{CHIP_LABEL[o['play_now'][0]]}</b> this week ({why}).")
    return lines
