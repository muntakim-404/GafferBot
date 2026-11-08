"""Multi-gameweek squad and transfer optimiser (mixed-integer linear program).

Given expected points for each player over the next W gameweeks, choose transfers,
starting XI and captain for every week to maximise discounted expected points
minus hit costs, subject to all FPL squad rules. Only the first week's moves are
meant to be executed; re-run each gameweek with fresh predictions.

Simplifications (state these when trusting the output):
  * prices are assumed fixed over the horizon (no price-change modelling);
  * chips are not modelled (no Wildcard/Free Hit/Bench Boost/Triple Captain);
  * bench value is a flat weight, not an auto-sub probability model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import pulp

from .config import (GK, LINEUP_MAX, LINEUP_MIN, MAX_FREE_TRANSFERS, MAX_PER_CLUB,
                     SQUAD_COUNT, SolverParams, captain_score)
from .data import TeamState


@dataclass
class GWPlan:
    gw: int
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    free_transfers: int        # available before this week's moves
    bank_after: float
    squad: list[int]
    lineup: list[int]
    bench: list[int]           # in FPL order: backup GK first, then outfield by xPts
    captain: int
    vice: int
    xpts: float                # starting XI incl. captain double


@dataclass
class Plan:
    weeks: list[GWPlan]
    objective: float
    status: str
    from_scratch: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class SolverError(RuntimeError):
    pass


def _mip_solver(time_limit: float | None = None):
    """HiGHS when highspy is installed (faster), otherwise the CBC binary bundled with PuLP 2.x."""
    if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
        return pulp.HiGHS(msg=False, timeLimit=time_limit)
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)


def _candidates(xp: pd.DataFrame, info: pd.DataFrame, state: TeamState | None, prm: SolverParams,
                weights: np.ndarray) -> list[int]:
    """Prune to a manageable pool: current squad + locks + best and best-value players per position."""
    total = pd.Series(xp.to_numpy() @ weights, index=xp.index)
    keep = set(state.squad if state else []) | set(prm.lock)
    df = info.loc[xp.index].assign(total=total)
    for _, g in df.groupby("pos"):
        b = g[g["can_buy"] & ~g.index.isin(prm.ban)]
        keep |= set(b["total"].nlargest(prm.pool_per_position).index)
        keep |= set((b["total"] / b["price"]).nlargest(max(5, prm.pool_per_position // 3)).index)
        keep |= set(b.nsmallest(5, "price").index)    # guarantees cheap enablers for feasibility
    return sorted(keep)


def solve(xp: pd.DataFrame, players: pd.DataFrame, state: TeamState | None,
          prm: SolverParams | None = None) -> Plan:
    """xp: index = player id, columns = gameweeks (from predictor.xpts_matrix).
    state=None builds a squad from scratch (e.g. to see an ideal wildcard team)."""
    prm = prm or SolverParams()
    gws = sorted(xp.columns)
    W = len(gws)
    info = players.set_index("id")
    if state:
        missing = [p for p in state.squad if p not in xp.index]
        if missing:
            raise SolverError(f"No predictions for squad players {missing}")
    weights = np.array([prm.horizon_decay ** i for i in range(W)])
    P = _candidates(xp, info, state, prm, weights)
    xpv = xp.loc[P, gws]
    pos, club, price = info.loc[P, "pos"].to_dict(), info.loc[P, "team"].to_dict(), info.loc[P, "price"].to_dict()
    owned0 = set(state.squad) if state else set()
    sell = {p: (state.selling_price.get(p, price[p]) if p in owned0 else price[p]) for p in P}
    can_buy = {p: bool(info.at[p, "can_buy"]) and p not in prm.ban for p in P}

    prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)
    B = pulp.LpVariable.dicts
    sq = B("sq", (P, range(W)), cat="Binary")
    lu = B("lu", (P, range(W)), cat="Binary")
    cp = B("cp", (P, range(W)), cat="Binary")

    # Weeks in which transfers happen. From scratch, week 0 is a free build.
    T = list(range(W)) if state else list(range(1, W))
    tin = B("tin", (P, T), cat="Binary")
    tout = B("tout", (P, T), cat="Binary")
    ft = {w: pulp.LpVariable(f"ft_{w}", 0, MAX_FREE_TRANSFERS, cat="Integer") for w in T + [W]}
    hits = {w: pulp.LpVariable(f"hits_{w}", 0, None, cat="Integer") for w in T}
    if T:
        prob += ft[T[0]] == (min(state.free_transfers, MAX_FREE_TRANSFERS) if state else 1)
    elif W:
        prob += ft[W] == 1

    if state:
        base_bank = state.bank
    else:
        base_bank = prm.budget - pulp.lpSum(price[p] * sq[p][0] for p in P)
        prob += base_bank >= 0
        for p in P:
            if not can_buy[p]:
                prob += sq[p][0] == 0

    bank = {}
    running = base_bank
    for w in T:
        for p in P:
            prev = (1 if p in owned0 else 0) if w == 0 else sq[p][w - 1]
            prob += sq[p][w] == prev + tin[p][w] - tout[p][w]
            prob += tin[p][w] + tout[p][w] <= 1
            if not can_buy[p]:
                prob += tin[p][w] == 0
        n = pulp.lpSum(tin[p][w] for p in P)
        prob += hits[w] >= n - ft[w]
        prob += hits[w] <= n
        nxt = T[T.index(w) + 1] if T.index(w) + 1 < len(T) else W
        prob += ft[nxt] <= ft[w] - n + hits[w] + 1
        if not prm.allow_hits:
            prob += hits[w] == 0
        if prm.max_transfers_per_gw is not None:
            prob += n <= prm.max_transfers_per_gw
        if w == T[0] and prm.max_transfers_first_gw is not None:
            prob += n <= prm.max_transfers_first_gw
        running = running + pulp.lpSum(sell[p] * tout[p][w] - price[p] * tin[p][w] for p in P)
        bank[w] = running
        prob += running >= 0

    for w in range(W):
        for k, cnt in SQUAD_COUNT.items():
            prob += pulp.lpSum(sq[p][w] for p in P if pos[p] == k) == cnt
            prob += pulp.lpSum(lu[p][w] for p in P if pos[p] == k) >= LINEUP_MIN[k]
            prob += pulp.lpSum(lu[p][w] for p in P if pos[p] == k) <= LINEUP_MAX[k]
        for c in set(club.values()):
            prob += pulp.lpSum(sq[p][w] for p in P if club[p] == c) <= MAX_PER_CLUB
        prob += pulp.lpSum(lu[p][w] for p in P) == 11
        prob += pulp.lpSum(cp[p][w] for p in P) == 1
        for p in P:
            prob += lu[p][w] <= sq[p][w]
            prob += cp[p][w] <= lu[p][w]
            if captain_score(pos[p], 0.0) is None:     # only MID/FWD may captain
                prob += cp[p][w] == 0
            if p in prm.lock:
                prob += sq[p][w] == 1
            if p in prm.ban:
                prob += sq[p][w] == 0

    def cap_val(p, gw) -> float:
        s = captain_score(pos[p], float(xpv.at[p, gw]))
        return 0.0 if s is None else s

    # XI and bench at raw xPts; the captain's extra points at the bias-corrected captain score.
    value = pulp.lpSum(
        weights[w] * pulp.lpSum(float(xpv.at[p, gws[w]]) * (lu[p][w] + prm.bench_weight * (sq[p][w] - lu[p][w]))
                                + cap_val(p, gws[w]) * cp[p][w] for p in P)
        for w in range(W))
    cost = pulp.lpSum(weights[w] * (prm.hit_cost + prm.hit_margin) * hits[w] for w in T)
    prob += value - cost + prm.ft_value * ft[W]

    prob.solve(_mip_solver(prm.time_limit))
    status = pulp.LpStatus[prob.status]
    sol = getattr(prob, "sol_status", prob.status)
    if sol == pulp.LpSolutionInfeasible or status == "Infeasible":
        raise SolverError("No legal plan satisfies all constraints. Usually a lock/ban needs more transfers "
                          "than allowed (e.g. with --no-hits or --max-transfers), or the budget is too tight.")
    if sol not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        raise SolverError(f"Solver found no solution (status {status}); try a longer --time-limit.")
    if sol == pulp.LpSolutionIntegerFeasible:
        status = "Feasible (time limit hit before proving optimal)"

    def on(v) -> bool:
        return (v.value() or 0) > 0.5

    weeks = []
    for w, gw in enumerate(gws):
        squad = [p for p in P if on(sq[p][w])]
        lineup = sorted((p for p in squad if on(lu[p][w])), key=lambda p: -xpv.at[p, gw])
        cap = next(p for p in lineup if on(cp[p][w]))
        eligible = sorted((p for p in lineup if p != cap and captain_score(pos[p], 0.0) is not None),
                          key=lambda p: -cap_val(p, gw))
        vice = eligible[0] if eligible else next((p for p in lineup if p != cap), cap)
        bench_pl = [p for p in squad if p not in lineup]
        bench = sorted(bench_pl, key=lambda p: (pos[p] != GK, -xpv.at[p, gw]))
        if w in T:
            t_in = [p for p in P if on(tin[p][w])]
            t_out = [p for p in P if on(tout[p][w])]
            h, f = int(round(hits[w].value())), int(round(ft[w].value()))
            bk = float(pulp.value(bank[w]))
        else:
            t_in, t_out, h, f = squad, [], 0, 0
            bk = float(pulp.value(base_bank))
        weeks.append(GWPlan(int(gw), t_in, t_out, h, f, round(bk, 1), squad, lineup, bench, cap, vice,
                            round(float(sum(xpv.at[p, gw] for p in lineup) + xpv.at[cap, gw]), 2)))
    notes = [] if status == "Optimal" else [f"Solver status: {status}. The plan is legal but may be sub-optimal."]
    return Plan(weeks, round(float(pulp.value(prob.objective)), 2), status, state is None, notes)


# --------------------------------------------------------------------- validation
def validate(plan: Plan, players: pd.DataFrame, state: TeamState | None, budget: float = 100.0) -> list[str]:
    """Independent re-check of every FPL rule on a finished plan. Returns a list of problems."""
    info = players.set_index("id")
    errs = []
    prev = set(state.squad) if state else None
    bank = state.bank if state else None
    for wk in plan.weeks:
        sq = set(wk.squad)
        tag = f"GW{wk.gw}"
        if len(sq) != 15:
            errs.append(f"{tag}: squad has {len(sq)} players")
        for k, cnt in SQUAD_COUNT.items():
            got = sum(info.at[p, "pos"] == k for p in sq)
            if got != cnt:
                errs.append(f"{tag}: {got} players in position {k}, need {cnt}")
        clubs = pd.Series([info.at[p, "team"] for p in sq]).value_counts()
        if clubs.max() > MAX_PER_CLUB:
            errs.append(f"{tag}: more than {MAX_PER_CLUB} from one club")
        lu = set(wk.lineup)
        if len(lu) != 11 or not lu <= sq:
            errs.append(f"{tag}: lineup invalid")
        for k in SQUAD_COUNT:
            got = sum(info.at[p, "pos"] == k for p in lu)
            if not LINEUP_MIN[k] <= got <= LINEUP_MAX[k]:
                errs.append(f"{tag}: formation breaks limits for position {k}")
        if wk.captain not in lu or wk.vice not in lu:
            errs.append(f"{tag}: captain/vice not in lineup")
        if captain_score(int(info.at[wk.captain, "pos"]), 0.0) is None:
            errs.append(f"{tag}: captain must be a midfielder or forward")
        if set(wk.bench) != sq - lu or (wk.bench and info.at[wk.bench[0], "pos"] != GK):
            errs.append(f"{tag}: bench invalid")
        if prev is None:   # from-scratch first week
            cost = sum(info.at[p, "price"] for p in sq)
            bank = budget - cost
        else:
            if sq != (prev - set(wk.transfers_out)) | set(wk.transfers_in):
                errs.append(f"{tag}: transfers don't explain squad change")
            for p in wk.transfers_out:
                bank += state.selling_price.get(p, info.at[p, "price"]) if state and p in state.squad else info.at[p, "price"]
            for p in wk.transfers_in:
                bank -= info.at[p, "price"]
            paid = max(0, len(wk.transfers_in) - wk.free_transfers)
            if wk.hits != paid:
                errs.append(f"{tag}: hits {wk.hits} but {paid} paid transfers")
        if bank < -1e-6:
            errs.append(f"{tag}: bank negative ({bank:.1f})")
        prev = sq
    return errs


# ------------------------------------------------------------------- utilities
def most_owned_squad(players: pd.DataFrame, budget: float = 100.0) -> list[int]:
    """The highest-ownership legal squad. Useful as a realistic test squad."""
    df = players[players["can_buy"]].set_index("id")
    prob = pulp.LpProblem("template", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", df.index, cat="Binary")
    prob += pulp.lpSum(df.at[p, "selected_by_percent"] * x[p] for p in df.index)
    prob += pulp.lpSum(df.at[p, "price"] * x[p] for p in df.index) <= budget
    for k, cnt in SQUAD_COUNT.items():
        prob += pulp.lpSum(x[p] for p in df.index if df.at[p, "pos"] == k) == cnt
    for c in df["team"].unique():
        prob += pulp.lpSum(x[p] for p in df.index if df.at[p, "team"] == c) <= MAX_PER_CLUB
    prob.solve(_mip_solver(60))
    return [p for p in df.index if (x[p].value() or 0) > 0.5]
