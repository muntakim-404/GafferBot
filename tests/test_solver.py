"""Solver tests on small synthetic player pools where the right answer is known."""
import numpy as np
import pandas as pd
import pytest

from fpl_agent.config import SolverParams
from fpl_agent.data import TeamState
from fpl_agent.solver import solve, validate


def make_players(seed=0, n_teams=20, per_team=(2, 5, 5, 3)):
    rng = np.random.default_rng(seed)
    rows, pid = [], 1
    for t in range(1, n_teams + 1):
        for pos, k in enumerate(per_team, start=1):
            for _ in range(k):
                price = {1: 4.5, 2: 4.5, 3: 5.5, 4: 6.0}[pos] + round(float(rng.uniform(0, 3)), 1)
                rows.append(dict(id=pid, web_name=f"P{pid}", team=t, pos=pos, price=price, can_buy=True))
                pid += 1
    return pd.DataFrame(rows)


def make_xp(players, gws, seed=1):
    rng = np.random.default_rng(seed)
    base = 1.0 + players["price"].to_numpy() * 0.4 + rng.normal(0, 0.5, len(players))
    return pd.DataFrame({g: np.clip(base + rng.normal(0, 0.3, len(players)), 0, None) for g in gws},
                        index=players["id"])


def cheap_state(players, ft=1):
    """A legal, cheap squad: the cheapest players per position, one club at a time."""
    squad = []
    for pos, k in {1: 2, 2: 5, 3: 5, 4: 3}.items():
        cand = players[players.pos == pos].sort_values("price")
        picked, clubs = [], {}
        for r in cand.itertuples():
            if clubs.get(r.team, 0) + sum(1 for p in squad if players.set_index("id").at[p, "team"] == r.team) < 3:
                picked.append(r.id)
                clubs[r.team] = clubs.get(r.team, 0) + 1
            if len(picked) == k:
                break
        squad += picked
    price = players.set_index("id")["price"]
    return TeamState(squad, {p: float(price[p]) for p in squad}, round(100 - price[squad].sum(), 1), ft, 1)


def test_from_scratch_is_valid_and_within_budget():
    pl = make_players()
    xp = make_xp(pl, [1, 2, 3])
    plan = solve(xp, pl, None, SolverParams(time_limit=60))
    assert validate(plan, pl, None) == []
    assert sum(pl.set_index("id").loc[plan.weeks[0].squad, "price"]) <= 100.0 + 1e-6


def test_transfer_plan_is_valid_with_selling_prices():
    pl = make_players()
    xp = make_xp(pl, [1, 2, 3, 4])
    st = cheap_state(pl, ft=2)
    # Selling price below current price for one owned player: the bank must use it.
    st.selling_price[st.squad[0]] -= 0.3
    plan = solve(xp, pl, st, SolverParams(time_limit=60))
    assert validate(plan, pl, st) == []


def test_big_gain_takes_a_hit_small_gain_does_not():
    pl = make_players()
    st = cheap_state(pl, ft=0)
    gws = [1, 2, 3]
    xp = pd.DataFrame(2.0, index=pl["id"], columns=gws)
    dud = next(p for p in st.squad if pl.set_index("id").at[p, "pos"] == 3)
    star = next(p for p in pl[pl.pos == 3]["id"] if p not in st.squad)
    xp.loc[dud] = 0.0
    xp.loc[star] = 12.0   # gain ~10/week over 3 weeks >> 4-point hit
    plan = solve(xp, pl, st, SolverParams(time_limit=60))
    assert star in plan.weeks[0].transfers_in and plan.weeks[0].hits == 1
    xp.loc[star] = 2.0 + 0.0
    xp.loc[dud] = 1.0     # gain ~1/week: not worth -4 (+1 margin)
    plan = solve(xp, pl, st, SolverParams(time_limit=60))
    assert plan.weeks[0].hits == 0


def test_rolls_transfer_when_nothing_is_worth_it():
    pl = make_players()
    st = cheap_state(pl, ft=1)
    xp = pd.DataFrame(2.0, index=pl["id"], columns=[1, 2])
    xp.loc[st.squad] = 2.0
    plan = solve(xp, pl, st, SolverParams(time_limit=60))
    assert plan.weeks[0].transfers_in == []
    assert plan.weeks[1].free_transfers == 2


def test_free_transfers_cap_at_five():
    pl = make_players()
    st = cheap_state(pl, ft=5)
    xp = pd.DataFrame(2.0, index=pl["id"], columns=[1, 2, 3])
    plan = solve(xp, pl, st, SolverParams(time_limit=60, ft_value=0.0))
    assert all(w.free_transfers <= 5 for w in plan.weeks)


def test_lock_ban_and_no_hits():
    pl = make_players()
    xp = make_xp(pl, [1, 2])
    st = cheap_state(pl, ft=1)
    top = xp[1].sort_values(ascending=False)
    banned = int(top.index[0])
    locked = int(next(p for p in top.index[5:] if p not in st.squad))
    plan = solve(xp, pl, st, SolverParams(time_limit=60, lock=[locked], ban=[banned], allow_hits=False))
    assert validate(plan, pl, st) == []
    for wk in plan.weeks:
        assert locked in wk.squad and banned not in wk.squad and wk.hits == 0


def test_three_per_club_even_when_one_club_dominates():
    pl = make_players()
    xp = make_xp(pl, [1])
    xp.loc[pl[pl.team == 1]["id"]] = 50.0
    plan = solve(xp, pl, None, SolverParams(time_limit=60))
    clubs = pl.set_index("id").loc[plan.weeks[0].squad, "team"]
    assert (clubs == 1).sum() == 3
    assert validate(plan, pl, None) == []


def test_captain_is_never_a_defender_or_keeper():
    pl = make_players()
    xp = make_xp(pl, [1, 2])
    defs = pl[pl.pos == 2]["id"].head(3)
    xp.loc[defs] = 20.0                       # defenders with absurd predictions
    plan = solve(xp, pl, None, SolverParams(time_limit=60))
    info = pl.set_index("id")
    for wk in plan.weeks:
        assert info.at[wk.captain, "pos"] in (3, 4) and info.at[wk.vice, "pos"] in (3, 4)
        assert any(d in wk.lineup for d in defs)          # ...but they still start
    assert validate(plan, pl, None) == []
    plan.weeks[0].captain = int(next(d for d in defs if d in plan.weeks[0].lineup))
    assert any("captain must be" in e for e in validate(plan, pl, None))


def test_validate_catches_a_broken_plan():
    pl = make_players()
    plan = solve(make_xp(pl, [1]), pl, None, SolverParams(time_limit=60))
    plan.weeks[0].squad = plan.weeks[0].squad[:-1]
    assert validate(plan, pl, None)
