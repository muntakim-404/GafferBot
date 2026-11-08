"""Chip planner tests."""
import os

import numpy as np
import pandas as pd
import pytest

from fpl_agent.chips import assign_chips, best_xi, chip_outlook, format_chips, remaining_chips
from fpl_agent.config import DEF, FWD, GK, MID
from test_solver import make_players

ARCHIVE = os.environ.get("FPL_ARCHIVE")
needs_2026 = pytest.mark.skipif(not ARCHIVE or "2026-27" not in ARCHIVE, reason="set FPL_ARCHIVE to the 2026-27 folder")


def test_remaining_chips_fallback_and_used():
    assert remaining_chips(11, []) == {"3xc": 19, "bboost": 19, "freehit": 19}
    used = [{"name": "wildcard", "event": 4}, {"name": "3xc", "event": 8}]
    assert remaining_chips(11, used) == {"bboost": 19, "freehit": 19}
    # second half: first-half usage doesn't count, everything refreshed
    assert remaining_chips(21, used) == {"3xc": 38, "bboost": 38, "freehit": 38}
    assert remaining_chips(25, [{"name": "bboost", "event": 22}]) == {"3xc": 38, "freehit": 38}


def test_remaining_chips_uses_fpl_windows_when_given():
    boot = [{"name": "3xc", "start_event": 1, "stop_event": 18}, {"name": "3xc", "start_event": 19, "stop_event": 38},
            {"name": "bboost", "start_event": 1, "stop_event": 18}, {"name": "bboost", "start_event": 19, "stop_event": 38},
            {"name": "freehit", "start_event": 2, "stop_event": 18}, {"name": "freehit", "start_event": 19, "stop_event": 38}]
    assert remaining_chips(11, [], boot) == {"3xc": 18, "bboost": 18, "freehit": 18}
    assert remaining_chips(19, [{"name": "3xc", "event": 5}], boot) == {"3xc": 38, "bboost": 38, "freehit": 38}


def _squad():
    pos = {}
    pid = 1
    squad = []
    for k, n in ((GK, 2), (DEF, 5), (MID, 5), (FWD, 3)):
        for _ in range(n):
            pos[pid] = k
            squad.append(pid)
            pid += 1
    return squad, pos


def test_best_xi_is_legal_and_picks_best_formation():
    squad, pos = _squad()
    xp = {p: 1.0 for p in squad}
    xp.update({3: 9, 4: 8, 5: 7, 6: 6, 7: 5})        # all five defenders are strong -> 5 at the back
    xp.update({8: 4.0})                               # best midfielder
    xi, cap, pts = best_xi(squad, pos, xp)
    c = {k: sum(pos[p] == k for p in xi) for k in (GK, DEF, MID, FWD)}
    assert len(xi) == 11 and c[GK] == 1 and c[DEF] == 5 and c[MID] >= 2 and c[FWD] >= 1
    # defenders can't captain: the best midfielder/forward gets the armband
    assert cap == 8 and pts == pytest.approx(sum(xp[p] for p in xi) + 4.0)


def test_captain_prefers_midfielder_when_forward_is_only_slightly_higher():
    squad, pos = _squad()
    xp = {p: 1.0 for p in squad}
    xp.update({8: 7.6, 13: 8.0})                      # MID 7.6 vs FWD 8.0: forwards' forecasts run hot
    _, cap, _ = best_xi(squad, pos, xp)
    assert cap == 8                                   # 7.10 corrected vs 6.79


def test_assign_chips_one_per_week_and_respects_last_week():
    v = pd.DataFrame({"tc": [5, 9, 5], "bb": [10, 12, 9], "fh": [1, 1, 7]}, index=[10, 11, 12])
    best = assign_chips(v, {"3xc": 19, "bboost": 19, "freehit": 19})
    assert len(set(best.values())) == 3
    assert best == {"3xc": 11, "bboost": 10, "freehit": 12}    # 9+10+7 beats 5+12+7
    assert assign_chips(v, {"3xc": 10})["3xc"] == 10           # can't use a week after its window


def _outlook(boost_gw, next_gw=10, chips=None):
    pl = make_players()
    gws = [10, 11, 12, 13]
    xp = pd.DataFrame({g: 2.0 for g in gws}, index=pl["id"])
    squad = []
    for k, n in ((GK, 2), (DEF, 5), (MID, 5), (FWD, 3)):
        squad += pl[pl.pos == k]["id"].head(n).tolist()
    star = squad[12]                                  # a forward
    xp.loc[star] = 6.0
    if boost_gw:
        xp.loc[star, boost_gw] = 12.0                # e.g. a double gameweek
    chips = chips or {"3xc": 19, "bboost": 19, "freehit": 19}
    return chip_outlook(xp, pl, {next_gw: squad}, 100.0, next_gw, chips, with_free_hit=False), star


def test_no_standout_means_hold():
    o, _ = _outlook(None)
    assert not o["detail"]["3xc"]["standout"] and o["play_now"] == []
    assert any("hold" in line for line in format_chips(o, {}))


def test_standout_later_week_is_named_but_not_played_now():
    o, star = _outlook(12)
    d = o["detail"]["3xc"]
    assert d["gw"] == 12 and d["standout"] and d["margin"] == pytest.approx(0.763 * (12.0 - 6.0), abs=0.01)
    assert o["play_now"] == []


def test_standout_this_week_says_play_now():
    o, _ = _outlook(10)
    assert o["play_now"] == ["3xc"]
    assert any("Triple Captain" in line and "this week" in line for line in format_chips(o, {}))


def test_running_out_of_weeks_forces_a_chip():
    o, _ = _outlook(None, next_gw=11, chips={"3xc": 13, "bboost": 13, "freehit": 13})
    assert o["forced"] and len(o["play_now"]) == 1


@needs_2026
def test_double_gameweek_makes_triple_captain_stand_out():
    """Give Man City a second fixture in GW14: Haaland's Triple Captain week should be GW14."""
    from fpl_agent.data import load_archive
    from fpl_agent.predictor import predict, xpts_matrix
    from fpl_agent.solver import most_owned_squad

    ds = load_archive(ARCHIVE)
    mci = int(ds.teams.loc[ds.teams["short_name"] == "MCI", "id"].iloc[0])
    fx = ds.fixtures
    other = int(fx[(fx["event"] == 14) & (fx["team_h"] != mci) & (fx["team_a"] != mci)]["team_a"].iloc[0])
    extra = pd.DataFrame({"id": [99999], "event": pd.array([14], dtype="Int64"), "team_h": [mci],
                          "team_a": [other], "finished": [False], "kickoff_time": [fx["kickoff_time"].max()]})
    ds.fixtures = pd.concat([fx, extra], ignore_index=True)
    squad = most_owned_squad(ds.players)
    haaland = int(ds.players.loc[ds.players["web_name"] == "Haaland", "id"].iloc[0])
    assert haaland in squad
    gws = list(range(ds.next_gw, 20))
    xp = xpts_matrix(predict(ds, gws))
    o = chip_outlook(xp, ds.players, {ds.next_gw: squad}, 100.0, ds.next_gw,
                     {"3xc": 19, "bboost": 19, "freehit": 19}, with_free_hit=False)
    d = o["detail"]["3xc"]
    assert d["gw"] == 14 and d["standout"] and o["values"].at[14, "captain"] == haaland
    # every chip's "next best" excludes weeks other chips are planned for, so it never exceeds the best
    assert all(x["runner_up"] <= x["value"] + 1e-9 for x in o["detail"].values())
