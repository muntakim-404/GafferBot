"""Data-layer tests.

The live API can't be reached from every machine, so `FakeClient` serves real
archived data in the exact JSON shapes the API returns. If the live loader and
the archive loader produce the same canonical history, the live path is right
up to the HTTP call itself.

Set FPL_ARCHIVE to a season folder (vaastav layout) to run the archive tests.
"""
import os

import numpy as np
import pandas as pd
import pytest

from fpl_agent.data import (FPLClient, estimate_free_transfers, load_archive, load_live, selling_price,
                            team_state_from_my_team, team_state_public)
from fpl_agent.predictor import predict

ARCHIVE = os.environ.get("FPL_ARCHIVE")
needs_archive = pytest.mark.skipif(not ARCHIVE, reason="set FPL_ARCHIVE to a season folder")


def test_selling_price_rules():
    assert selling_price(60, 63) == 61     # keep half the rise, rounded down
    assert selling_price(60, 64) == 62
    assert selling_price(60, 61) == 60
    assert selling_price(60, 58) == 58     # falls passed on in full


def test_free_transfer_replay():
    hist = [{"event": 2, "event_transfers": 0}, {"event": 3, "event_transfers": 1},
            {"event": 4, "event_transfers": 9}, {"event": 5, "event_transfers": 3}]
    # GW2 roll -> 2 for GW3; GW3 use 1 -> 2 for GW4; GW4 wildcard keeps 2 (no +1) -> 2; GW5 use 3 (one hit) -> 1
    assert estimate_free_transfers(hist, {4: "wildcard"}, started=1, upto_gw=5) == 1
    assert estimate_free_transfers(hist[:3], {4: "wildcard"}, started=1, upto_gw=4) == 2
    assert estimate_free_transfers([{"event": g, "event_transfers": 0} for g in range(2, 12)], {}, 1, 11) == 5


def test_chip_keeps_count_without_plus_one():
    # 1 FT going into a Free Hit week -> still 1 the week after (the case that over-counted before).
    hist = [{"event": g, "event_transfers": 1} for g in range(2, 10)]
    hist[3]["event_transfers"] = 15          # GW5 Free Hit
    assert estimate_free_transfers(hist, {5: "freehit"}, started=1, upto_gw=9) == 1
    assert estimate_free_transfers(hist, {5: "wildcard"}, started=1, upto_gw=9) == 1


def test_my_team_parsing():
    js = {"picks": [{"element": i, "selling_price": 50 + i} for i in range(1, 16)],
          "transfers": {"bank": 7, "limit": 3, "made": 1}}
    st = team_state_from_my_team(js, next_gw=12)
    assert st.free_transfers == 2 and st.bank == 0.7 and st.selling_price[1] == 5.1


class FakeClient(FPLClient):
    def __init__(self, season_dir):
        self.access_token = None
        self.raw = pd.read_csv(f"{season_dir}/gws/merged_gw.csv", low_memory=False)
        self.players = pd.read_csv(f"{season_dir}/players_raw.csv")
        self.teams = pd.read_csv(f"{season_dir}/teams.csv")
        self.fx = pd.read_csv(f"{season_dir}/fixtures.csv")
        self.last = int(self.raw["round"].max())
        self.squad = []

    @staticmethod
    def _records(df):
        return [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                for r in df.to_dict("records")]

    def bootstrap(self, max_age=None):
        deadlines = self.fx.groupby("event")["kickoff_time"].min()
        events = [{"id": g, "deadline_time": deadlines.get(g), "finished": g <= self.last,
                   "is_next": g == self.last + 1, "is_current": g == self.last} for g in range(1, 39)]
        return {"elements": self._records(self.players), "teams": self._records(self.teams), "events": events}

    def fixtures(self, max_age=None):
        return self._records(self.fx)

    def element_summary(self, pid, max_age=None):
        rows = self.raw[self.raw["element"] == pid].drop(columns=["name", "position", "team", "xP", "GW"])
        return {"history": self._records(rows)}

    def entry(self, eid, max_age=None):
        return {"started_event": 1, "current_event": self.last}

    def entry_history(self, eid, max_age=None):
        return {"current": [{"event": g, "event_transfers": 1 if g == 3 else 0} for g in range(1, self.last + 1)],
                "chips": []}

    def entry_transfers(self, eid, max_age=None):
        return [{"element_in": self.squad[0], "element_in_cost": 50, "element_out": 999, "element_out_cost": 45,
                 "event": 3, "time": "2026-09-01T10:00:00Z"}]

    def picks(self, eid, gw, max_age=None):
        return {"picks": [{"element": p} for p in self.squad], "entry_history": {"bank": 12}, "active_chip": None}


@needs_archive
def test_live_loader_matches_archive_loader():
    arch = load_archive(ARCHIVE)
    live = load_live(FakeClient(ARCHIVE), verbose=False)
    played = arch.players.loc[arch.players["season_minutes"] > 0, "id"]
    a = arch.history[arch.history["element"].isin(played)].reset_index(drop=True)
    b = live.history.reset_index(drop=True)
    assert len(a) == len(b)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    assert live.last_finished_gw == arch.last_finished_gw and live.next_gw == arch.next_gw


@needs_archive
def test_public_team_state_and_predictions():
    client = FakeClient(ARCHIVE)
    ds = load_live(client, verbose=False)
    played = ds.history.groupby("element")["minutes"].sum().sort_values(ascending=False).index
    client.squad = list(played[:15])
    st = team_state_public(client, 1, ds)
    assert st.bank == 1.2 and len(st.squad) == 15
    # Without a login the plan must use the safe minimum, while still reporting the estimate.
    assert st.free_transfers == 1 and st.ft_assumed
    assert st.ft_estimate == min(5, ds.last_finished_gw - 1)  # one transfer in GW3, rolled the rest
    assert team_state_public(client, 1, ds, free_transfers=3).free_transfers == 3
    now = ds.players.set_index("id")["price"]
    p0 = client.squad[0]                                      # bought for £5.0m in GW3
    assert st.selling_price[p0] == selling_price(50, int(round(now[p0] * 10))) / 10
    pred = predict(ds, [ds.next_gw])
    assert pred["xpts"].notna().all() and (pred["xpts"] > -1).all()
