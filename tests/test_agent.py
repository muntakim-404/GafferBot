"""Step 3 tests: timing windows, commands, failure alerts, shadow scoring, Telegram safety.

The tick tests use real 2026/27 data through FakeClient (see test_data.py);
set FPL_ARCHIVE to the 2026-27 folder to run them.
"""
import json
import os

import pandas as pd
import pytest
import requests

from fpl_agent.agent import AgentConfig, load_state, score_lineup, tick
from fpl_agent.config import DEF, FWD, GK, MID
from fpl_agent.telegram import Telegram, TelegramError, split_message

ARCHIVE = os.environ.get("FPL_ARCHIVE")
needs_2026 = pytest.mark.skipif(not ARCHIVE or "2026-27" not in ARCHIVE, reason="set FPL_ARCHIVE to the 2026-27 folder")


# ----------------------------------------------------------------- unit tests
def test_score_lineup_autosubs_and_vice_captain():
    pos = {1: GK, 2: DEF, 3: DEF, 4: DEF, 5: MID, 6: MID, 7: MID, 8: MID, 9: FWD, 10: FWD, 11: FWD,
           12: GK, 13: DEF, 14: MID, 15: FWD}
    lineup, bench = list(range(1, 12)), [12, 14, 13, 15]
    mins = {i: 90 for i in pos}
    mins.update({1: 0, 2: 0, 9: 0})              # GK, a defender and the captain don't play
    pts = {i: 2 for i in pos}
    pts.update({13: 6, 14: 1, 5: 10})
    total, subs, doubled = score_lineup(lineup, bench, captain=9, vice=5, pos=pos, pts=pts, mins=mins)
    # GK -> 12; DEF 2 can't go to MID 14 (only 2 DEF left), so -> DEF 13; FWD 9 -> MID 14 (2 FWD remain)
    assert subs == [(1, 12), (2, 13), (9, 14)]
    assert doubled == 5                          # vice gets the double
    assert total == (2 + 6 + 2 + 2 + 10 + 2 + 2 + 2 + 1 + 2 + 2) + 10


def test_telegram_errors_never_contain_the_token(monkeypatch):
    token = "123456:SECRET-TOKEN"

    def boom(url, **kw):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(TelegramError) as e:
        Telegram(token, "42").send("hi")
    assert token not in str(e.value) and "<token>" in str(e.value)


def test_unreachable_fpl_alerts_once_a_day(tmp_path):
    class Down:
        def bootstrap(self, max_age=None):
            raise RuntimeError("FPL API kept failing for bootstrap-static")
    tg = FakeTelegram()
    cfg = AgentConfig(team_id=1, root=tmp_path)
    t = pd.Timestamp("2026-11-06 10:17", tz="UTC")
    assert tick(cfg, Down(), tg, t) == 1 and len(tg.sent) == 1 and "Can't reach FPL" in tg.sent[0]
    assert tick(cfg, Down(), tg, t + pd.Timedelta(hours=1)) == 1 and len(tg.sent) == 1
    assert tick(cfg, Down(), tg, t + pd.Timedelta(days=1)) == 1 and len(tg.sent) == 2


def test_split_message():
    text = "\n".join("x" * 100 for _ in range(100))
    parts = split_message(text, limit=1000)
    assert all(len(p) <= 1000 for p in parts) and "".join(parts).count("x") == 100 * 100


# ------------------------------------------------------------ fakes for ticks
class FakeTelegram:
    def __init__(self, updates=None, chat_id="42"):
        self.chat_id, self.sent, self._updates = chat_id, [], updates or []

    def send(self, text, chat_id=None):
        self.sent.append(text)

    def updates(self, offset):
        return [u for u in self._updates if u["update_id"] >= offset]


def msg(uid, text, chat=42):
    return {"update_id": uid, "message": {"chat": {"id": chat, "type": "private"}, "text": text}}


def make_client(fail=False):
    from test_data import FakeClient

    from fpl_agent.data import load_archive
    from fpl_agent.solver import most_owned_squad

    class AgentFake(FakeClient):
        def element_summary(self, pid, max_age=None):
            if fail:
                raise RuntimeError("FPL is down")
            return super().element_summary(pid, max_age)

        def live(self, gw, max_age=None):
            g = self.raw[self.raw["GW"] == gw].groupby("element")[["total_points", "minutes"]].sum()
            return {"elements": [{"id": int(i), "stats": {"total_points": int(r.total_points),
                                                          "minutes": int(r.minutes)}} for i, r in g.iterrows()]}

        def picks(self, eid, gw, max_age=None):
            js = super().picks(eid, gw)
            js["entry_history"].update({"points": 50, "event_transfers_cost": 4})
            js["picks"] = [{"element": p, "is_captain": k == 0} for k, p in enumerate(self.squad)]
            return js

    c = AgentFake(ARCHIVE)
    c.squad = most_owned_squad(load_archive(ARCHIVE).players)
    return c


def deadline(client, gw):
    ev = {e["id"]: e for e in client.bootstrap()["events"]}
    return pd.Timestamp(ev[gw]["deadline_time"]).tz_convert("UTC")


# --------------------------------------------------------------- tick tests
@needs_2026
def test_early_and_final_once_each(tmp_path):
    client, tg = make_client(), FakeTelegram()
    cfg = AgentConfig(team_id=1, tz="Asia/Dhaka", horizon=2, root=tmp_path, chips=False)
    d = deadline(client, 7)
    assert tick(cfg, client, tg, d - pd.Timedelta(hours=30)) == 0 and tg.sent == []
    tick(cfg, client, tg, d - pd.Timedelta(hours=20))
    assert len(tg.sent) == 1 and "GW7" in tg.sent[0] and "early look" in tg.sent[0]
    assert "Asia/Dhaka" in tg.sent[0] and "assumed" in tg.sent[0]
    tick(cfg, client, tg, d - pd.Timedelta(hours=6))            # no duplicate
    assert len(tg.sent) == 1
    tick(cfg, client, tg, d - pd.Timedelta(hours=3, minutes=50))
    assert len(tg.sent) == 2 and "final check" in tg.sent[1]
    tick(cfg, client, tg, d - pd.Timedelta(minutes=20))
    assert len(tg.sent) == 2
    assert (tmp_path / "logs" / "plans" / "gw07-early.json").exists()
    plan = json.loads((tmp_path / "logs" / "plans" / "gw07-final.json").read_text(encoding="utf-8"))
    assert len(plan["lineup"]) == 11 and plan["captain"] in plan["lineup"]


@needs_2026
def test_commands_and_ignoring_strangers(tmp_path):
    client = make_client()
    tg = FakeTelegram([msg(1, "/ft 3"), msg(2, "/ft 5", chat=999), msg(3, "/plan")])
    cfg = AgentConfig(team_id=1, horizon=2, root=tmp_path, chips=False)
    tick(cfg, client, tg, deadline(client, 7) - pd.Timedelta(hours=30))
    st = load_state(cfg)
    assert st["ft"] == {"gw": 7, "value": 3} and st["telegram_offset"] == 3
    assert "3 free transfer" in tg.sent[0] and "building a plan" in tg.sent[1]
    assert "on request" in tg.sent[2] and "Free transfers: 3 ·" in tg.sent[2]
    n = len(tg.sent)
    tick(cfg, client, tg, deadline(client, 7) - pd.Timedelta(hours=29))   # same updates are not re-read
    assert len(tg.sent) == n


@needs_2026
def test_failure_is_reported_once_and_retried(tmp_path):
    client, tg = make_client(fail=True), FakeTelegram()
    cfg = AgentConfig(team_id=1, horizon=2, root=tmp_path, chips=False)
    d = deadline(client, 7)
    assert tick(cfg, client, tg, d - pd.Timedelta(hours=10)) == 1
    assert len(tg.sent) == 1 and "Couldn't build" in tg.sent[0]
    assert tick(cfg, client, tg, d - pd.Timedelta(hours=8)) == 1
    assert len(tg.sent) == 1                                   # no spam within the early window
    assert load_state(cfg)["sent"].get("7", []) == []          # still due, so it keeps retrying
    assert tick(cfg, client, tg, d - pd.Timedelta(hours=3)) == 1
    assert len(tg.sent) == 2 and "final" in tg.sent[1]         # the final attempt gets one alert of its own


@needs_2026
def test_shadow_scoring_of_a_finished_gameweek(tmp_path):
    client, tg = make_client(), FakeTelegram()
    cfg = AgentConfig(team_id=1, horizon=2, root=tmp_path, chips=False)
    raw = client.raw[client.raw["GW"] == 6]
    agg = raw.groupby(["element", "position"], as_index=False)[["total_points", "minutes"]].sum()
    played = agg[agg["minutes"] > 0].sort_values("total_points", ascending=False)
    pick = lambda pos, n: played[played["position"] == pos]["element"].head(n).astype(int).tolist()
    lineup = pick("GK", 1) + pick("DEF", 4) + pick("MID", 4) + pick("FWD", 2)
    bench = (played[played["position"] == "GK"]["element"].iloc[1:2].astype(int).tolist()
             + played[played["position"] == "DEF"]["element"].iloc[4:5].astype(int).tolist()
             + played[played["position"] == "MID"]["element"].iloc[4:5].astype(int).tolist()
             + played[played["position"] == "FWD"]["element"].iloc[2:3].astype(int).tolist())
    posmap = {"GK": GK, "DEF": DEF, "MID": MID, "FWD": FWD}
    pos = {str(int(r.element)): posmap[r.position] for r in agg.itertuples() if int(r.element) in lineup + bench}
    plan = {"gw": 6, "window": "final", "lineup": lineup, "bench": bench, "captain": lineup[5],
            "vice": lineup[6], "hits": 1, "transfers_in": [], "transfers_out": [], "pos": pos,
            "names": {k: f"P{k}" for k in pos}}
    (tmp_path / "logs" / "plans").mkdir(parents=True)
    (tmp_path / "logs" / "plans" / "gw06-final.json").write_text(json.dumps(plan), encoding="utf-8")

    tick(cfg, client, tg, deadline(client, 7) - pd.Timedelta(hours=30))
    pts = agg.set_index("element")["total_points"]
    expected = int(pts.loc[lineup].sum() + pts.loc[lineup[5]]) - 4
    df = pd.read_csv(tmp_path / "logs" / "shadow.csv")
    assert df.iloc[0]["gw"] == 6 and df.iloc[0]["bot_points"] == expected and df.iloc[0]["your_points"] == 46
    assert len(tg.sent) == 1 and "GW6 result" in tg.sent[0]
    tick(cfg, client, tg, deadline(client, 7) - pd.Timedelta(hours=29))   # scored once only
    assert len(pd.read_csv(tmp_path / "logs" / "shadow.csv")) == 1 and len(tg.sent) == 1
