"""Predictor tests: absence handling and return dates.

The archive tests use real 2026/27 data (set FPL_ARCHIVE to the 2026-27 folder).
"""
import os

import numpy as np
import pandas as pd
import pytest

from fpl_agent.config import ModelParams
from fpl_agent.predictor import absence_mask, return_gameweek

ARCHIVE = os.environ.get("FPL_ARCHIVE")
needs_2026 = pytest.mark.skipif(not ARCHIVE or "2026-27" not in ARCHIVE, reason="set FPL_ARCHIVE to the 2026-27 folder")

UTC = "UTC"
DEADLINES = {10: pd.Timestamp("2026-11-06 18:30", tz=UTC), 11: pd.Timestamp("2026-11-20 18:30", tz=UTC),
             12: pd.Timestamp("2026-11-27 18:30", tz=UTC), 13: pd.Timestamp("2026-12-02 18:30", tz=UTC),
             20: pd.Timestamp("2027-01-08 18:30", tz=UTC)}
REF = DEADLINES[10]


@pytest.mark.parametrize("news, expected", [
    ("Hamstring injury - Expected back 24 Nov", 12),   # misses GW11 (matches 21-23 Nov)
    ("Hamstring injury - Expected back 21 Nov", 11),   # back for GW11's matches
    ("Knee injury - Expected back 05 Jan", 20),        # year rolls over
    ("Suspended until 28 Nov", 12),
    ("Knee injury - Unknown return date", None),
    ("Thigh injury - 75% chance of playing", None),
    ("", None),
])
def test_return_gameweek(news, expected):
    assert return_gameweek(news, DEADLINES, REF) == expected


def _hist(rows):
    """rows: {player_id: [minutes in date order]} -> history frame."""
    out = []
    for pid, mins in rows.items():
        for i, m in enumerate(mins):
            out.append({"element": pid, "minutes": m,
                        "kickoff_time": pd.Timestamp("2026-08-20", tz=UTC) + pd.Timedelta(days=7 * i)})
    return pd.DataFrame(out)


def test_absence_mask_rules():
    h = _hist({
        1: [90, 90, 90, 90, 0, 0, 90],   # regular, absence ended -> excluded
        2: [90, 90, 90, 90, 0, 0],       # regular, ongoing, flagged -> excluded
        3: [90, 90, 90, 90, 0, 0],       # regular, ongoing, NOT flagged (lost place?) -> kept
        4: [90, 0, 20, 0, 0, 90],        # rotation player -> kept
        5: [0, 0, 90, 90, 90],           # zeros before he ever played -> kept
    })
    ex = absence_mask(h, flagged={2}, p=ModelParams())
    by = {pid: ex[h["element"] == pid].tolist() for pid in h["element"].unique()}
    assert by[1] == [False] * 4 + [True, True, False]
    assert by[2] == [False] * 4 + [True, True]
    assert not any(by[3]) and not any(by[4]) and not any(by[5])
    assert not absence_mask(h, {2}, ModelParams(exclude_absences=False)).any()


@needs_2026
def test_injured_regular_is_not_punished_twice():
    """Amenda: 90,90,90,90 then injured (0,0), 'Unknown return date'.
    Not available next GW, but once the injury ramp reaches 100% he should be
    expected to play ~full minutes again, not ~39 (the old double-counting)."""
    from fpl_agent.data import load_archive
    from fpl_agent.predictor import predict
    ds = load_archive(ARCHIVE)
    pid = int(ds.players.loc[ds.players["web_name"] == "Amenda", "id"].iloc[0])
    gws = list(range(ds.next_gw, ds.next_gw + 6))
    pred = predict(ds, gws).set_index(["id", "gw"])
    assert pred.loc[(pid, ds.next_gw), "xpts"] == 0
    played = [g for g in gws if pred.loc[(pid, g), "xmins"] > 0]
    assert max(pred.loc[(pid, g), "xmins"] for g in played) > 80


@needs_2026
def test_return_date_drives_availability():
    """Elanga: 'Expected back 25 Oct' -> out before the return gameweek, playing after."""
    from fpl_agent.data import load_archive
    from fpl_agent.predictor import predict
    ds = load_archive(ARCHIVE)
    pid = int(ds.players.loc[ds.players["web_name"] == "Elanga", "id"].iloc[0])
    ref = pd.Timestamp(ds.deadlines[ds.next_gw])
    rgw = return_gameweek("Expected back 25 Oct", ds.deadlines, ref)
    assert rgw is not None and rgw > ds.next_gw
    pred = predict(ds, list(range(ds.next_gw, rgw + 2))).set_index(["id", "gw"])
    for g in range(ds.next_gw, rgw):
        assert pred.loc[(pid, g), "xpts"] == 0
    assert 0 < pred.loc[(pid, rgw), "xmins"] < pred.loc[(pid, rgw + 1), "xmins"]
    assert pred.loc[(pid, rgw + 1), "xmins"] > 70
