"""Captain study tests on synthetic data where the right answer is known."""
import numpy as np
import pandas as pd
import pytest

from fpl_agent.captaincy import compare_rules, fit_calibration
from fpl_agent.config import DEF, FWD, MID, captain_score


def test_fit_calibration_recovers_a_known_bias():
    rng = np.random.default_rng(0)
    x = rng.uniform(3, 10, 4000)
    df = pd.DataFrame({"pos": FWD, "xpts": x, "actual": 0.5 + 0.7 * x + rng.normal(0, 0.5, len(x))})
    a, b = fit_calibration(df)[FWD]
    assert a == pytest.approx(0.5, abs=0.1) and b == pytest.approx(0.7, abs=0.02)


def test_captain_score_only_for_midfielders_and_forwards():
    assert captain_score(DEF, 8.0) is None
    assert captain_score(MID, 8.0) > captain_score(FWD, 8.0)     # forwards' forecasts run hotter


def test_compare_rules_rewards_the_calibrated_rule_when_forwards_are_overrated():
    """Each week: a forward forecast at 8 who really scores ~5, a midfielder forecast at 7.5 who scores ~7.5,
    and a defender forecast at 9 who scores ~3. The calibrated MID/FWD rule should learn to pick the midfielder."""
    rng = np.random.default_rng(1)
    rows = []
    for season in ("s1", "s2", "s3"):
        for gw in range(1, 31):
            for pid, pos, xp, true in ((1, FWD, 8.0, 5.0), (2, MID, 7.5, 7.5), (3, DEF, 9.0, 3.0),
                                       (4, MID, 4.0, 4.0), (5, FWD, 3.5, 3.5)):
                for jitter in range(3):    # a few rows per player so the fit has spread
                    rows.append({"season": season, "gw": gw, "id": pid * 10 + jitter, "pos": pos,
                                 "xpts": xp + rng.normal(0, 0.3), "actual": true + rng.normal(0, 1.0),
                                 "xmins": 85, "mins": 90, "selected": 1000 * (6 - pid)})
    table, _ = compare_rules(pd.DataFrame(rows))
    assert table.loc["calibrated, MID/FWD only (current)", "all"] > table.loc["highest xPts (old rule)", "all"] + 2
