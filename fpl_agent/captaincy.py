"""Captain study: how should the captain be chosen?

Walk-forward over past seasons (the model only sees data from before each
gameweek), each rule picks a captain from players expected to play 60+
minutes; if he doesn't play, the rule's second choice scores (vice captain).
Fitted rules are scored leave-one-season-out, so every number is out of sample.

Run:  python -m fpl_agent captain-study --archive data/2023-24 --archive data/2024-25 --archive data/2025-26
Its last lines are the CAPTAIN_CALIBRATION to paste into config.py after any
change to the predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FWD, MID, ModelParams
from .data import load_archive
from .predictor import predict

ATTACKERS = (MID, FWD)


def build_dataset(season_dirs: list[str], start_gw: int = 5, params: ModelParams | None = None,
                  verbose: bool = True) -> pd.DataFrame:
    rows = []
    for d in season_dirs:
        ds = load_archive(d)
        act = ds.history.groupby(["element", "round"]).agg(actual=("total_points", "sum"), mins=("minutes", "sum"),
                                                           selected=("selected", "max"))
        rounds = set(act.index.get_level_values("round"))
        for g in range(start_gw, ds.last_finished_gw + 1):
            if g not in rounds:
                continue
            p = predict(ds.as_of(g - 1), [g], params)
            p = p[p["xpts"] > 0].join(act.xs(g, level="round"), on="id", how="inner")
            p["season"] = ds.source
            rows.append(p[["season", "gw", "id", "web_name", "pos", "price", "xmins", "xpts", "selected",
                           "actual", "mins"]])
        if verbose:
            print(f"  {ds.source}: done")
    return pd.concat(rows, ignore_index=True)


def fit_calibration(df: pd.DataFrame, min_xpts: float = 3.0) -> dict:
    """actual ≈ a + b·xPts per position, on predictions >= min_xpts (where captain choices live)."""
    out = {}
    for k, g in df[df["xpts"] >= min_xpts].groupby("pos"):
        b, a = np.polyfit(g["xpts"], g["actual"], 1)
        out[int(k)] = (float(a), float(b))
    return out


def _pick(week: pd.DataFrame, score: pd.Series, attackers_only: bool) -> float:
    pool = week[week["xmins"] >= 60]
    if attackers_only:
        pool = pool[pool["pos"].isin(ATTACKERS)]
    order = pool.assign(_s=score.loc[pool.index]).sort_values(["_s", "xpts"], ascending=False)
    cap, vice = order.iloc[0], order.iloc[1]
    return float(cap["actual"] if cap["mins"] > 0 else vice["actual"])


def compare_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        raise ValueError("Need at least two seasons (calibration is scored leave-one-season-out).")
    df = df.copy()
    df["calibrated"] = np.nan
    for s in seasons:
        cal = fit_calibration(df[df["season"] != s])
        m = df["season"] == s
        df.loc[m, "calibrated"] = [cal[k][0] + cal[k][1] * x if k in cal else x
                                   for k, x in zip(df.loc[m, "pos"], df.loc[m, "xpts"])]
    rules = {
        "highest xPts (old rule)": ("xpts", False),
        "highest xPts, MID/FWD only": ("xpts", True),
        "calibrated, MID/FWD only (current)": ("calibrated", True),
        "most-owned MID/FWD (the crowd)": ("selected", True),
    }
    res = {}
    for name, (col, att) in rules.items():
        res[name] = pd.Series({(s, gw): _pick(w, w[col], att) for (s, gw), w in df.groupby(["season", "gw"])})
    base = res["highest xPts (old rule)"]
    rows = []
    for name, r in res.items():
        diff = r - base
        row = {"rule": name, **{s: r.xs(s).mean() for s in seasons}, "all": r.mean(),
               "vs_old": diff.mean(), "se": diff.std(ddof=1) / np.sqrt(len(diff))}
        rows.append(row)
    return pd.DataFrame(rows).set_index("rule"), fit_calibration(df)


def run_study(season_dirs: list[str]) -> None:
    print("Building walk-forward predictions...")
    df = build_dataset(season_dirs)
    table, cal = compare_rules(df)
    seasons = [c for c in table.columns if c not in ("all", "vs_old", "se")]
    print(f"\nAverage captain points per gameweek ({df.groupby(['season', 'gw']).ngroups} gameweeks):")
    print(f"{'rule':<36}" + "".join(f"{s.split(':')[-1]:>10}" for s in seasons) + f"{'all':>8}{'vs old rule':>15}")
    for name, r in table.iterrows():
        vs = "" if name.startswith("highest xPts (old") else f"{r.vs_old:+.2f} ± {r.se:.2f}"
        print(f"{name:<36}" + "".join(f"{r[s]:>10.2f}" for s in seasons) + f"{r['all']:>8.2f}{vs:>15}")
    print("\nCalibration fitted on all seasons (paste into config.py if it changed):")
    print("CAPTAIN_CALIBRATION = {MID: (%.3f, %.3f), FWD: (%.3f, %.3f)}" % (*cal[MID], *cal[FWD]))
