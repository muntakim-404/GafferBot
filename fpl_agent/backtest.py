"""Walk-forward backtest.

For each gameweek g the model only sees data up to g-1 (Dataset.as_of), predicts
g, and is scored against what actually happened. Baselines:
  * form4  - average points over the player's last 4 team fixtures;
  * season - average points per team fixture so far this season.

Why there is no "FPL's own xP" baseline: the archive's `xP` column was checked
and it leaks results (correlation ~0.75-0.83 with actual points, and up to half
the rows equal the actual score exactly), so it is recorded after matches and
would be an unfair, meaningless comparison.

Metrics per gameweek, averaged over the backtest:
  * mae / spearman over all players whose team played;
  * top11   - actual points of the 11 highest-predicted players (ignores formation),
              i.e. "if I picked by this model, what would I have scored";
  * captain - actual points of the single highest-predicted player;
  * oracle  - the best possible top11 with hindsight, for scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelParams
from .data import Dataset
from .predictor import predict

PREDICTORS = ("model", "form4", "season")


def _baselines(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.sort_values("kickoff_time")
    form4 = h.groupby("element").tail(4).groupby("element")["total_points"].mean()
    season = h.groupby("element")["total_points"].mean()
    return pd.DataFrame({"form4": form4, "season": season})


def _metrics(df: pd.DataFrame, col: str) -> dict:
    top = df.sort_values(col, ascending=False)
    ranked = df[col].rank()
    return {
        "mae": float(np.abs(df[col] - df["actual"]).mean()),
        "spearman": float(ranked.corr(df["actual"].rank())) if ranked.std() > 0 else np.nan,
        "top11": float(top["actual"].head(11).sum()),
        "captain": float(top["actual"].iloc[0]),
    }


def run_backtest(ds: Dataset, start_gw: int = 5, end_gw: int | None = None,
                 params: ModelParams | None = None, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary per predictor, per-player predictions for every gameweek)."""
    end_gw = end_gw or ds.last_finished_gw
    actual_all = ds.history.groupby(["element", "round"])["total_points"].sum()
    rounds = set(actual_all.index.get_level_values("round"))
    rows, preds = [], []
    for g in range(start_gw, end_gw + 1):
        if g not in rounds:
            continue
        past = ds.as_of(g - 1)
        df = pd.DataFrame({"actual": actual_all.xs(g, level="round")})
        df["model"] = predict(past, [g], params).set_index("id")["xpts"].reindex(df.index).fillna(0.0)
        df = df.join(_baselines(past.history)).fillna({"form4": 0.0, "season": 0.0})
        df["gw"] = g
        preds.append(df.rename_axis("element").reset_index())
        oracle = df["actual"].nlargest(11).sum()
        for col in PREDICTORS:
            rows.append({"gw": g, "predictor": col, **_metrics(df, col), "oracle": oracle})
        if verbose:
            got = {r["predictor"]: r["top11"] for r in rows if r["gw"] == g}
            print(f"  GW{g:>2}: " + "  ".join(f"{k} {v:.0f}" for k, v in got.items()) + f"  (oracle {oracle:.0f})")
    per_gw = pd.DataFrame(rows)
    summary = (per_gw.groupby("predictor")[["mae", "spearman", "top11", "captain", "oracle"]].mean()
               .assign(gws=per_gw.groupby("predictor").size()).sort_values("top11", ascending=False))
    return summary, pd.concat(preds, ignore_index=True)
