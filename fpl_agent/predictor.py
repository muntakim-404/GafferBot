"""Baseline expected-points (xPts) model.

Built from interpretable components rather than one black box, so each part can
be tested and replaced on its own:

  xPts = appearance + goals + assists + clean sheet + goals conceded
         + saves + defensive contributions + bonus + cards

Every component is (probability or expected minutes) x (per-minute rate) x
(opponent/venue adjustment). Rates are recency-weighted and shrunk toward a
prior so that players with few minutes don't get extreme values.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config import (APPEAR_60_BONUS, APPEAR_PTS, ASSIST_PTS, CLEAN_SHEET_PTS, CONCEDED_PER_2,
                     DC_PTS, DC_THRESHOLD, GK, GOAL_PTS, RED_PTS, SAVES_PER_3, YELLOW_PTS,
                     ModelParams)
from .data import Dataset

_N = np.arange(0, 41)


def _exp_floor_div(lam: np.ndarray, k: int) -> np.ndarray:
    """E[floor(X / k)] for X ~ Poisson(lam): e.g. -1 per 2 goals conceded, +1 per 3 saves."""
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-9)
    pmf = poisson.pmf(_N[None, :], lam[:, None])
    return (pmf * (_N // k)[None, :]).sum(axis=1)


def _recency_weights(df: pd.DataFrame, key: str, halflife: float, window: int) -> pd.Series:
    rank = df.groupby(key)["kickoff_time"].rank(method="first", ascending=False) - 1
    return np.power(0.5, rank / halflife).where(rank < window, 0.0)


# ------------------------------------------------------------------ team ratings
def team_ratings(hist: pd.DataFrame, p: ModelParams) -> dict:
    """Attack and defence multipliers per team (1.0 = league average), plus venue factors."""
    default = {"mu": 1.35, "home": p.default_home_factor, "away": p.default_away_factor, "att": {}, "dfn": {}}
    if hist.empty:
        return default
    tf = hist.groupby(["fixture", "team"], as_index=False).agg(
        xg=("expected_goals", "sum"), was_home=("was_home", "first"), kickoff_time=("kickoff_time", "first"))
    opp = tf[["fixture", "team", "xg"]].rename(columns={"team": "opp", "xg": "xga"})
    tf = tf.merge(opp, on="fixture")
    tf = tf[tf["team"] != tf["opp"]].reset_index(drop=True)
    if tf.empty:
        return default
    mu = tf["xg"].mean()
    n, k = len(tf), p.venue_prior_fixtures
    home = (n * tf.loc[tf["was_home"], "xg"].mean() / mu + k * p.default_home_factor) / (n + k)
    away = (n * tf.loc[~tf["was_home"], "xg"].mean() / mu + k * p.default_away_factor) / (n + k)
    # Remove the venue effect before averaging, so a run of home games doesn't inflate a rating.
    tf["xg_adj"] = tf["xg"] / np.where(tf["was_home"], home, away)
    tf["xga_adj"] = tf["xga"] / np.where(tf["was_home"], away, home)
    w = _recency_weights(tf, "team", p.team_halflife, p.team_window)
    agg = tf.assign(w=w, a=tf["xg_adj"] * w, d=tf["xga_adj"] * w).groupby("team")[["w", "a", "d"]].sum()
    kt = p.team_prior_matches
    att = (agg["a"] + kt * mu) / (agg["w"] + kt) / mu
    dfn = (agg["d"] + kt * mu) / (agg["w"] + kt) / mu
    return {"mu": mu, "home": home, "away": away, "att": att.to_dict(), "dfn": dfn.to_dict()}


# --------------------------------------------------------------------- absences
def absence_mask(hist: pd.DataFrame, flagged: set, p: ModelParams) -> pd.Series:
    """True for 0-minute rows that should not count against a player's minutes.

    A streak of consecutive 0-minute fixtures is excluded when the player was a
    regular before it (>= regular_share of the previous regular_lookback fixtures
    at 60+ minutes) and either the streak has ended (he has played since) or he
    is currently flagged by FPL (injury/suspension). An ongoing streak without
    a flag is kept: that looks like a lost place, which the minutes model should see.
    """
    excl = np.zeros(len(hist), dtype=bool)
    if hist.empty or not p.exclude_absences:
        return pd.Series(excl, index=hist.index)
    h = pd.DataFrame({"element": hist["element"].to_numpy(), "minutes": hist["minutes"].to_numpy(),
                      "t": hist["kickoff_time"].to_numpy(), "pos": np.arange(len(hist))})
    h = h.sort_values(["element", "t"], kind="stable")
    for pid, g in h.groupby("element", sort=False):
        m, idx = g["minutes"].to_numpy(), g["pos"].to_numpy()
        n, i = len(m), 0
        while i < n:
            if m[i] > 0:
                i += 1
                continue
            j = i
            while j < n and m[j] == 0:
                j += 1
            prior = m[max(0, i - p.regular_lookback):i]
            regular = len(prior) >= 2 and (prior >= 60).mean() >= p.regular_share
            if regular and (j < n or pid in flagged):
                excl[idx[i:j]] = True
            i = j
    return pd.Series(excl, index=hist.index)


# --------------------------------------------------------------- player features
def _price_prior(players: pd.DataFrame, season: pd.DataFrame, col: str, use_price: bool,
                 min_minutes: float = 270.0) -> pd.Series:
    """Per-minute prior rate by position, optionally as a linear function of price.

    Price is a decent proxy for attacking role, so a new or rarely-used £10m
    player is pulled toward other £10m players rather than toward the average.
    """
    df = players[["pos", "price"]].join(season[["minutes", col]], how="left").fillna({"minutes": 0.0, col: 0.0})
    prior = pd.Series(0.0, index=df.index)
    for _, g in df.groupby("pos"):
        tot = g["minutes"].sum()
        mean_rate = g[col].sum() / tot if tot > 0 else 0.0
        fit = g[g["minutes"] >= min_minutes]
        if use_price and len(fit) >= 8 and fit["price"].nunique() > 2:
            y = fit[col] / fit["minutes"]
            b, a = np.polyfit(fit["price"], y, 1, w=np.sqrt(fit["minutes"]))
            pred = (a + b * g["price"]).clip(lower=0.3 * mean_rate, upper=max(y.max(), mean_rate))
        else:
            pred = pd.Series(mean_rate, index=g.index)
        prior.loc[g.index] = pred
    return prior


def player_features(ds: Dataset, p: ModelParams) -> tuple[pd.DataFrame, dict]:
    players = ds.players.set_index("id")
    idx = players.index
    hist = ds.history
    f = pd.DataFrame(index=idx)

    # Minutes: recency-weighted over the player's recent team fixtures (0-minute rows
    # included, except absences of regular starters; see absence_mask).
    flagged = set(players.index[players["status"].isin(["i", "d", "s", "n"])]) if ds.has_availability else set()
    hmin = hist[~absence_mask(hist, flagged, p)]
    if len(hmin):
        w = _recency_weights(hmin, "element", p.minutes_halflife, p.minutes_window)
        g = pd.DataFrame({"element": hmin["element"], "w": w, "m": hmin["minutes"] * w,
                          "a": (hmin["minutes"] > 0) * w, "s": (hmin["minutes"] >= 60) * w}).groupby("element").sum()
        f["xmins"] = (g["m"] / g["w"]).reindex(idx)
        f["p_any"] = (g["a"] / g["w"]).reindex(idx)
        f["p60"] = (g["s"] / g["w"]).reindex(idx)
    else:
        f["xmins"] = f["p_any"] = f["p60"] = np.nan
    no_hist = f["xmins"].isna()
    zero_season = players["season_minutes"].fillna(-1).eq(0) & (ds.last_finished_gw >= 2)
    f.loc[no_hist & zero_season, ["xmins", "p_any", "p60"]] = 0.0
    fill = no_hist & ~zero_season
    f.loc[fill, "p_any"] = p.no_history_p_any
    f.loc[fill, "p60"] = p.no_history_p60
    f.loc[fill, "xmins"] = p.no_history_p60 * 85 + (p.no_history_p_any - p.no_history_p60) * 25

    # Per-minute rates over recent appearances, shrunk toward a prior.
    app = hist[hist["minutes"] > 0]
    cols = ["minutes", "expected_goals", "expected_assists", "saves", "bonus", "yellow_cards", "red_cards"]
    season = app.groupby("element")[cols].sum() if len(app) else pd.DataFrame(columns=cols)
    if len(app):
        w = _recency_weights(app, "element", p.rate_halflife, p.rate_window)
        ws = app[cols].mul(w, axis=0).assign(element=app["element"]).groupby("element").sum().reindex(idx).fillna(0.0)
    else:
        ws = pd.DataFrame(0.0, index=idx, columns=cols)
    specs = {  # feature: (column, use price in prior, prior strength in minutes)
        "xg_rate": ("expected_goals", True, p.rate_prior_minutes),
        "xa_rate": ("expected_assists", True, p.rate_prior_minutes),
        "bonus_rate": ("bonus", True, p.rate_prior_minutes),
        "save_rate": ("saves", False, p.rate_prior_minutes),
        "yc_rate": ("yellow_cards", False, p.card_prior_minutes),
        "rc_rate": ("red_cards", False, p.card_prior_minutes),
    }
    for feat, (col, use_price, k) in specs.items():
        prior = _price_prior(players, season, col, use_price)
        f[feat] = (ws[col] + k * prior) / (ws["minutes"] + k)
    f.loc[players["pos"] != GK, "save_rate"] = 0.0

    # Defensive contributions: how often the player reaches the threshold when playing 60+.
    thr = hist["pos"].map(DC_THRESHOLD)
    full = hist[(hist["minutes"] >= 60) & thr.notna()].copy()
    if len(full):
        full["hit"] = (full["defensive_contribution"] >= full["pos"].map(DC_THRESHOLD)).astype(float)
        w = _recency_weights(full, "element", p.rate_halflife, p.rate_window)
        dc = full.assign(w=w, h=full["hit"] * w).groupby("element")[["w", "h"]].sum().reindex(idx).fillna(0.0)
        prior = players["pos"].map(full.groupby("pos")["hit"].mean()).fillna(0.0)
        f["dc_rate"] = (dc["h"] + p.dc_prior_matches * prior) / (dc["w"] + p.dc_prior_matches)
    else:
        f["dc_rate"] = 0.0
    f.loc[players["pos"] == GK, "dc_rate"] = 0.0

    # FPL awards assists more generously than xA counts them; calibrate both to actual outcomes.
    calib = {
        "goal_mult": (app["goals_scored"].sum() + 30) / (app["expected_goals"].sum() + 30) if len(app) else 1.0,
        "assist_mult": (app["assists"].sum() + 30) / (app["expected_assists"].sum() + 30) if len(app) else 1.0,
    }
    return f, calib


# ------------------------------------------------------------------ availability
_RETURN_RE = re.compile(r"(?:expected back|suspended until)\s+(\d{1,2})\s+([a-z]{3})", re.IGNORECASE)
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun",
                                       "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def return_gameweek(news: str, deadlines: dict, ref: pd.Timestamp) -> int | None:
    """Gameweek a player is due back, from FPL news like
    'Hamstring injury - Expected back 24 Nov' or 'Suspended until 08 Nov'.

    The year is inferred as the one that puts the date nearest to `ref`.
    The return gameweek is the first one whose deadline is no more than two
    days before the date (most of a gameweek's matches follow its deadline).
    """
    m = _RETURN_RE.search(news or "")
    if not m or m.group(2).lower() not in _MONTHS:
        return None
    day, month = int(m.group(1)), _MONTHS[m.group(2).lower()]
    best = None
    for year in (ref.year - 1, ref.year, ref.year + 1):
        try:
            d = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
        except ValueError:
            continue
        if best is None or abs(d - ref) < abs(best - ref):
            best = d
    if best is None:
        return None
    later = [gw for gw, t in deadlines.items() if pd.notna(t) and t >= best - pd.Timedelta(days=2)]
    return min(later) if later else None


def availability(pf: pd.DataFrame, next_gw: int, deadlines: dict, p: ModelParams, known: bool) -> np.ndarray:
    """Probability each player-fixture row (needs status, chance_next, news, gw) is available."""
    if not known:
        return np.ones(len(pf))
    k = (pf["gw"] - next_gw).clip(lower=0).to_numpy()
    st = pf["status"].to_numpy()
    ch = pf["chance_next"].to_numpy(dtype=float) / 100.0
    base = np.where(~np.isnan(ch), ch, np.where(st == "a", 1.0, np.where(st == "d", 0.5, 0.0)))
    rec = np.where(st == "s", 1.0, p.injury_recovery_per_gw)  # suspensions usually last one match
    a = np.minimum(1.0, base + rec * k)

    # Flagged players whose news gives a return date: follow the date instead of the ramp.
    ref = deadlines.get(next_gw)
    if ref is not None and pd.notna(ref):
        ref = pd.Timestamp(ref)
        ref = ref.tz_localize("UTC") if ref.tzinfo is None else ref
        flagged = pf.loc[pf["status"] != "a", ["id", "news"]].drop_duplicates("id")
        ret = {r.id: return_gameweek(r.news, deadlines, ref) for r in flagged.itertuples()}
        rgw = pf["id"].map(ret).to_numpy(dtype=float)       # NaN when no date
        gw = pf["gw"].to_numpy()
        has = ~np.isnan(rgw) & (rgw > next_gw)
        dated = np.where(gw < rgw, 0.0, np.where(gw == rgw, p.return_week_avail, 1.0))
        a = np.where(has, dated, a)
    return np.where(st == "u", 0.0, a)  # 'u' = left the club / out for the season


# ----------------------------------------------------------------------- predict
def predict(ds: Dataset, gws: list[int], params: ModelParams | None = None,
            starters: list[int] = (), out: list[int] = ()) -> pd.DataFrame:
    """Expected points per player per gameweek (doubles summed, blanks = 0).

    starters: ids you know will start (e.g. back from injury) -> treated as nailed.
    out:      ids you know will miss the whole horizon -> 0.
    """
    p = params or ModelParams()
    f, calib = player_features(ds, p)
    rt = team_ratings(ds.history, p)
    if len(starters):
        s = f.index.isin(starters)
        f.loc[s, ["p_any", "p60", "xmins"]] = [0.95, 0.85, 82.0]

    fx = ds.fixtures[ds.fixtures["event"].isin(gws) & ~ds.fixtures["finished"]]
    tf = pd.concat([
        pd.DataFrame({"team": fx["team_h"], "opp": fx["team_a"], "home": True, "gw": fx["event"].astype(int)}),
        pd.DataFrame({"team": fx["team_a"], "opp": fx["team_h"], "home": False, "gw": fx["event"].astype(int)}),
    ], ignore_index=True)
    pf = ds.players[["id", "team", "pos", "status", "chance_next", "news", "can_buy"]].merge(tf, on="team")
    pf = pf.join(f, on="id")

    next_gw = ds.next_gw or min(gws)
    k = (pf["gw"] - next_gw).clip(lower=0).to_numpy()
    avail = availability(pf, next_gw, ds.deadlines, p, ds.has_availability)
    if len(starters):
        avail = np.where(pf["id"].isin(starters) & (k == 0), 1.0, avail)
    if len(out):
        avail = np.where(pf["id"].isin(out), 0.0, avail)

    xmins = pf["xmins"].to_numpy() * avail
    p_any = pf["p_any"].to_numpy() * avail
    p60 = pf["p60"].to_numpy() * avail
    xmins_if_plays = np.where(pf["p_any"] > 0, pf["xmins"] / pf["p_any"].clip(lower=1e-6), 0.0)

    home = pf["home"].to_numpy()
    own_venue = np.where(home, rt["home"], rt["away"])
    opp_venue = np.where(home, rt["away"], rt["home"])
    opp_att = pf["opp"].map(rt["att"]).fillna(1.0).to_numpy()
    opp_dfn = pf["opp"].map(rt["dfn"]).fillna(1.0).to_numpy()
    own_dfn = pf["team"].map(rt["dfn"]).fillna(1.0).to_numpy()
    pos = pf["pos"].to_numpy()

    e_goals = pf["xg_rate"].to_numpy() * xmins * opp_dfn * own_venue * calib["goal_mult"]
    e_assists = pf["xa_rate"].to_numpy() * xmins * opp_dfn * own_venue * calib["assist_mult"]
    lam_conc = rt["mu"] * own_dfn * opp_att * opp_venue   # goals the player's team concedes over 90'
    lam_saves = pf["save_rate"].to_numpy() * xmins_if_plays * opp_att * opp_venue

    comp = pd.DataFrame({
        "id": pf["id"], "gw": pf["gw"],
        "xmins": xmins,
        "pts_appear": APPEAR_PTS * p_any + APPEAR_60_BONUS * p60,
        "pts_goals": np.vectorize(GOAL_PTS.get)(pos) * e_goals,
        "pts_assists": ASSIST_PTS * e_assists,
        "pts_cs": np.vectorize(CLEAN_SHEET_PTS.get)(pos) * p60 * np.exp(-lam_conc),
        "pts_conceded": np.vectorize(CONCEDED_PER_2.get)(pos) * p60 * _exp_floor_div(lam_conc, 2),
        "pts_saves": SAVES_PER_3 * p_any * _exp_floor_div(lam_saves, 3),
        "pts_dc": DC_PTS * pf["dc_rate"].to_numpy() * p60,
        "pts_bonus": pf["bonus_rate"].to_numpy() * xmins,
        "pts_cards": (YELLOW_PTS * pf["yc_rate"].to_numpy() + RED_PTS * pf["rc_rate"].to_numpy()) * xmins,
        "n_fix": 1,
    })
    pts_cols = [c for c in comp.columns if c.startswith("pts_")]
    comp["xpts"] = comp[pts_cols].sum(axis=1)
    out_df = comp.groupby(["id", "gw"], as_index=False).sum()

    grid = pd.MultiIndex.from_product([ds.players["id"], sorted(gws)], names=["id", "gw"])
    out_df = out_df.set_index(["id", "gw"]).reindex(grid, fill_value=0.0).reset_index()
    out_df = out_df.merge(ds.players[["id", "web_name", "team", "pos", "price"]], on="id")
    out_df["avail_next"] = out_df["id"].map(pd.Series(avail[pf["gw"].to_numpy() == next_gw],
                                                      index=pf.loc[pf["gw"] == next_gw, "id"]).groupby(level=0).first())
    return out_df


def xpts_matrix(pred: pd.DataFrame) -> pd.DataFrame:
    """Wide form: index = player id, columns = gameweeks."""
    return pred.pivot_table(index="id", columns="gw", values="xpts", aggfunc="sum").fillna(0.0)
