"""Data layer.

Two sources produce the same canonical `Dataset`:
  * the live FPL API (`FPLClient` + `load_live`), used for real decisions;
  * a season archive in the vaastav/Fantasy-Premier-League layout (`load_archive`),
    used for backtests and offline testing. Its files mirror the API's field names,
    so both paths go through the same canonicalisation code.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import MAX_FREE_TRANSFERS

API_BASE = "https://fantasy.premierleague.com/api"

HIST_NUM_COLS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "saves", "bonus", "bps", "yellow_cards", "red_cards", "own_goals", "penalties_missed",
    "penalties_saved", "expected_goals", "expected_assists", "expected_goals_conceded",
    "defensive_contribution", "total_points", "value", "selected",
]
HIST_COLS = ["element", "pos", "round", "fixture", "kickoff_time", "team", "opponent_team",
             "was_home"] + HIST_NUM_COLS


# ----------------------------------------------------------------------------- dataset
@dataclass
class Dataset:
    players: pd.DataFrame    # id, web_name, team, pos, price, status, chance_next, news, can_buy, ...
    teams: pd.DataFrame      # id, name, short_name
    fixtures: pd.DataFrame   # id, event, team_h, team_a, finished, kickoff_time
    history: pd.DataFrame    # one row per player per finished fixture (HIST_COLS)
    last_finished_gw: int
    next_gw: int | None
    deadlines: dict = field(default_factory=dict)   # gw -> UTC Timestamp
    has_availability: bool = True                   # False when injury flags are unknown
    source: str = ""

    def as_of(self, gw: int) -> "Dataset":
        """The data as it looked right after gameweek `gw` finished.

        Used by the backtest so that no future information leaks into a prediction.
        Injury flags are not archived historically, so they are reset to 'available'.
        """
        hist = self.history[self.history["round"] <= gw].copy()
        last = hist.sort_values("kickoff_time").groupby("element").last()
        p = self.players.copy()
        p["price"] = p["id"].map(last["value"] / 10.0).fillna(p["price"])
        p["team"] = p["id"].map(last["team"]).fillna(p["team"]).astype(int)
        p["season_minutes"] = p["id"].map(hist.groupby("element")["minutes"].sum()).fillna(0.0)
        p["status"], p["chance_next"], p["news"] = "a", np.nan, ""
        p["can_buy"] = p["id"].isin(last.index)
        fx = self.fixtures.copy()
        fx["finished"] = fx["event"].le(gw).fillna(False).astype(bool)
        return Dataset(p, self.teams, fx, hist, gw, gw + 1 if gw < 38 else None,
                       self.deadlines, False, f"{self.source}@GW{gw}")

    def name(self, pid: int) -> str:
        row = self.players.loc[self.players["id"] == pid]
        return row["web_name"].iloc[0] if len(row) else str(pid)


# ------------------------------------------------------------------- canonicalisation
def _col(df: pd.DataFrame, name: str, default):
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def _to_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: v if isinstance(v, (bool, np.bool_)) else str(v).strip().lower() == "true")


def canon_players(elements: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "id": elements["id"].astype(int),
        "web_name": elements["web_name"].astype(str),
        "team": elements["team"].astype(int),
        "pos": elements["element_type"].astype(int),
        "price": pd.to_numeric(elements["now_cost"]) / 10.0,
        "status": _col(elements, "status", "a").fillna("a").astype(str),
        "chance_next": pd.to_numeric(_col(elements, "chance_of_playing_next_round", np.nan), errors="coerce"),
        "news": _col(elements, "news", "").fillna("").astype(str),
        "selected_by_percent": pd.to_numeric(_col(elements, "selected_by_percent", 0), errors="coerce").fillna(0.0),
        "season_minutes": pd.to_numeric(_col(elements, "minutes", np.nan), errors="coerce"),
    })
    can_select = _to_bool(_col(elements, "can_select", True))
    removed = _to_bool(_col(elements, "removed", False))
    df["can_buy"] = can_select & ~removed & (df["status"] != "u")
    # element_type 5 was the 2025/26 Assistant Manager chip; it is not a player.
    return df[df["pos"].isin([1, 2, 3, 4])].reset_index(drop=True)


def canon_teams(teams: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"id": teams["id"].astype(int), "name": teams["name"].astype(str),
                         "short_name": teams["short_name"].astype(str)})


def canon_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "id": fixtures["id"].astype(int),
        "event": pd.to_numeric(fixtures["event"], errors="coerce").astype("Int64"),
        "team_h": fixtures["team_h"].astype(int),
        "team_a": fixtures["team_a"].astype(int),
        "finished": _to_bool(fixtures["finished"]),
        "kickoff_time": pd.to_datetime(fixtures["kickoff_time"], utc=True, errors="coerce"),
    })


def canonicalize_history(rows: pd.DataFrame, fixtures: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Turn element-summary `history` rows (or archive rows) into canonical form.

    The player's club for each match is taken from the fixture itself, so
    mid-season moves between Premier League clubs are handled correctly.
    """
    if rows.empty:
        return pd.DataFrame(columns=HIST_COLS)
    h = rows.copy()
    for c in HIST_NUM_COLS:
        h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0.0) if c in h.columns else 0.0
    h["element"] = h["element"].astype(int)
    h["fixture"] = h["fixture"].astype(int)
    h["round"] = h["round"].astype(int)
    h["was_home"] = _to_bool(h["was_home"])
    fx = fixtures.set_index("id")
    h = h[h["fixture"].isin(fx.index[fx["finished"]])]
    th, ta = h["fixture"].map(fx["team_h"]), h["fixture"].map(fx["team_a"])
    h["team"] = np.where(h["was_home"], th, ta).astype(int)
    h["opponent_team"] = np.where(h["was_home"], ta, th).astype(int)
    h["kickoff_time"] = pd.to_datetime(h["kickoff_time"], utc=True)
    h = h.merge(players[["id", "pos"]], left_on="element", right_on="id", how="inner").drop(columns="id")
    return h[HIST_COLS].sort_values(["element", "kickoff_time"]).reset_index(drop=True)


def _deadlines_from_fixtures(fx: pd.DataFrame) -> dict:
    # Approximation for archive data only: the live API gives exact deadlines.
    first = fx.dropna(subset=["event"]).groupby("event")["kickoff_time"].min()
    return {int(gw): t - pd.Timedelta(minutes=90) for gw, t in first.items() if pd.notna(t)}


# --------------------------------------------------------------------- archive source
def load_archive(season_dir: str | Path) -> Dataset:
    d = Path(season_dir)
    players = canon_players(pd.read_csv(d / "players_raw.csv"))
    teams = canon_teams(pd.read_csv(d / "teams.csv"))
    fx = canon_fixtures(pd.read_csv(d / "fixtures.csv"))
    hist = canonicalize_history(pd.read_csv(d / "gws" / "merged_gw.csv", low_memory=False), fx, players)
    last = int(hist["round"].max()) if len(hist) else 0
    return Dataset(players, teams, fx, hist, last, last + 1 if last < 38 else None,
                   _deadlines_from_fixtures(fx), True, f"archive:{d.name}")


# ------------------------------------------------------------------------ live source
class FPLAuthError(RuntimeError):
    pass


class FPLClient:
    """Thin, polite, caching client for the (unofficial) FPL API.

    Public endpoints need no login. `my_team` needs an access token, sent as
    `X-API-Authorization: Bearer <token>` (FPL's current OIDC-based login).
    """

    def __init__(self, cache_dir: str | Path = ".fpl_cache", access_token: str | None = None,
                 min_interval: float = 0.3, timeout: float = 20.0,
                 user_agent: str = "fpl-agent/0.1 (personal use)"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.access_token = access_token
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._last_call = 0.0

    def get(self, path: str, max_age: float | None = None, auth: bool = False):
        cache = self.cache_dir / (path.strip("/").replace("/", "_") + ".json")
        if max_age and cache.exists() and time.time() - cache.stat().st_mtime < max_age:
            return json.loads(cache.read_text(encoding="utf-8"))
        headers = {}
        if auth:
            if not self.access_token:
                raise FPLAuthError("This endpoint needs an access token (--access-token / FPL_ACCESS_TOKEN).")
            headers["X-API-Authorization"] = f"Bearer {self.access_token}"
        for attempt in range(5):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                r = self.session.get(f"{API_BASE}/{path.strip('/')}/", headers=headers, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                data = r.json()
                if not auth:  # never cache personal authenticated responses on disk
                    cache.write_text(json.dumps(data), encoding="utf-8")
                return data
            if r.status_code in (401, 403) and auth:
                raise FPLAuthError(f"FPL rejected the access token ({r.status_code}). It has probably expired.")
            if r.status_code in (429, 500, 502, 503, 504):   # rate limited or "game is updating"
                time.sleep(2 * 2 ** attempt)
                continue
            r.raise_for_status()
        raise RuntimeError(f"FPL API kept failing for {path}")

    def bootstrap(self, max_age=1800): return self.get("bootstrap-static", max_age)
    def fixtures(self, max_age=1800): return self.get("fixtures", max_age)
    def element_summary(self, pid, max_age=6 * 3600): return self.get(f"element-summary/{pid}", max_age)
    def entry(self, eid, max_age=600): return self.get(f"entry/{eid}", max_age)
    def entry_history(self, eid, max_age=600): return self.get(f"entry/{eid}/history", max_age)
    def entry_transfers(self, eid, max_age=600): return self.get(f"entry/{eid}/transfers", max_age)
    def picks(self, eid, gw, max_age=600): return self.get(f"entry/{eid}/event/{gw}/picks", max_age)
    def live(self, gw, max_age=600): return self.get(f"event/{gw}/live", max_age)
    def my_team(self, eid): return self.get(f"my-team/{eid}", None, auth=True)


def load_live(client: FPLClient, refresh: bool = False, verbose: bool = True) -> Dataset:
    bs = client.bootstrap(max_age=None if refresh else 1800)
    players = canon_players(pd.DataFrame(bs["elements"]))
    teams = canon_teams(pd.DataFrame(bs["teams"]))
    fx = canon_fixtures(pd.DataFrame(client.fixtures(max_age=None if refresh else 1800)))
    events = pd.DataFrame(bs["events"])
    finished = events[events["finished"].astype(bool)]
    last = int(finished["id"].max()) if len(finished) else 0
    nxt = events[events["is_next"].astype(bool)]
    next_gw = int(nxt["id"].iloc[0]) if len(nxt) else None
    deadlines = {int(r.id): pd.Timestamp(r.deadline_time) for r in events.itertuples()}

    # Players with zero minutes this season contribute nothing to the model, so
    # their history is skipped (saves ~30% of requests). The predictor knows
    # this from `season_minutes == 0`.
    ids = players.loc[players["season_minutes"].fillna(1) > 0, "id"].tolist()
    rows = []
    for i, pid in enumerate(ids):
        js = client.element_summary(pid)
        for r in js.get("history", []):
            r.setdefault("element", pid)
            rows.append(r)
        if verbose and (i + 1) % 100 == 0:
            print(f"  fetched {i + 1}/{len(ids)} player histories")
    hist = canonicalize_history(pd.DataFrame(rows), fx, players)
    return Dataset(players, teams, fx, hist, last, next_gw, deadlines, True, "live")


# ------------------------------------------------------------------------ team state
@dataclass
class TeamState:
    squad: list[int]
    selling_price: dict[int, float]
    bank: float
    free_transfers: int
    next_gw: int
    notes: list[str] = field(default_factory=list)
    ft_assumed: bool = False          # True when the FT count is a safe default, not known
    ft_estimate: int | None = None    # what the history replay suggested


def selling_price(purchase_tenths: int, now_tenths: int) -> int:
    """FPL keeps half of any price rise, rounded down to £0.1m; falls are passed on in full."""
    return now_tenths if now_tenths <= purchase_tenths else purchase_tenths + (now_tenths - purchase_tenths) // 2


def estimate_free_transfers(history_current: list[dict], chips: dict[int, str], started: int, upto_gw: int) -> int:
    """Replay the transfer history to estimate free transfers for gameweek upto_gw + 1.

    Rules (2024/25 onward): +1 per gameweek, bank up to 5. When a Wildcard or
    Free Hit is played, the saved count is kept as it was, with NO +1 for the
    next gameweek ("If you had 2 saved free transfers, you will still have 2").

    This is only an estimate: anything FPL doesn't expose in the history (a
    special top-up, a rule change) makes it wrong, so callers must not trust it
    blindly.
    """
    by_gw = {h["event"]: h for h in history_current}
    ft = 1
    for g in range(started + 1, upto_gw + 1):
        if chips.get(g) in ("wildcard", "freehit"):
            continue                       # count carries over unchanged
        made = by_gw.get(g, {}).get("event_transfers", 0)
        ft = min(MAX_FREE_TRANSFERS, max(0, ft - made) + 1)
    return ft


def team_state_from_my_team(js: dict, next_gw: int) -> TeamState:
    """Exact state from the authenticated my-team endpoint."""
    squad = [p["element"] for p in js["picks"]]
    sp = {p["element"]: p["selling_price"] / 10.0 for p in js["picks"]}
    tr = js.get("transfers", {}) or {}
    limit, made = tr.get("limit"), tr.get("made", 0) or 0
    notes = []
    if limit is None:
        ft = MAX_FREE_TRANSFERS
        notes.append("FPL reports unlimited transfers this week (chip active?); treating as 5.")
    else:
        ft = max(0, int(limit) - int(made))
    return TeamState(squad, sp, tr.get("bank", 0) / 10.0, ft, next_gw, notes)


def team_state_public(client: FPLClient, entry_id: int, ds: Dataset, free_transfers: int | None = None) -> TeamState:
    """State from public endpoints only (no login).

    Selling prices are reconstructed from your transfer history. This reflects
    your squad at the LAST deadline: transfers already made for the coming
    gameweek are invisible here (use an access token for exact state).
    """
    notes = []
    entry = client.entry(entry_id)
    hist = client.entry_history(entry_id)
    started = int(entry.get("started_event") or 1)
    cur = int(entry.get("current_event") or ds.last_finished_gw)
    chips = {c["event"]: c["name"] for c in hist.get("chips", [])}
    squad_gw = cur - 1 if chips.get(cur) == "freehit" else cur   # Free Hit squads revert
    picks = client.picks(entry_id, squad_gw)
    squad = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10.0

    fh_events = {e for e, n in chips.items() if n == "freehit"}
    bought: dict[int, int] = {}
    for t in sorted(client.entry_transfers(entry_id), key=lambda t: t["time"]):
        if t["event"] in fh_events or t["event"] > cur:
            continue
        bought[t["element_in"]] = int(t["element_in_cost"])

    price_now = dict(zip(ds.players["id"], (ds.players["price"] * 10).round().astype(int)))
    sp = {}
    for pid in squad:
        if pid in bought:
            buy = bought[pid]
        else:  # part of the original squad: price at the start of the manager's first gameweek
            rows = ds.history[(ds.history["element"] == pid) & (ds.history["round"] >= started)]
            if len(rows):
                buy = int(rows.sort_values("kickoff_time")["value"].iloc[0])
            else:
                buy = price_now.get(pid, 0)
                notes.append(f"Purchase price unknown for {ds.name(pid)}; using current price.")
        sp[pid] = selling_price(buy, price_now.get(pid, buy)) / 10.0

    assumed, est = False, None
    if free_transfers is None:
        # Overestimating free transfers turns a "free" transfer into a real -4,
        # so plan with the guaranteed minimum (everyone has at least 1) unless told otherwise.
        est = estimate_free_transfers(hist.get("current", []), chips, started, cur)
        free_transfers, assumed = 1, True
        if est > 1:
            notes.append(f"History suggests you may have {est} free transfers, but that can't be confirmed "
                         f"without a login. Planning with 1 so a wrong guess can't cost you -4. "
                         f"Check the Transfers page and re-run with --ft N if you have more.")
    notes.append("Squad is as of the last deadline; transfers already made this week are not visible without a token.")
    return TeamState(squad, sp, bank, int(free_transfers), cur + 1, notes, assumed, est)


def resolve_players(names: list[str], players: pd.DataFrame) -> list[int]:
    """Map names (case-insensitive web_name) or numeric ids to player ids."""
    out = []
    for raw in names:
        s = raw.strip()
        if not s:
            continue
        if s.isdigit():
            out.append(int(s))
            continue
        m = players[players["web_name"].str.lower() == s.lower()]
        if len(m) == 0:
            m = players[players["web_name"].str.lower().str.contains(s.lower(), regex=False)]
        if len(m) != 1:
            opts = ", ".join(f"{r.web_name}({r.id})" for r in m.head(8).itertuples()) or "no match"
            raise ValueError(f"'{s}' is ambiguous or unknown: {opts}. Use the player id instead.")
        out.append(int(m["id"].iloc[0]))
    return out
