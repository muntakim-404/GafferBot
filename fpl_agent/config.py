"""Game rules (2026/27) and model/solver parameters.

Scoring values were checked against Fantasy Football Scout's 2026/27 scoring
table. Re-check them at the start of every season: if FPL changes a rule, this
is the only file that should need editing.
"""
from dataclasses import dataclass, field

GK, DEF, MID, FWD = 1, 2, 3, 4
POS_NAME = {GK: "GK", DEF: "DEF", MID: "MID", FWD: "FWD"}

# ---- Scoring -----------------------------------------------------------------
GOAL_PTS = {GK: 10, DEF: 6, MID: 5, FWD: 4}
ASSIST_PTS = 3
CLEAN_SHEET_PTS = {GK: 4, DEF: 4, MID: 1, FWD: 0}   # needs 60+ minutes
CONCEDED_PER_2 = {GK: -1, DEF: -1, MID: 0, FWD: 0}
SAVES_PER_3 = 1                                      # goalkeepers only
APPEAR_PTS = 1                                       # any minutes
APPEAR_60_BONUS = 1                                  # extra for 60+ minutes
YELLOW_PTS, RED_PTS = -1, -3
DC_THRESHOLD = {GK: None, DEF: 10, MID: 12, FWD: 12}  # defensive-contribution actions
DC_PTS = 2

# ---- Squad rules -------------------------------------------------------------
SQUAD_COUNT = {GK: 2, DEF: 5, MID: 5, FWD: 3}
LINEUP_MIN = {GK: 1, DEF: 3, MID: 2, FWD: 1}
LINEUP_MAX = {GK: 1, DEF: 5, MID: 5, FWD: 3}
MAX_PER_CLUB = 3
START_BUDGET = 100.0
MAX_FREE_TRANSFERS = 5
HIT_COST = 4

# ---- Captaincy (see `python -m fpl_agent captain-study`) ---------------------
# Only midfielders and forwards may captain, ranked by a "captain score": the
# model's xPts corrected for each position's forecast bias (actual ≈ a + b·xPts,
# fitted on 2023/24-2025/26 predictions with xPts >= 3). Out of sample this beat
# the plain "highest xPts" rule in all three seasons (+0.3 to +0.6 captain
# points per gameweek). Forwards' high forecasts run hottest: 8.0 xPts -> 6.8.
# Refit after any change to the predictor.
CAPTAIN_CALIBRATION = {MID: (-0.015, 0.936), FWD: (0.690, 0.763)}


def captain_score(pos: int, xpts: float) -> float | None:
    """Bias-corrected captain value, or None if that position may not captain."""
    cal = CAPTAIN_CALIBRATION.get(pos)
    return None if cal is None else cal[0] + cal[1] * xpts


@dataclass
class ModelParams:
    """Knobs for the baseline predictor. Values marked (guess) are untuned."""
    # Minutes model: recency-weighted over the player's last N team fixtures.
    minutes_window: int = 6
    minutes_halflife: float = 2.0
    # Absences: a run of 0-minute games by a regular starter is treated as an
    # injury/suspension (not a loss of place) and left out of the minutes
    # estimate, if he has since played again or is currently flagged.
    # Availability is handled separately, so counting the zeros too would
    # punish the injury twice.
    exclude_absences: bool = True
    regular_lookback: int = 4        # fixtures before the absence used to judge "regular"
    regular_share: float = 0.75      # share of those with 60+ minutes
    # Per-90 rates: recency-weighted over the last N appearances.
    rate_window: int = 12
    rate_halflife: float = 5.0
    rate_prior_minutes: float = 360.0      # shrinkage strength toward the price-based prior (guess)
    card_prior_minutes: float = 900.0
    # Bonus and defensive-contribution hit rate: prior strength in matches.
    bonus_prior_matches: float = 4.0
    dc_prior_matches: float = 4.0
    # Team attack/defence ratings. Long memory: tuned on 2025/26 + 2024/25 and
    # confirmed on untouched 2023/24 (+0.9 pts/GW on top-11, both season halves).
    team_window: int = 38
    team_halflife: float = 20.0
    team_prior_matches: float = 4.0
    venue_prior_fixtures: float = 40.0
    default_home_factor: float = 1.10      # (guess) used until enough fixtures exist
    default_away_factor: float = 0.90
    # Availability recovery per gameweek ahead for flagged players (guess),
    # used when FPL's news gives no return date.
    injury_recovery_per_gw: float = 0.25
    # When the news says "Expected back <date>" / "Suspended until <date>":
    # 0 before that gameweek, this in the return gameweek, 1.0 after (guess).
    return_week_avail: float = 0.75
    # Prior for players with no history rows at all (new signings) (guess).
    no_history_p_any: float = 0.5
    no_history_p60: float = 0.35


@dataclass
class SolverParams:
    horizon_decay: float = 0.85    # weight of gameweek w is decay**(w-1)
    bench_weight: float = 0.10     # value of bench xPts relative to starters
    ft_value: float = 1.5          # value of each free transfer still banked at horizon end
    hit_cost: float = HIT_COST
    # Extra penalty per hit on top of the real 4 points. The backtest shows the
    # model over-predicts its top picks (~6.9 predicted vs ~6.0 actual), so the
    # gain from a transfer is inflated; a hit must clear 4 + margin. (guess)
    hit_margin: float = 1.0
    allow_hits: bool = True
    max_transfers_first_gw: int | None = None
    max_transfers_per_gw: int | None = None
    lock: list[int] = field(default_factory=list)   # player ids that must stay/be in the squad
    ban: list[int] = field(default_factory=list)    # player ids that must not be in the squad
    pool_per_position: int = 60                     # candidate pruning for speed
    time_limit: float = 120.0
    budget: float = START_BUDGET                    # only used when building from scratch
