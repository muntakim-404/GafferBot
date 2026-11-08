# fpl-agent: predictor, transfer planner and Telegram bot (steps 1–3)

Predicts expected points (xPts) for every Premier League player over the next few
gameweeks, then solves for the best transfers, starting XI and captain under all FPL
rules. Step 3 runs it on GitHub on a schedule and sends you the plan on Telegram before
each deadline (see "Step 3" below). It only reads from FPL and never changes your team.

## Install

Needs **Python 3.10 or newer** (`python --version`). Run all commands from the folder
that contains `requirements.txt` (Windows "Extract All" often creates
`fpl-agent\fpl-agent` — `cd` into the inner one).

**Windows (PowerShell)** — run one line at a time:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation fails with *"running scripts is disabled on this system"*, run this
once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

(Or skip activation and use `.venv\Scripts\python.exe` in place of `python` everywhere.)

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PuLP is pinned below 3.0 on purpose: PuLP 3/4 rewrote its API.

## Use it

Your team id is the number in the URL of your Points page
(`fantasy.premierleague.com/entry/<TEAM_ID>/event/...`).

```bash
# Recommended transfers/XI/captain for the next gameweek, planned over 5 weeks
python -m fpl_agent plan --team-id 1234567

# Common options
python -m fpl_agent plan --team-id 1234567 --ft 2            # override free-transfer count
python -m fpl_agent plan --team-id 1234567 --no-hits         # never take -4
python -m fpl_agent plan --team-id 1234567 --max-transfers 1
python -m fpl_agent plan --team-id 1234567 --lock Saka --ban Haaland
python -m fpl_agent plan --team-id 1234567 --out Isak --starter Saliba   # your team-news knowledge
python -m fpl_agent plan --team-id 1234567 --json plan.json  # machine-readable plan for the agent layer
python -m fpl_agent plan --from-scratch                      # ideal £100m squad (wildcard view)

# Top players by position over the horizon
python -m fpl_agent predict --horizon 5 --top 10
```

The first live run fetches ~500 player histories (≈3 minutes at a polite rate), then
caches them in `.fpl_cache/` for 6 hours.

**Team state.** Without a login, the tool rebuilds your squad, bank and selling
prices from public endpoints. Two limits: it sees your squad as of the *last
deadline* (transfers you've already made this week are invisible), and it can't see
your free-transfer count. **Without `--ft` it plans with 1 free transfer**, the
guaranteed minimum, so a wrong guess can never cost you a −4. It prints what your
history suggests; check the Transfers page and pass `--ft N` if you have more. If you have an FPL access token, pass `--access-token` (or set
`FPL_ACCESS_TOKEN`: `$env:FPL_ACCESS_TOKEN="..."` in PowerShell,
`export FPL_ACCESS_TOKEN=...` in bash) to read your exact team, selling prices and free transfers. The
token is sent as `X-API-Authorization: Bearer …`, is never written to disk, and
expires after about an hour.

Nothing here makes transfers on your account. It only reads.

## How it works

**Data** (`data.py`). Public FPL API: `bootstrap-static` (players, prices, injury
flags), `fixtures`, and `element-summary/{id}` (per-match history incl. xG/xA and
defensive actions). An archive loader reads past seasons in the
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
layout; both sources go through the same canonicalisation code.

**Predictor** (`predictor.py`). xPts per player per gameweek is the sum of:

| Component | How it's estimated |
|---|---|
| Minutes | P(plays), P(60+), expected minutes from the last 6 team fixtures, recency-weighted, skipping games a regular starter missed through injury; × availability from FPL's injury flag and "Expected back <date>" news |
| Goals, assists | Per-minute xG/xA over recent appearances, shrunk toward a price-based prior; × opponent's defensive weakness × venue; calibrated to actual goals/assists |
| Clean sheet, goals conceded | Poisson on (own defence × opponent attack × venue) |
| Saves | Per-minute saves × opponent attack, −/+ via Poisson floor(saves/3) |
| Defensive contributions | Recent rate of reaching the threshold (10 DEF / 12 MID-FWD) when playing 60+ |
| Bonus, cards | Per-minute rates, shrunk toward priors |

Doubles are summed; blanks are 0. Scoring values live in `config.py` (checked for 2026/27).

**Solver** (`solver.py`). A mixed-integer program over the horizon (default 5 GWs,
each later week weighted 0.85×): squad, XI, captain and transfers for every week;
£ budget with your real selling prices; 2/5/5/3; max 3 per club; valid formations;
free transfers bank up to 5; hits cost 4 (+1 safety margin, see below); ending free
transfers are worth 1.5 points each so it rolls when nothing is worth doing. Every
plan is re-checked by an independent validator before it's shown.

## Backtest results (honest numbers)

Walk-forward on the full **2025/26** season, GW5–38 (34 gameweeks). For each gameweek
the model only sees earlier data. `top11` = actual points of the 11 highest-predicted
players (formation ignored); the hindsight-perfect top 11 averaged 140.

| Predictor | top11 | captain | Spearman | MAE |
|---|---|---|---|---|
| **This model (v2)** | **54.6** | **6.7** | **0.750** | **0.978** |
| This model (v1, first release) | 52.7 | 6.8 | 0.749 | 0.965 |
| Season average | 47.6 | 6.5 | 0.691 | 1.055 |
| Last-4 form | 43.4 | 4.6 | 0.728 | 1.036 |

Run it yourself: `python -m fpl_agent backtest --archive data/2025-26 --start-gw 5`.

**v2 changes** (team-strength memory over the whole season; absences of regular
starters no longer count against their minutes; FPL return dates used for injured
players). Average top-11 points per gameweek, GW5–38:

| Season | v1 | v2 | |
|---|---|---|---|
| 2025/26 | 52.7 | 54.6 | used for tuning |
| 2024/25 | 62.1 | 64.0 | used for tuning |
| 2023/24 | 59.7 | 60.5 | **untouched holdout**: the honest estimate of the gain |

v1 lost to the season-average baseline in most second-half weeks of 2025/26 (53.3 vs
51.2 average); v2 averages 56.3 there. Captain picks got slightly worse in all three
seasons (−0.1 to −0.5 points), so captaincy remains the weakest call.

What this does and doesn't show:
- It beats the naive baselines on every ranking metric, but the margin over "season
  average" is modest (~7 points per gameweek on 11 players). This is a baseline, not an edge.
- The backtest has no injury flags (they aren't archived), so the live model, which
  uses them, should do somewhat better than this.
- **It over-predicts its top picks**: players predicted 5.5+ averaged 6.9 predicted vs
  6.0 actual. That inflates the apparent gain from transfers, which is why hits carry
  an extra 1-point margin (`hit_margin` in `config.py`).
- FPL's own `xP` column in the archive is **not** a usable benchmark: it correlates
  0.75–0.83 with actual points and equals the actual score in up to half the rows, so
  it's recorded after matches.

## Known limitations (in order of how much they likely cost)

1. **Captaincy**: improved by the captain study (see "Captaincy"), but when the top two
   captain scores are within about a point, the choice is still close to a coin flip.
   The model ignores what other managers do; that's about rank, not points.
2. **Injury returns are approximate**: games a regular starter missed through injury no
   longer count against him, and "Expected back <date>" news sets the return gameweek.
   With no date, availability ramps up 25% per gameweek (a guess). A player who was
   dropped (not injured) and has no flag is still treated as dropped, which is usually
   right. Use `--starter NAME` / `--out NAME` when you know better.
3. **Early season (GW1–4) is weak**: there's no carry-over from last season yet.
4. **No price-change modelling, no chips, flat bench weight.**
5. **Team ratings ignore schedule strength** (a team that played tough opponents looks worse).
6. **The live HTTP path was not run against the real API during development**
   (FPL was unreachable from the build environment). It was tested with a fake client
   serving real 2026/27 data in the API's exact JSON shape, and produced identical
   results to the archive path. Expect to fix small things on your first live run.

## Tests

```bash
# Solver + rules tests (no data needed)
python -m pytest -q tests/test_solver.py
# Data-layer tests against a real archived season
python -m fpl_agent download-archive --season 2026-27
FPL_ARCHIVE=data/2026-27 python -m pytest -q tests   # bash
```

```powershell
$env:FPL_ARCHIVE="data\2026-27"; python -m pytest -q tests  # PowerShell
```

Archive data: `python -m fpl_agent download-archive --season 2025-26` saves the four
files the tool needs from the vaastav repository into `data/2025-26/`.

## Step 3: automatic plans on Telegram (GitHub Actions)

GitHub is asked to start a run every 30 minutes. Most runs end within a minute. The
first run within **24 hours** of a deadline sends an *early look*, and the first within
**4 hours** a *final check* (each once). The windows are wide on purpose: in practice
GitHub can leave several hours between scheduled runs (see "Reliable timing").
Change them with the repository variables `FPL_EARLY_HOURS` / `FPL_FINAL_HOURS`.
After every gameweek it scores what its plan would have got against what you actually
got, and keeps the running total in `logs/shadow.csv`. That record is how you decide
whether to trust it with automatic transfers (step 5). It only reads from FPL and never
changes your team.

Reply to the bot (picked up at its next run, usually within an hour):
`/ft N` sets your free transfers for the coming gameweek · `/plan` sends a plan at the
next run · `/status` · `/help`.

### 1. Telegram bot
1. In Telegram, message **@BotFather** → `/newbot` → pick a name and a username ending
   in `bot`. Copy the **token** it gives you and keep it private.
2. Open your new bot and send it any message (e.g. `hi`).
3. On your PC:
   ```powershell
   $env:TELEGRAM_BOT_TOKEN="<your token>"
   python -m fpl_agent telegram-setup
   ```
   It prints your **chat id** and sends a test message to Telegram.

### 2. Try it on your PC first
```powershell
$env:FPL_TEAM_ID="<your team id>"
$env:FPL_TIMEZONE="Asia/Dhaka"
python -m fpl_agent tick --dry-run --force     # prints the message instead of sending
$env:TELEGRAM_CHAT_ID="<chat id>"
python -m fpl_agent tick --force               # really sends it
```

### 3. Put the project on GitHub (public repo)
1. Create a GitHub account, install **GitHub Desktop** and sign in.
2. **File → Add local repository** → choose this folder → it says it isn't a git
   repository → click **create a repository** → **Create repository**.
   `.gitignore` keeps `.venv`, `data` and caches out.
3. **Publish repository** → **untick "Keep this code private"** → Publish.
   (Free GitHub accounts currently can't reliably run scheduled jobs in private repos.
   Your secrets stay hidden either way; your code and logs become public.)

### 4. Add the secrets
On github.com → your repository → **Settings → Secrets and variables → Actions**:
- **Secrets** tab → *New repository secret*, three times:
  `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `FPL_TEAM_ID`
- **Variables** tab → *New repository variable*: `FPL_TIMEZONE` = `Asia/Dhaka`

### 5. Test run
**Actions** tab → enable workflows if asked → **fpl-agent** → **Run workflow** → tick
*Send a plan now* → **Run workflow**. It takes about 4 minutes; the plan should arrive in Telegram.

If it fails, click the run to read the log:
- *Can't reach FPL* (also sent to Telegram): FPL may be blocking GitHub's servers.
  That would need a different host.
- *Permission denied* when pushing: **Settings → Actions → General → Workflow
  permissions → Read and write permissions**.

### Reliable timing (recommended, free, ~10 minutes)
GitHub treats scheduled runs as best-effort. On this project's first week, gaps of 4–8
hours between runs were observed. A free outside timer that *asks* GitHub to run the
workflow fixes that, because requested runs start right away.

1. **GitHub token:** github.com → your profile picture → **Settings → Developer settings
   → Personal access tokens → Fine-grained tokens → Generate new token**.
   - Repository access: **Only select repositories** → your bot's repository.
   - Permissions → Repository permissions → **Actions: Read and write**. Nothing else.
   - Copy the token (it starts with `github_pat_`). It can only start/stop this
     repository's workflow runs; it can't read your secrets or change your code.
2. **cron-job.org:** create a free account → **Create cronjob**:
   - URL: `https://api.github.com/repos/<your-username>/<your-repo>/actions/workflows/fpl-agent.yml/dispatches`
   - Schedule: every hour (e.g. at minute 20).
   - Advanced → Request method **POST**; headers
     `Authorization: Bearer <your github_pat token>`,
     `Accept: application/vnd.github+json`, `X-GitHub-Api-Version: 2022-11-28`;
     request body `{"ref":"main"}`.
   - Save, then **Test run**: a new run should appear in your Actions tab within
     seconds (GitHub answers `204`).
3. Keep the GitHub schedule as a backup. Duplicate runs are harmless: the bot never sends
   the same message twice.

### Day to day
- **No early message by 18 hours before a deadline, or no final by 3 hours?** Check the
  Actions tab, and set up "Reliable timing" above if you haven't.
- **Changing the code later:** the bot commits its logs to GitHub, so in GitHub Desktop
  click **Fetch/Pull origin first**, then copy in new files, **Commit**, **Push origin**.
- Scheduled jobs in public repos stop after 60 days without commits. The bot's own
  log commits each gameweek should keep it running.

## Chips: when to play Triple Captain, Bench Boost and Free Hit

Each plan message has a **Chips** section, and on your PC:

```powershell
python -m fpl_agent chips --team-id <your id>
```

prints, for every gameweek left in the current half (first half: up to GW19), the
expected *extra* points from each chip:
- **Triple Captain:** your captain's expected points.
- **Bench Boost:** your bench's expected points.
- **Free Hit:** the best one-week squad you could afford, minus your own XI.

It uses the squad your transfer plan expects you to own that week, reads which chips
you've already used from FPL, and gives each remaining chip its own week (FPL allows
one chip per gameweek).

**A week only "stands out" if it beats the next best by a clear margin** (Triple Captain
1.5 points; Bench Boost and Free Hit 2). Otherwise it says *hold*. Week-to-week
differences are usually smaller than the model's error, and waiting keeps the option of
a double gameweek or better news. It only says *play this week* when this week stands
out, or when there are no more weeks than chips left. Double gameweeks are where chips
earn the most, and they show up as clear standouts (tested with a fake extra Man City
fixture: Triple Captain on Haaland jumped from about +7 to +14).

Not validated against past seasons. Triple Captain uses the same captain rule as below.
Treat it as a guide and re-check every week.

## Captaincy

**Rule:** only midfielders and forwards can captain, ranked by a *captain score*. That is
the model's expected points corrected for each position's forecast bias: forwards' high
forecasts run hottest (8.0 expected becomes 6.8), midfielders' only slightly (8.0 becomes
7.5). Messages show the top three captain scores. When the top two are within a point,
they warn "close call".

**Evidence** (`python -m fpl_agent captain-study --archive data/2023-24 --archive data/2024-25 --archive data/2025-26`):
102 gameweeks, walk-forward, the calibration scored on seasons it wasn't fitted to.
Average captain points per gameweek (a non-playing captain is replaced by the rule's vice):

| Rule | 2023/24 | 2024/25 | 2025/26 | All | vs old rule |
|---|---|---|---|---|---|
| Highest expected points (old rule) | 7.59 | 8.85 | 6.68 | 7.71 | |
| Highest expected points, MID/FWD only | 7.59 | 8.85 | 7.38 | 7.94 | +0.24 ± 0.16 |
| **Corrected, MID/FWD only (current)** | 7.82 | 9.68 | 7.29 | **8.26** | **+0.56 ± 0.29** |
| Most-owned MID/FWD (follow the crowd) | 6.47 | 7.12 | 6.15 | 6.58 | −1.13 ± 0.57 |

- The current rule won in every season, and in all 12 variations of the study's own
  settings (gain +0.3 to +0.6). That makes it **probably better, not proven**: it's
  about 2 standard errors, and six rules were tried.
- Following the crowd's captain cost about 1.1 points a week in expectation. Doing so
  protects your rank against everyone who owns that player; it doesn't score more.
- Applying the same correction to transfer decisions made no difference (+0.08 ± 0.43),
  so it's used for the captain only.
- Refit after any change to the predictor: the command prints the new
  `CAPTAIN_CALIBRATION` for `config.py`.

## Layout

```
fpl_agent/
  config.py     rules, scoring, tunable parameters
  data.py       API client, archive loader, team state, selling prices
  predictor.py  xPts model
  solver.py     multi-week MILP + validator
  chips.py      chip planner (Triple Captain, Bench Boost, Free Hit)
  captaincy.py  captain study: compare captain rules, refit the calibration
  backtest.py   walk-forward evaluation
  agent.py      scheduled tick: deadline windows, Telegram commands, shadow scoring
  telegram.py   Telegram client (token never printed)
  cli.py        `python -m fpl_agent ...`
.github/workflows/fpl-agent.yml   the scheduled GitHub Actions job
logs/           plans sent and the shadow log (written by the bot)
state/          the bot's memory between runs (written by the bot)
tests/          solver rules, data layer, predictor, agent
```

## Next steps

4. LLM news layer: turn press-conference news into `--out` / `--starter` overrides.
5. Autonomous fallback with guardrails (free transfers only, injured starters only, no
   chips), only after the shadow log shows the plans are at least as good as yours.
