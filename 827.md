# Team 827: frame-level possession from automatic tracking

More Than A Hack 2027, Phase 1. This document explains how `827.csv` was produced. The code to
regenerate it is in this repository, and the steps are in [Reproduction](#reproduction).

## Approach

The pipeline has four parts: team-level features, three gradient-boosted models, a small Smart
Tagging adjustment, and a decoder that produces whole possessions.

1. **Features** (`src/features.py`). Five rows a second, summed over each team, because the 46
   track slots do not follow people. Directions are measured towards the goal each team attacks,
   worked out per half from where the teams stand. Possession persists through loose balls, so the
   last and next moments of apparent control are carried across ball gaps, and windows look back
   and ahead separately.
2. **Three models** (`src/model.py`). LightGBM heads for "is the ball dead", "which team has it" and
   "is a change near". The dead-ball head sees only player columns.
3. **Smart Tagging** (`src/tags.py`). A small nudge on the team odds, described below.
4. **Decoder** (`src/decode.py`). Viterbi over home, away and dead ball, in integer arithmetic. Dead
   ball has two hidden states, so a restart can cost differently for each team.

**Why this rather than something else.** Spell-based scoring rewards whole possessions, so we
tested what actually limits it. On external matches, perfect team calls were worth far more than
perfect dead-ball calls. More elaborate decoders added nothing once evidence was noisy, and deep
temporal models had too little data and no evidence of surviving the move from clean to noisy
tracking.

The players-only dead-ball head came from a failure. Our first model read the ball when judging
dead ball, and its calls did not transfer between noise simulators, because each simulator treats a
dead ball differently. Using only the players raised alive-or-dead accuracy from 0.58-0.69 to 0.90
on a simulator the model had not trained on.

## Training data

No labels from the provided game were used, by hand or otherwise, and nothing was trained, fitted,
calibrated or tuned on it, on its video or on its ground truth. All training and tuning used
external data.

| Dataset | Use | Source and licence | Labels |
|---|---|---|---|
| IDSSE, 7 German matches | Training, tuning | [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0 (Bassek et al., 2025) | Provider's per-frame ball status and possession |
| Metrica sample games 1 and 2 | Training | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), no licence file, data not redistributed, credited here | Derived from the provider's events |
| Metrica sample game 3 | Final test only | As above | As above |
| SkillCorner, 5 matches | Real-tracking development check | [SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT | Provider's phases and detections |
| SkillCorner, 15 matches | Final test only | As above | As above |

We did not hand-label anything on any game. All these datasets are fetched by script at pinned
versions and are not in this repository.

**Bridging clean and noisy tracking.** The training matches are far cleaner than the provided
tracking, so every match is degraded by two simulators written independently of each other
(`src/noise.py`, `src/noise_b.py`), at two noise levels, and with the teams swapped. Neither
simulator reads labels: permuting them leaves the output byte-identical. Noise settings were not
measured from the provided match. A test harness scores each candidate by training on one simulator
and testing on the other, so a model cannot pass by learning one simulator's habits.

## Assets used and dropped

| Asset | Decision |
|---|---|
| Tactical tracking | Used, the main input |
| Smart Tagging | Used at a small fixed weight |
| Homography | Not used |
| Tactical and broadcast video | Not used |
| Broadcast tracking | Not used |
| Public 5-minute ground truth | Inspected by a person to understand the labels, as the organisers allow. Not used for training, tuning or choosing between versions |

**Smart Tagging.** Phases name a team. Attacking phases point to that team, defending phases to the
other, and tags without a team say nothing. No other match has these tags, so the weight was chosen
on simulated tags on external matches. A weight of 0.5 did no harm at 55% reliability and added
about 0.01 at higher reliabilities. Heavier weights hurt, because team errors sit at handovers that
a roughly timed phase cannot pinpoint.

**Dropped after testing.** Ball-touch and likely-receiver features gained 0.01-0.02 on simulated
noise and lost 0.011 on real tracking, in a paired test. That was within noise and did not justify
the added complexity. Our own ball detection, whistle detection and broadcast alignment were not
pursued: restoring every missing ball frame with the true position lifted the plain baseline to
only about 0.27-0.28.

## Possession rules

- **Dead ball** comes from how the players behave: speed, players on the lines, formation.
- **Loose balls** stay with the last team, because possession is carried through gaps.
- **Short touches.** On training data, a spell under one second between two spells of the other
  team is given back to that team, following the challenge's statement that a simple touch or
  deflection is not a change of possession. A person inspecting the public window saw short,
  scrappy tackles of a few seconds credited as possession, so the limit was kept small. The limit
  was set from the definition, not fitted to any statistic of the ground truth.
- **Restarts.** The decoder has separate costs for the same team and the other team restarting.
- **Before the second-half kickoff**, the published graded frames are labelled dead ball.
- **Abstention.** We write no `X` on graded frames: on the ranking metric it equals repeating the
  previous label, so it gains nothing.

## Results

All scores are on external data and held out at the level of whole matches. **They are not forecasts
of the score on the provided match.**

| Test | Our model | Baseline |
|---|---|---|
| Real automatic tracking: 5 SkillCorner matches, no simulator | 0.618 | 0.414 |
| Train on one simulator, test on the other | 0.355 and 0.479 | 0.241 and 0.226 |
| Train on both, test on each, 9 matches held out in turn | 0.454 and 0.507 | 0.241 and 0.226 |
| Same, with ball gaps five times longer | 0.475 | 0.199 |
| Five games of one club held out together | 0.430 and 0.464 | 0.237 and 0.216 |
| German matches only, tested on Metrica | 0.522 and 0.575 | 0.247 and 0.261 |
| Metrica games only, tested on German matches | 0.400 and 0.435 | 0.239 and 0.217 |

The model beat the baseline on all nine held-out matches under both simulators. SkillCorner's
broadcast tracking has a filled-in ball, longer spells and its own labels, which is why even the
baseline scores higher there.

We have no score on the provided match, beyond confirming that its file is valid and plausible.

## Limitations

- **Noise is simulated.** We could not measure how the provided tracking fails, so we trained across
  a wide range. The real pattern may differ.
- **Very short possessions are mostly missed.** The model predicts fewer possessions than the truth
  on external matches, and spells under two seconds are rarely recovered.
- **Team calls weaken in long ball gaps.**
- **Label conventions.** Training labels come from other providers. Training on German matches and
  testing on Metrica's cost at most about 0.02, but the provided match's labels may differ further.
- **With more time** we would run a real tracker over open video with trusted labels, to learn noise
  from data rather than simulate it.

## Reproduction

Requires [uv](https://docs.astral.sh/uv/) and the match bundle unpacked into `data/`, as in
`data/README.md`. Python and every package are pinned in `uv.lock`.

```bash
uv sync
uv run python -m src.submit     # writes evaluation/results/827.csv from the saved weights
uv run pytest                   # invented data only
```

The weights in `models/v1` are trained only on external matches. To retrain them:

```bash
uv run python -m src.external.idsse download
uv run python -m src.external.idsse cache
uv run python -m src.external.metrica download
uv run python -m src.external.metrica cache
uv run python -m src.train_final
```

Regeneration reads saved weights, because LightGBM only promises repeatable training on one system.
Prediction is single-threaded, features are rounded, and the decoder uses integers. Retraining may
not reproduce the weights byte for byte on another machine.

LLMs assisted with writing code. No match data or results were shown to a hosted LLM: checks on the
provided match ran through `src/blind.py`, which returns only pass or fail.
