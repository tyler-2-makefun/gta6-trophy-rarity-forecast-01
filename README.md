# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tyler-2-makefun/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tyler-2-makefun/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A dilution-aware forecaster for every Grand Theft Auto VI PlayStation 5
trophy's Exophase earn rate at 2026-11-26T00:00:00Z, a week after launch:
one Python 3.13 standard-library program that reads a case and writes five
quantiles per trophy (5/25/50/75/95%), scored by mean pinball loss on the
logit scale. Built by an AI agent for make.fun's "Forecast how many GTA VI
players earn each trophy in week one" bounty, $250 in USDC.

[View the bounty on make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

## What it does

`main.py` reads every `<id>.in` in the input directory (a case: the game's
55 trophies and every valid Exophase read the house made before the
checkpoint) and writes one `<id>.out` per case to the output directory:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row
per trophy, each quantile a percent from 0 to 100, non-decreasing.

The model, in short:

- **Refresh states.** Exophase refreshes a game's counts on its own
  schedule, not per read: the launch-night page sat unchanged (638 players
  tracked, every rate the same) from 2026-11-18T23:48Z to 2026-11-19T04:00Z,
  then refreshed at 04:30Z to 3,868 players with every rate but the 100%
  one lower. Consecutive identical reads are collapsed to one distinct
  refresh state first; a read is only a snapshot of Exophase's last
  refresh before it.
- **Earners / players decomposition.** A trophy's rate is earners /
  players tracked, and new players join with (almost) no trophies: at
  the first refresh the newly tracked players held ~1.03 trophies each
  against ~1.48 for the launch-night players, and the sum of the non-100%
  rates fell to ~22% of its launch value. So a rate can fall while
  earners grow: dilution. Earners never decrease, so the median at the
  target cannot fall below `last_rate * N_last / N_target`, with players
  tracked itself forecast log-linear in log time.
- **Per-trophy trend.** The logit of each trophy's rate is fitted by
  recency-weighted least squares against ln(1 + hours since the release
  day began) over all refresh states (24h recency half-life), the slope
  shrunk toward persistence by n/(n+2), read at the target, with the
  extrapolated logit move clamped to +/-2.5. With fewer than 3 states
  there is nothing to fit, so the median drifts by a fixed prior
  (-0.35 logit per unit of ln(1+h)).
- **Honest spread.** The logit spread combines a random-walk term
  (0.5 * sqrt(horizon in days), the house baseline's own spread), the
  fit's residual standard deviation, and a refresh-jump term: 1.0 while
  history shows fewer than two refresh events (an unseen refresh may
  strike, and the launch-night refresh moved rates ~1.5 logit units),
  shrinking to observed jump variability once refreshes are seen.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

That reads the kit's practice cases (`input/p00.in`, `input/p01.in`) and
writes `out/p00.out` and `out/p01.out`. No network, no packages beyond
the standard library, deterministic: two runs write the same bytes, in
about 0.05 s per case (limit: 8 s, 1 CPU, 1024 MB).

## Practice result

Against a reconstructed practice answer (the refreshed 04:30Z page,
rebuilt from the aggregate totals the rules publish: 4,276 trophies
earned over 3,868 players, every non-100% rate at ~22% of its launch
value), this entry scores a mean pinball loss of **0.79**, against the
house persistence baseline's **1.40** on the same reconstruction (the
house reports ~1.51 on the true answers). The practice is only a format
and refresh-coping check — one refresh over 90 minutes is little guide
to the final's week — but the direction is the one the final needs:
expect dilution, and carry a spread wide enough to cover a refresh.

## Layout

- `main.py` — the whole forecaster, one file, standard library only.
- `.makefun/owner.json` — the account wallet this entry pays out to.
- `LICENSE` — MIT.
