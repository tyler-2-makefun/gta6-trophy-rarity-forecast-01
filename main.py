"""gta6-trophy-rarity-forecast-01 entry: dilution-aware trophy-rate forecaster.

Forecasts, for every Grand Theft Auto VI PS5 trophy, five quantiles of its
Exophase earn rate at the case's target, from the house's recorded reads of
the game's trophy page. Pure Python 3.13 standard library, offline,
deterministic: the same case always writes the same bytes.

Model (written from scratch for this bounty; the kit's starter is only the
house persistence baseline, which this entry deliberately goes beyond):

  1. Refresh states. Exophase does NOT refresh a game's counts on every
     read: the house saw the launch-night page unchanged (638 players
     tracked, every rate the same) from 2026-11-18T23:48Z through
     2026-11-19T04:00Z, then refreshed at 04:30Z to 3,868 players with
     every rate but the 100% one lower. Consecutive identical reads carry
     no new information, so the history is first collapsed to distinct
     refresh states (first-seen time, players tracked, rates). A read is
     only a snapshot of Exophase's last refresh before it.

  2. Earners / players decomposition. A trophy's rate is
     earners / players_tracked. New players join with (almost) no trophies:
     at the first refresh the newly tracked players held ~1.03 trophies
     each against ~1.48 for the launch-night players, and the sum of the
     non-100% rates fell to ~22% of its launch value. So a rate can fall
     even while earners grow: dilution. Two structural consequences are
     enforced:
       - earners never decrease: the median rate at the target cannot fall
         below last_rate * N_last / N_target (the no-new-earners floor);
       - players tracked is itself forecast (log-linear in log time) and
         the floor uses that forecast.

  3. Per-trophy trend. For each trophy, the logit of its rate is fitted
     by recency-weighted least squares against ln(1 + hours since the
     release day began), the same clock the house curve baseline uses,
     but over all refresh states (not just the last 24h) with a 24h
     recency half-life, and the slope is shrunk toward persistence
     (0, i.e. "rates stay where they are") by n/(n+2) over n states.
     The fitted line is read at the target. The total extrapolated logit
     move from the last rate is clamped to +/-2.5 so a short, noisy
     history cannot extrapolate to absurdity. With fewer than 3 states
     there is nothing to fit: the median drifts by a fixed prior
     (-0.35 logit per unit of ln(1+h)), a small allowance for the
     dilution a refresh brings.

  4. Honest spread. The pinball loss on the logit scale punishes a
     spread that is too narrow (every trophy outside it) as much as one
     that is too wide. The logit spread s combines three independent
     sources:
       - a random-walk term, 0.5 * sqrt(horizon in days) (the house
         baseline's own spread);
       - the fit's residual standard deviation (or 0.35 when no fit);
       - a refresh-jump term: when history shows fewer than two refresh
         events, an unseen refresh may still strike before the target,
         and the launch-night refresh moved rates by ~1.5 logit units,
         so s_jump = 1.0; once refreshes are observed, the term shrinks
         to the observed jump variability over the expected number of
         refreshes in the horizon.
     Quantiles are inv_logit(median_logit + z_tau * s) at the standard
     normal's 5/25/50/75/95% points, rounded to 4 decimals half-up.

  5. Edge cases. A trophy at 100% in the last read (every tracked player
     holds it, e.g. the starter trophy) is forecast at 100% on every
     quantile. Rates are clamped to [0.01, 99.99] before any logit.
     Missing rates in a read are skipped for that trophy. A trophy with
     no usable rate anywhere falls back to a 1% median with a wide
     spread, so the program always writes a valid row.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only. It reads $MAKEFUN_INPUT_DIR when
set, else /input/, and writes to $MAKEFUN_OUTPUT_DIR when set, else
/output/: one <id>.out per <id>.in, {"trophies": [{"id": n, "q": [...]}]},
55 rows for the 55 trophies, each q five finite numbers 0..100,
non-decreasing, file at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import (none is shipped; this is one file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Model constants (fixed in advance, never tuned on the practice) ---
# Random-walk spread, logit units per square-root day (house baseline's).
SIGMA_RW = 0.5
# Recency half-life for the trend fit, in hours.
HALF_LIFE_H = 24.0
# Slope shrinkage: b *= n / (n + SHRINK_K) over n refresh states.
SHRINK_K = 2.0
# Largest logit move the extrapolated median may make from the last rate.
MAX_MOVE = 2.5
# Median drift prior (logit per unit ln(1+h)) when too few states to fit:
# a small allowance for the dilution a refresh brings.
DRIFT_PRIOR = -0.35
# Spread of the fit residual when no fit exists.
SPREAD_NO_FIT = 0.35
# Refresh-jump spread when history shows fewer than two refresh events:
# an unseen refresh may strike, and the launch-night refresh moved rates
# by ~1.5 logit units.
SPREAD_JUMP_UNSEEN = 1.0
# Prior mean interval between Exophase refreshes, in hours.
REFRESH_INTERVAL_PRIOR_H = 6.0
# Prior players-tracked growth when only one state exists (per day).
GROWTH_PRIOR_PER_DAY = 1.0


def clamp_rate(pct):
    """A rate in percent, clamped to the range the logit is defined on."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """An ISO 8601 timestamp (with Z or an offset) as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_start(case):
    """Midnight UTC of the release day, the zero of the model's clock."""
    try:
        day = date.fromisoformat(str(case.get("game", {}).get("release_date", "")))
        return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        first = case["history"][0]["at"]
        day = parse_time(first).date()
        return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def hours_since(start, moment):
    return (moment - start).total_seconds() / 3600.0


def distinct_states(case):
    """Collapse the history to distinct refresh states.

    Consecutive reads with the same players tracked and the same rates
    are one state: Exophase had not refreshed between them. Each state
    keeps its first-seen time, players tracked and rates.
    """
    states = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if states and states[-1]["key"] == key:
            continue
        states.append(
            {
                "key": key,
                "at": parse_time(read["at"]),
                "players": read["players_tracked"],
                "rates": read["rates"],
            }
        )
    return states


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw)
    return (a, b, resid)


def forecast_players(states, start, x_target, horizon_h):
    """Forecast players tracked at the target, log-linear in log time."""
    if len(states) < 2:
        last = states[-1]["players"]
        return last * (1.0 + GROWTH_PRIOR_PER_DAY * horizon_h / 24.0)
    t_last = states[-1]["at"]
    xs, ys, ws = [], [], []
    for s in states:
        h = hours_since(start, s["at"])
        xs.append(math.log1p(max(0.0, h)))
        ys.append(math.log(max(1.0, float(s["players"]))))
        ws.append(0.5 ** ((t_last - s["at"]).total_seconds() / 3600.0 / HALF_LIFE_H))
    fit = weighted_line(xs, ys, ws)
    if fit is None:
        return float(states[-1]["players"])
    a, b, _ = fit
    n_target = math.exp(a + b * x_target)
    last = float(states[-1]["players"])
    return min(max(n_target, last), last * 200.0)


def refresh_events(states):
    """Consecutive state pairs: each is one Exophase refresh.

    Returns (common shifts, intervals in hours): the common shift is the
    median over trophies of the logit move across the refresh.
    """
    shifts, gaps = [], []
    for prev, cur in zip(states, states[1:]):
        moves = []
        for r0, r1 in zip(prev["rates"], cur["rates"]):
            if r0 is None or r1 is None:
                continue
            moves.append(logit(r1) - logit(r0))
        if moves:
            moves.sort()
            shifts.append(moves[len(moves) // 2])
        gaps.append((cur["at"] - prev["at"]).total_seconds() / 3600.0)
    return shifts, gaps


def forecast_trophy(idx, case, states, start, x_target, x_last, n_target, horizon_h):
    """Five quantiles for one trophy (index idx into the case's trophies)."""
    last_rate = None
    for s in reversed(states):
        if s["rates"][idx] is not None:
            last_rate = s["rates"][idx]
            break
    if last_rate is None:
        # No usable rate anywhere: a flat 1% median, wide spread.
        return ("fixed", [round4(inv_logit(logit(1.0) + z * 1.5)) for z in Z])
    if last_rate >= 99.995:
        # Every tracked player holds it (the starter trophy): stays 100%.
        return ("fixed", [100, 100, 100, 100, 100])

    t_last = states[-1]["at"]
    xs, ys, ws = [], [], []
    for s in states:
        r = s["rates"][idx]
        if r is None:
            continue
        h = hours_since(start, s["at"])
        xs.append(math.log1p(max(0.0, h)))
        ys.append(logit(r))
        ws.append(0.5 ** ((t_last - s["at"]).total_seconds() / 3600.0 / HALF_LIFE_H))

    base = logit(last_rate)
    if len(xs) >= 3:
        fit = weighted_line(xs, ys, ws)
        if fit is not None:
            a, b, resid = fit
            n = len(xs)
            b *= n / (n + SHRINK_K)
            median = a + b * x_target
            spread_fit = min(max(resid, 0.15), 1.5)
        else:
            median = base + DRIFT_PRIOR * (x_target - x_last)
            spread_fit = SPREAD_NO_FIT
    else:
        median = base + DRIFT_PRIOR * (x_target - x_last)
        spread_fit = SPREAD_NO_FIT

    # The extrapolated median may not move further than MAX_MOVE logit
    # units from the last rate.
    median = min(max(median, base - MAX_MOVE), base + MAX_MOVE)
    # Dilution floor: earners never decrease, so the rate at the target
    # cannot fall below last_rate * N_last / N_target.
    n_last = float(states[-1]["players"])
    floor = logit(last_rate * n_last / n_target)
    median = max(median, floor)
    median = min(median, logit(99.99))

    # Spread: random walk + fit residual + refresh-jump uncertainty.
    d_days = horizon_h / 24.0
    spread_rw = SIGMA_RW * math.sqrt(max(0.0, d_days))
    return ("model", median, spread_fit, spread_rw)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    if not case["trophies"]:
        raise ValueError(f"{name} has no trophies")

    start = release_start(case)
    target = parse_time(case["target"])
    at = parse_time(case["at"])
    horizon_h = (target - at).total_seconds() / 3600.0
    x_target = math.log1p(max(0.0, hours_since(start, target)))

    states = distinct_states(case)
    n_target = forecast_players(states, start, x_target, horizon_h)
    shifts, gaps = refresh_events(states)
    x_last = math.log1p(max(0.0, hours_since(start, states[-1]["at"])))

    # Refresh-jump spread: an unseen refresh is the big unknown when
    # history shows fewer than two refresh events; once refreshes are
    # observed, the term follows their observed variability over the
    # expected number of refreshes in the horizon.
    if len(shifts) < 2:
        spread_jump = SPREAD_JUMP_UNSEEN
    else:
        mean_shift = sum(shifts) / len(shifts)
        var = sum((c - mean_shift) ** 2 for c in shifts) / len(shifts)
        std_c = math.sqrt(var)
        gaps_sorted = sorted(g for g in gaps if g > 0)
        interval = gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else REFRESH_INTERVAL_PRIOR_H
        interval = max(interval, 1.0)
        n_ref = min(horizon_h / interval, 40.0)
        spread_jump = min(0.8, 0.3 * math.sqrt(max(0.0, n_ref)) * max(std_c, 0.3))

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        kind, *vals = forecast_trophy(i, case, states, start, x_target, x_last, n_target, horizon_h)
        if kind == "fixed":
            rows.append({"id": trophy["id"], "q": vals[0]})
            continue
        median, spread_fit, spread_rw = vals
        spread = math.sqrt(spread_fit ** 2 + spread_rw ** 2 + spread_jump ** 2)
        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit(median + z * spread))
            q = min(max(q, 0.01), 100.0)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
