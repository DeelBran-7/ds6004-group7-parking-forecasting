# DS6004 Group 7 — Parking occupancy forecasting

**Part A: Forecasting.** This module turns 15-minute HDB parking snapshots into
15- and 30-minute occupancy forecasts. Its CSV output is ready for the
recommendation and evaluation modules to read.

这是截图里“同学 A”的预测模块。快速查看结果：
[试采预测](examples/pilot/predictions.csv) ·
[预测诊断](examples/pilot/forecast_diagnostics.csv) ·
[数据审计](examples/pilot/audit.json)。

**Teacher question 2:** the [arrival-risk extension](RISK_MODEL.md) directly
forecasts the probability of zero free spaces and the probability of fewer
than three free spaces. It includes XGBoost, CatBoost, two comparators,
chronological probability calibration, and matched replay on the pilot.

## Run it

From the repository root, with Python 3.11:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python forecast.py \
  --history data/pilot_15min_history.csv \
  --holidays data/sg_public_holidays_2020_2027.csv \
  --output outputs/pilot
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell and run
the `python forecast.py` command on one line. On macOS, if XGBoost cannot
load `libomp.dylib`, install the OpenMP runtime with `brew install libomp`.

Pass `--carparks KAM KAS SLS` to rerun on those three pilot IDs. This is a
provisional subset; use the agreed visitor-eligible cohort for the final study.
Pass `--max-lag-seconds` to change the default 900-second freshness limit.

## Method

Occupancy is `1 - lots_available / total_lots` on the 0–1 scale. For each
carpark and decision time `t`, a label is a valid observed snapshot for the
same carpark at **exactly** `t+h`. Missing labels stay blank. Features include
current occupancy and capacity, observation age, exact 15/30/45/60-minute
lags, and occupancy mean, standard deviation and observation count over
`(t−60 min, t]`. Time of day, weekday and weekend features refer to the
known arrival time `t+h`.

| Model | Definition |
| --- | --- |
| B0 | Carry forward observed occupancy at `t`. |
| B1 | Observed occupancy at `t+h−7 days` for the same carpark. Missing when that snapshot is unavailable. |
| M0 | XGBoost using the occupancy, capacity and calendar features above, plus carpark indicators fitted on the training set. |
| M1 | Same training rows and settings as M0; adds the public-holiday flag for the arrival date. |

We fit independent models for the two horizons. M0 and M1 use fixed settings
(200 rounds, depth 4, learning rate 0.05, seed 42). Both use the same
chronological 60/20/20 split on distinct observed decision times. Origins in
the 30 minutes immediately before each split boundary are excluded, so no
training or validation label reaches the next split. Forecast features use
only observed data at or before `t` and calendar information known in advance.

The history input must have the columns `timestamp`, `carpark_id`,
`total_lots`, `lots_available`, and `lag_seconds`. It should already be on a
15-minute grid and contain car lots only; if `lot_type` is present, only `C`
rows are loaded. Naive timestamps are Singapore local time; explicit offsets
are converted to Singapore time. Duplicate keys and off-grid timestamps are
rejected. Stale readings, negative ages and invalid lot counts are flagged;
their occupancy is unavailable as a feature or label.

The holidays CSV needs a `date` column and a complete official holiday list
for each included year, with observed holidays as listed by MOM. Dates in
uncovered years remain unknown. Use calendar-covered years for both models
in the formal holiday ablation.

## Handoff to Parts B and C

`outputs/pilot/predictions.csv` has one row per observed decision origin and
horizon in the validation or test split. Join on **`decision_time`,
`carpark_id`, `horizon_min`**. All timestamps are Singapore time.

| Columns | How to use them |
| --- | --- |
| `split`, `target_time` | Use `validation` to set the ranking configuration; use `test` for final replay. `target_time` is the observed arrival time to check. |
| `current_is_valid`, `current_occupancy`, `current_lots_available`, `total_lots` | Use current data only when `current_is_valid` is true. The raw lot and capacity columns can be populated on invalid rows. `total_lots` is capacity at decision time. |
| `B0`, `B1`, `M0`, `M1` | Forecast occupancy ratios. Blank is unavailable. Select the model using validation data; do not choose it from test performance. |
| `actual_is_observed`, `actual_occupancy`, `actual_lots_available`, `actual_total_lots` | Outcomes for scoring only. Full at arrival means `actual_lots_available == 0`. Near-full means strictly less than 10% of arrival capacity is free. |

For the C0/C comparison, join the separate distance and visitor eligibility
data, then use the same eligible candidates and weights for both rules.
Require complete, observed arrival outcomes for the comparison. Do not
silently drop a candidate only from one strategy when a forecast is missing.
`forecast_diagnostics.csv` reports test denominators, MAE, RMSE and near-full
precision/recall; compare models on the same scored rows if coverage differs.
`audit.json` records input and split coverage. The CLI also saves four fitted
XGBoost models; `carpark_indicator_ids` in each model records the mapping for
`park_0`, `park_1`, etc.

## What the pilot establishes

The supplied CSV has 1,122 rows across ACB, KAM, KAS and SLS from 31 October
2026 at 00:15 to 2 November at 23:45. Relative to a full grid for these four
IDs, 26 carpark-time observations are missing. The test split has 224 and
220 usable arrival labels at 15 and 30 minutes, respectively.

| Horizon | B0 MAE | M0 MAE | M1 MAE | B1 scored rows |
| --- | ---: | ---: | ---: | ---: |
| 15 min | 2.82 pp | 3.40 pp | 3.40 pp | 0 |
| 30 min | 4.41 pp | 4.85 pp | 4.85 pp | 0 |

MAE is expressed here in percentage points of occupancy. B0 has the lowest
overall MAE in this run. The pilot has no holiday dates and no week-ago
matches, so it cannot measure the holiday feature's benefit or B1's accuracy.
ACB is Albert Centre; the four-ID run is a pipeline check, not the final
Chinatown cohort. The proposal targets roughly 8–12 eligible HDB carparks
within 1 km of a fixed Chinatown MRT destination. Final experiments need
the verified cohort and a longer history covering multiple holidays and weeks.

## Data sources

- `data/pilot_15min_history.csv`: Group 7 pilot extract supplied for this module.
  Original source: [HDB Carpark Availability](https://data.gov.sg/datasets/d_ca933a644e55d34fe21f28b8052fac63/view).
- `data/sg_public_holidays_2020_2027.csv`: [MOM Singapore Public Holidays](https://data.gov.sg/datasets/d_8ef23381f9417e4d4254ee8b4dcdb176/view).
- Use [HDB Carpark Information](https://data.gov.sg/datasets/d_23f946fa557947f93a8043bbef41dd09/view)
  for the separate distance and visitor-eligibility checks.
