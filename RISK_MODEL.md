# Arrival risk between departure and arrival

To answer the instructor's question, we freeze the information available at
departure `t`. We predict an event at `t+15` or `t+30` and score it against
the recorded free-space count at exactly that arrival time. We also report
the subgroup that had free spaces at departure, where drivers can encounter
a carpark that filled up during the trip.

| Forecast | Observed event |
| --- | --- |
| Full at arrival | Exactly **0** free car spaces at arrival. |
| Low safety margin | **Fewer than K** free car spaces at arrival, default `K=3` (0–2 spaces). This includes full arrivals but is reported separately. |

The thresholds are in spaces, not occupancy percentages. For example, 90%
occupancy leaves about 2 free spaces in a 20-space carpark and about 60 in a
600-space carpark. The two forecasts use capacity and free-space information
available at departure. API snapshots record whether a carpark had free
spaces; they cannot identify an individual driver's parking outcome or
attribute a change to a particular driver.

## Reproduce the run

Use the same Python 3.11 environment and macOS OpenMP setup described in the
[README](README.md). From the repository root:

```sh
python -m pip install -r requirements-risk.txt
python forecast_risk.py \
  --history data/pilot_15min_history.csv \
  --holidays data/sg_public_holidays_2020_2027.csv \
  --low-spaces-threshold 3 \
  --output outputs/pilot_risk
python -m unittest discover -s tests -v
```

For the main experiment, agree on `K` and the eligible carparks before
looking at the test results. Pass the approved cohort using `--carparks` or
filter the input CSV. The bundled run uses all four pilot IDs, including
Albert Centre (ACB); it is a pipeline check for Part A, not the final
Chinatown cohort.

## Models and probability calibration

- **XGBoost and CatBoost:** fit separate binary classifiers for each event
  and horizon. Inputs include the free-space count, current capacity,
  recent occupancy lags and changes, time of day and calendar information.
- **Persistence:** assume the current full or low-space state persists;
  calibrate the two score levels using the calibration period.
- **Occupancy proxy:** fit XGBoost occupancy regression with the same
  features and training rows. Convert its estimate to expected free spaces
  using departure capacity. Map the score `1/(1+expected_free)` to each
  event probability using the same calibration period. This compares direct
  event forecasts with a capacity-aware use of occupancy point forecasts.

Fit the base models on the first 60% of observed decision times, fit Platt
calibration on the next 20%, and score on the final 20%. The 30-minute
embargo prevents labels from reaching the next split. The pilot uses 650
training and 202 calibration labels per horizon. Test labels never select
the calibrator, model or threshold. If a calibration set lacks enough examples
of either class, report the fallback and retain the classifier's raw scores;
leave the occupancy proxy probability blank when it cannot be calibrated.

Full is a subset of low-space risk. When the fitted probabilities cross,
raise the low-space estimate to the full-risk estimate so the primary full
forecast stays unchanged. Report the number of changes and evaluate the final
probabilities as exported, with raw classifier metrics for comparison.

Use Brier score and log loss (lower is better) to score probabilities, and
average precision to assess rare-event ranking. The held-out reliability
table reports the number of cases, mean probability and observed event
frequency in each decile. For example, among the 30-minute CatBoost forecasts
for open-at-departure sites, two full-risk predictions fell in the 10–20%
bin (mean 13.8%); neither filled up. Two observations cannot validate or
reject calibration around 10%. The calibration segment contained zero
open-to-full cases, so the fitted calibrators are pooled across departure
states; subgroup calibration needs more history.

## Pilot findings

Six observations at each horizon had free spaces at departure and no free
spaces on arrival: **6/181 (3.31%)** at 15 minutes and **6/177 (3.39%)** at
30 minutes. Among sites with at least 3 free spaces at departure, **1/173**
and **2/169** fell below that margin, respectively. These denominators are
carpark-time observations, not unique drivers.

| Model | Full-risk Brier, 15 min | Full-risk Brier, 30 min |
| --- | ---: | ---: |
| Calibrated persistence | 0.0331 | 0.0339 |
| Calibrated occupancy proxy | 0.0270 | 0.0359 |
| XGBoost direct risk | 0.0266 | 0.0286 |
| CatBoost direct risk | **0.0239** | **0.0283** |

The table uses only cases open at departure. CatBoost has the lowest full-risk
Brier score at both horizons in this pilot. Persistence has the lowest
low-space Brier score in the same subgroup (0.0152 and 0.0205). Neither
comparison establishes a winner for the full project.

For the separate recommendation check, rank only sites open at departure.
Every strategy sees the same candidates and uses only departure-time data to
rank them. Retain only decision times with complete observations and scores
for the compared candidates; use actual arrival counts to score the chosen
site. The most-current-spaces baseline already had zero first-choice full
arrivals at 56 and 55 matched decision times, respectively. Both direct risk
models also had zero, so this pilot does not establish an improvement in
recommendations. The pilot ranks by risk only. For the proposal's final test,
use the verified visitor eligibility and the same distance rule and weight
for both compared strategies, tune choices on a separate validation period,
and report paired full and low-space outcomes with uncertainty by day.

## Files for the group

| File | Contents |
| --- | --- |
| [risk_predictions.csv](examples/pilot_risk/risk_predictions.csv) | Test predictions. Join with Part B on `decision_time`, `carpark_id`, `horizon_min`. `p_full_*` is primary; `p_low_spaces_*` is separate. Columns starting `actual_` are evaluation-only. |
| [risk_metrics.csv](examples/pilot_risk/risk_metrics.csv) | Probability metrics for all valid rows, open-at-departure rows, and rows with at least K spaces at departure. |
| [risk_reliability.csv](examples/pilot_risk/risk_reliability.csv) | Test reliability bins with counts and observed rates. |
| [risk_calibration.csv](examples/pilot_risk/risk_calibration.csv) | Fit/calibration time boundaries, label counts and calibrator status. |
| [availability_transitions.csv](examples/pilot_risk/availability_transitions.csv) | Observed space decreases and transitions to full or low-space arrivals. |
| [risk_replay.csv](examples/pilot_risk/risk_replay.csv) | Matched ranking results, first-choice full rate and first-choice low-space rate. |

**可以直接向老师解释：**我们在出发时固定当时能看到的信息，分别预测 15 或
30 分钟后满位和低余位的概率。用独立时间段校准，再在更晚的真实快照上检查
预测概率是否与发生频率接近。我们单独评估出发时有位的情况，并让不同策略在
相同的候选和出发时间上比较到达结果，因此可以直接量化途中余位变化带来的风险。

References: [XGBoost binary logistic objective](https://xgboost.readthedocs.io/en/stable/parameter.html),
[CatBoost probability prediction](https://catboost.ai/docs/en/features/prediction),
and [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
