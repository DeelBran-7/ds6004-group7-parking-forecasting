# Part B Chinatown parking recommendation and historical replay

This module runs the proposal's four recommendation strategies on identical
historical cases. It also shows how a calibrated full-at-arrival probability
can replace predicted occupancy in the same distance-aware ranking.

## Pilot destination and candidate set

The fixed destination is **Chinatown MRT Exit A**. The LTA [MRT exits
dataset](https://data.gov.sg/datasets/d_b39d3a0871985372d7e1637193335da5/view)
gives the exit location. The [destination file](data/chinatown_destination.json)
records the source feature and conversion to SVY21. The
[HDB carpark dataset](https://data.gov.sg/datasets/d_23f946fa557947f93a8043bbef41dd09/view)
supplies the carpark reference points and short-term hours.

| ID | Parking facility | Straight-line distance to Exit A | Included in the 1 km pilot? |
| --- | --- | ---: | --- |
| KAM | Kreta Ayer multistorey car park, Smith Street | 202 m | Yes |
| SLS | Sago Lane car park | 243 m | Yes |
| KAS | Kreta Ayer surface car park | 366 m | Yes |
| ACB | Albert Centre basement car park | 2,236 m | No |

KAM, KAS and SLS list whole-day short-term parking and night parking. These
are facility hours; the pilot does not verify individual reserved bays or
historic temporary closures. Distances are straight-line proxies for walking
distance. The pilot only screens IDs available in the supplied CSV, not all
HDB carparks within 1 km. We refit Part A on these three pilot carparks before
running Part B.

## How the replay works

| Strategy | Rule |
| --- | --- |
| A | Nearest visitor-eligible carpark. |
| B | Most car spaces available at departure. |
| C0 | Current occupancy and distance. |
| C | M1 forecast occupancy at arrival and distance. |
| C_XGB_RISK / C_CAT_RISK | XGBoost / CatBoost calibrated probability of zero spaces at arrival and distance. |

The weighted strategies minimize `w × distance / 1,000 m + (1−w) × signal`.
Occupancy and risk are on the 0–1 scale, and ties go to the nearer carpark
then ID. This setup makes **C vs C0** a direct test of replacing current
occupancy with forecast occupancy. The extra risk comparisons also change
the predicted quantity, from occupancy to probability of zero spaces.

Use the fixed weight grid `{0.25, 0.50, 0.75}`. Select one shared weight by
minimizing C0 first-choice full arrivals on the validation period, preferring
0.50 on ties. Apply that weight to every weighted strategy and both horizons.
The test results include a separate sensitivity table for the full grid.
The risk models do not select the weight using test outcomes.

At each departure and horizon, first construct the expected candidate set
using distance and the visitor schedule at arrival. Keep the historical
case only when every expected carpark has a valid departure snapshot,
forecast and observed arrival snapshot. Every compared strategy ranks the
same candidates using only departure-time information. Then score the first
choice and top three against the actual arrival records. A secondary run
keeps only candidates with space at departure, to focus on changes during
the trip. If optional risk forecasts are present, all test strategies use
their shared complete cases; omit `--risk` for the core comparison if risk
coverage is incomplete. Core results and denominators match the extended
run in this pilot.

Report full at arrival (zero spaces), near-full (less than 10% free), and low
safety margin (fewer than three spaces) separately. Also report top-three
feasibility and the mean extra straight-line distance beyond the nearest
eligible carpark. Arrival snapshot availability is a proxy for what a driver
might encounter, not a recorded outcome for an individual driver.

## Pilot results

The validation period has 51 complete cases per horizon. All grid weights
tie for C0 full arrivals, so the shared weight is **0.50**. The test has
**56 complete 15-minute cases** and **55 complete 30-minute cases**, out
of 57 observed departure times per horizon.

| Strategy | First choice full, 15 min | First choice full, 30 min | Mean extra distance |
| --- | ---: | ---: | ---: |
| A | 0/56 | 0/55 | 0 m |
| B | 0/56 | 0/55 | 0 m |
| C0 | 0/56 | 0/55 | 0 m |
| C | 0/56 | 0/55 | 0 m |
| C_XGB_RISK | 0/56 | 0/55 | 0 m |
| C_CAT_RISK | 0/56 | 0/55 | 0 m |

Every first choice was KAM. At the selected weight, the C–C0 difference is
0 percentage points for full arrivals and 0 m for distance. All test cases
had at least one available top-three option; with at most three candidates,
this metric offers little separation. The test covers one date, so we do not
report a day-block confidence interval. More eligible carparks and more test
days are needed to assess whether forecasts change recommendations and reduce
full arrivals. The result establishes that Part B runs, with no measured
benefit over the baselines in this pilot.

## Reproduce and inspect

From the repository root with the environment in the [README](README.md):

```sh
python -m pip install -r requirements-risk.txt
python forecast.py --history data/pilot_15min_history.csv \
  --holidays data/sg_public_holidays_2020_2027.csv --carparks KAM KAS SLS \
  --output outputs/chinatown_forecast
python forecast_risk.py --history data/pilot_15min_history.csv \
  --holidays data/sg_public_holidays_2020_2027.csv --carparks KAM KAS SLS \
  --output outputs/chinatown_risk
python recommendation.py --forecasts outputs/chinatown_forecast/predictions.csv \
  --risk outputs/chinatown_risk/risk_predictions.csv \
  --metadata data/chinatown_pilot_carparks.csv --destination data/chinatown_destination.json \
  --output outputs/chinatown_recommendation
python -m unittest discover -s tests -v
```

- [Test summary](examples/chinatown_recommendation/recommendation_summary.csv):
  one row per horizon, cohort and strategy at the selected weight.
- [Paired differences](examples/chinatown_recommendation/paired_differences.csv):
  full arrivals avoided or introduced and distance changes versus C0.
- [Ranked decisions](examples/chinatown_recommendation/recommendation_rankings.csv):
  top three per case, including the arrival observations used for evaluation.
- [Case audit](examples/chinatown_recommendation/replay_case_audit.csv):
  expected candidate counts and reasons for excluding incomplete cases.
- [Validation weights](examples/chinatown_recommendation/validation_weights.csv)
  and [test sensitivity](examples/chinatown_recommendation/weight_sensitivity.csv):
  the shared distance-weight experiment.
