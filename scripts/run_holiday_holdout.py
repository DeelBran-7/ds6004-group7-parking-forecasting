"""Retrospective RQ1 supplement with a chronological, full-year test window.

Train and validation both precede the test. Training contains both holiday and
non-holiday examples. Only M0/M1 occupancy regressors are fitted; no risk or
recommendation models are retrained. Existing main-study outputs are untouched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import forecast
from evaluation_stats import holm_adjust, paired_day_inference

VALIDATION_START = pd.Timestamp("2025-09-05")
TEST_START = pd.Timestamp("2025-11-05")
SOURCE_START = pd.Timestamp("2024-11-05")
SOURCE_END = pd.Timestamp("2026-11-04 23:45")
MODELS = ("B0", "B1", "M0", "M1")
REPETITIONS = 10_000
SEED = 20_261_105


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holiday_calendar(path):
    """One row per calendar holiday date, with episodes fixed by consecutive days."""
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    names = data.groupby("date")["holiday"].agg(lambda x: " / ".join(sorted(set(x))))
    result = names.reset_index().sort_values("date").reset_index(drop=True)
    start = result["date"].diff().ne(pd.Timedelta(days=1))
    result["episode_id"] = result["date"].where(start).ffill().dt.strftime("%Y-%m-%d")
    return result


def add_target_calendar(frame, calendar):
    """Holiday membership and resampling keys use arrival/target dates, not origins."""
    result = frame.copy()
    result["target_date"] = pd.to_datetime(result["target_time"]).dt.normalize()
    result = result.merge(calendar.rename(columns={"date": "target_date"}),
                          on="target_date", how="left", validate="many_to_one")
    result["target_is_holiday"] = result["holiday"].notna()
    if not result["is_holiday"].eq(result["target_is_holiday"].astype(int)).all():
        raise ValueError("Prediction holiday flags disagree with the target-date calendar")
    return result


def validate_audit(audit):
    """Reject cached forecasts from the old standard or shared-boundary experiments."""
    if pd.Timestamp(audit["first_time"]) != SOURCE_START or pd.Timestamp(audit["last_time"]) != SOURCE_END:
        raise ValueError("Unexpected source date coverage")
    expected = {
        "train": (SOURCE_START, VALIDATION_START - pd.Timedelta(minutes=45)),
        "validation": (VALIDATION_START, TEST_START - pd.Timedelta(minutes=45)),
        "test": (TEST_START, SOURCE_END),
    }
    for horizon in ("15", "30"):
        parts = audit["horizons"][horizon]
        for split, (first, last) in expected.items():
            part = parts[split]
            if pd.Timestamp(part["first_decision_time"]) != first or pd.Timestamp(part["last_decision_time"]) != last:
                raise ValueError(f"Incorrect {horizon}-minute {split} boundaries")
            if part["unknown_holiday_labels"]:
                raise ValueError("Unknown calendar coverage")
        if parts["train"]["holiday_days"] < 1:
            raise ValueError("Training must contain holidays; non-holiday-only training cannot test RQ1")


def common_scored(frame):
    test = frame.loc[frame["split"].eq("test") & frame["forecast_label_eligible"].eq(True)].copy()
    finite = np.isfinite(test[["actual_occupancy", *MODELS]].to_numpy(dtype=float)).all(axis=1)
    return test.loc[finite].copy()


def training_feature_counts(history_path, holidays_path):
    """Check the actual fitting rows, not just the calendar's nominal dates."""
    history, audit = forecast.load_history(history_path)
    if pd.Timestamp(audit["first_time"]) != SOURCE_START or pd.Timestamp(audit["last_time"]) != SOURCE_END:
        raise ValueError("Unexpected source date coverage")
    calendar = forecast.load_holidays(holidays_path)
    counts = {}
    for horizon in (15, 30):
        examples = forecast.make_examples(history, calendar, horizon, (VALIDATION_START, TEST_START))
        rows = examples.loc[examples["split"].eq("train") & examples["forecast_label_eligible"]]
        flags = rows["is_holiday"].value_counts().to_dict()
        if set(flags) != {0.0, 1.0}:
            raise ValueError("Actual training feature must contain both holiday=0 and holiday=1")
        counts[str(horizon)] = {"non_holiday": int(flags[0.0]), "holiday": int(flags[1.0])}
    return counts


def paired_rows(frame, horizon, cohort, block_column):
    rows = []
    truth = frame["actual_occupancy"].to_numpy()
    for candidate, reference in (("M1", "M0"), ("M0", "B0")):
        candidate_loss = np.abs(frame[candidate].to_numpy() - truth)
        reference_loss = np.abs(frame[reference].to_numpy() - truth)
        result = paired_day_inference(frame[block_column].to_numpy(), candidate_loss, reference_loss,
                                      repetitions=REPETITIONS, seed=SEED)
        # The shared helper calls its block count n_days; here the key is explicit.
        result["n_blocks"] = result.pop("n_days")
        rows.append(dict(horizon_min=horizon, cohort=cohort, comparison=f"{candidate}-{reference}",
                         candidate=candidate, reference=reference, metric="mae", block_unit=block_column,
                         n_target_dates=int(frame["target_date"].nunique()),
                         relative_reduction=(1 - result["candidate_mean"] / result["reference_mean"]
                                             if result["reference_mean"] else None), **result))
    return rows


def write_comparisons(rows, path):
    frame = pd.DataFrame(rows)
    frame["p_value_holm"] = holm_adjust([None if pd.isna(v) else v for v in frame["p_value"]])
    frame.to_csv(path, index=False, float_format="%.10g")
    return frame


def summarize(predictions, calendar, output):
    data = add_target_calendar(predictions, calendar)
    metrics, daily, day_comparisons, episode_comparisons = [], [], [], []
    for horizon, part in data.groupby("horizon_min", sort=True):
        eligible = part.loc[part["split"].eq("test") & part["forecast_label_eligible"].eq(True)]
        scored = common_scored(part)
        for cohort in ("all", "holiday", "non_holiday"):
            group = scored if cohort == "all" else scored.loc[scored["target_is_holiday"].eq(cohort == "holiday")]
            eligible_group = eligible if cohort == "all" else eligible.loc[eligible["target_is_holiday"].eq(cohort == "holiday")]
            for model in MODELS:
                error = group[model] - group["actual_occupancy"]
                metrics.append(dict(horizon_min=horizon, cohort=cohort, model=model,
                                    n_eligible=len(eligible_group), n_scored=len(group),
                                    n_target_dates=group["target_date"].nunique(),
                                    mae=error.abs().mean(), rmse=np.sqrt(error.pow(2).mean())))
            day_comparisons.extend(paired_rows(group, horizon, cohort, "target_date"))
            if cohort == "holiday":
                episode_comparisons.extend(paired_rows(group, horizon, cohort, "episode_id"))
                for date, day in group.groupby("target_date", sort=True):
                    m0 = (day["M0"] - day["actual_occupancy"]).abs().mean()
                    m1 = (day["M1"] - day["actual_occupancy"]).abs().mean()
                    daily.append(dict(horizon_min=horizon, target_date=date.strftime("%Y-%m-%d"),
                                      holiday=day["holiday"].iat[0], episode_id=day["episode_id"].iat[0],
                                      n_scored=len(day), n_carparks=day["carpark_id"].nunique(),
                                      M0_mae=m0, M1_mae=m1, delta_mae=m1-m0,
                                      relative_reduction=1-m1/m0 if m0 else None))
    metrics = pd.DataFrame(metrics)
    daily = pd.DataFrame(daily)
    metrics.to_csv(output / "forecast_metrics.csv", index=False, float_format="%.10g")
    daily.to_csv(output / "holiday_daily_metrics.csv", index=False, float_format="%.10g")
    day_results = write_comparisons(day_comparisons, output / "paired_holiday_comparisons.csv")
    episode_results = write_comparisons(episode_comparisons, output / "holiday_episode_comparisons.csv")
    return metrics, daily, day_results, episode_results


def draw_figures(metrics, daily, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#21394b", "text.color": "#21394b",
                         "axes.edgecolor": "#a9bac6", "xtick.color": "#526b80", "ytick.color": "#526b80"})
    blue, gray = "#0077a5", "#74899c"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
    dates = sorted(daily["target_date"].unique())
    n_episodes = daily["episode_id"].nunique()
    positions = np.arange(len(dates))
    for ax, horizon in zip(axes, (15, 30)):
        values = daily.loc[daily.horizon_min.eq(horizon)].set_index("target_date").reindex(dates)
        delta = 100 * values["delta_mae"]
        ax.barh(positions, delta, color=[blue if x < 0 else "#bc572b" for x in delta], height=.62)
        ax.axvline(0, color="#526b80", lw=.8)
        ax.set_yticks(positions, dates)
        ax.set_xlabel("M1 minus M0 MAE (percentage points)")
        n_holiday_dates = daily.loc[daily.horizon_min.eq(horizon), "target_date"].nunique()
        ax.set_title(f"{horizon} minutes | {n_holiday_dates} holiday dates", loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=.17); ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Holiday feature: effect on each held-out holiday date", x=.02, y=.98, ha="left", fontsize=16, fontweight="bold")
    fig.text(.02,.92,"Blue / negative: M1 is better. Orange / positive: M0 is better. Same model settings and common samples.",fontsize=10)
    fig.text(.02,.035,"Retrospective supplement | Test: 5 Nov 2025 to 4 Nov 2026 | Training contains holidays and non-holidays.",fontsize=9)
    fig.text(.02,.012,f"The {len(dates)} scored holiday dates form {n_episodes} calendar episodes. Bars are point estimates; paired uncertainty is reported separately.",fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.89]); fig.savefig(output / "holiday_daily_effect.png",dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.7,5.5), sharey=True)
    groups = ["all", "holiday", "non_holiday"]
    labels = ["All test dates", "Holiday dates", "Non-holiday dates"]
    for ax,horizon in zip(axes,(15,30)):
        part=metrics.loc[metrics.horizon_min.eq(horizon)]
        y=np.arange(3)
        for offset,model,color in ((-.19,"M0",gray),(.19,"M1",blue)):
            value=part.loc[part.model.eq(model)].set_index("cohort").reindex(groups)
            bars=ax.barh(y+offset,100*value.mae,height=.34,label=model,color=color)
            ax.bar_label(bars,fmt="%.3f",padding=4,fontsize=9)
        ax.set_yticks(y,labels); ax.set_xlim(0,3.5); ax.set_xlabel("MAE (percentage points)")
        ax.set_title(f"{horizon} minutes",loc="left",fontweight="bold"); ax.grid(axis="x",alpha=.17); ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Does the public-holiday flag improve occupancy forecasts?",x=.02,y=.98,ha="left",fontsize=16,fontweight="bold")
    fig.text(.02,.92,"M0: XGBoost without the flag | M1: identical model plus the flag | Lower MAE is better.",fontsize=10)
    axes[1].legend(loc="lower right")
    n_test_dates = int(metrics.loc[metrics.cohort.eq("all"), "n_target_dates"].max())
    fig.text(.02,.04,f"Chronological test: {n_test_dates} scored dates, including {len(dates)} holiday dates ({n_episodes} consecutive-date episodes).",fontsize=9)
    fig.text(.02,.015,"No test-based tuning. Retrospective sensitivity analysis, not a new independent confirmation.",fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.88]); fig.savefig(output / "holiday_cohort_mae.png",dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history",type=Path,required=True)
    parser.add_argument("--holidays",type=Path,default=ROOT/"data/sg_public_holidays_2020_2027.csv")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/rq_supplement/holiday")
    parser.add_argument("--reuse-predictions",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    contract=dict(source_filename=args.history.name,source_sha256=file_sha256(args.history),
                  calendar_filename=args.holidays.name,calendar_sha256=file_sha256(args.holidays),
                  forecasting_code_sha256=file_sha256(ROOT/"forecast.py"),
                  validation_start=VALIDATION_START.isoformat(),test_start=TEST_START.isoformat(),
                  expected_source_start=SOURCE_START.isoformat(),expected_source_end=SOURCE_END.isoformat(),
                  embargo_minutes=30,model_pair="M0/M1: fixed XGBoost parameters; M1 adds only is_holiday")
    manifest_path=output/"experiment_manifest.json"
    if args.reuse_predictions:
        manifest=json.loads(manifest_path.read_text())
        if manifest["contract"]!=contract:
            raise ValueError("Cached experiment contract differs from requested data/code/boundaries")
        training_counts=manifest["training_feature_counts"]
        if any(min(training_counts[h].values())<1 for h in ("15","30")):
            raise ValueError("Cached training feature lacks both binary values")
        audit=json.loads((output/"forecast/audit.json").read_text())
    else:
        training_counts=training_feature_counts(args.history,args.holidays)
        audit=forecast.run(args.history,args.holidays,output/"forecast",
                           split_boundaries=(VALIDATION_START,TEST_START))
    validate_audit(audit)
    calendar=holiday_calendar(args.holidays)
    predictions=pd.read_csv(output/"forecast/predictions.csv",parse_dates=["decision_time","target_time"])
    metrics,daily,days,episodes=summarize(predictions,calendar,output)
    period_rows=[]
    for name,start,end in (("train",SOURCE_START,VALIDATION_START),("validation",VALIDATION_START,TEST_START),
                           ("test",TEST_START,SOURCE_END.normalize()+pd.Timedelta(days=1))):
        dates=calendar.loc[calendar.date.ge(start)&calendar.date.lt(end)]
        period_rows.append(dict(period=name,first_date=start.strftime("%Y-%m-%d"),
                                last_date=(end-pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                calendar_dates=(end-start).days,holiday_dates=len(dates),
                                holiday_episodes=dates.episode_id.nunique()))
    pd.DataFrame(period_rows).to_csv(output/"holiday_period_coverage.csv",index=False)
    manifest=dict(contract=contract,training_feature_counts=training_counts,repetitions=REPETITIONS,seed=SEED,
                  interpretation="Retrospective supplement after inspecting the existing study; not independent confirmation",
                  training="Both holiday and non-holiday rows; four regression fits only; no additional tuning",
                  validation="Reserved chronological segment; model parameters remain fixed, not tuned on this segment or test",
                  support="Same test rows finite for actual occupancy and all B0/B1/M0/M1, with same-segment forecast labels",
                  grouping="Holiday status and resampling dates use target_time in Asia/Singapore",
                  estimand="Observation-weighted mean absolute loss difference; negative M1-M0 favors the holiday feature",
                  primary_inference="Target-date block bootstrap (10000), two-sided block sign flips, Holm within paired_holiday_comparisons.csv",
                  sensitivity="Consecutive calendar holiday dates combined into episodes; Holm separately within holiday_episode_comparisons.csv",
                  uncertainty="95% percentile intervals are not multiplicity-adjusted. Dates/episodes, not carpark-time rows, are resampling units",
                  limitations=[f"Only {daily.target_date.nunique()} scored test holiday dates and {daily.episode_id.nunique()} consecutive-date episodes",
                               "Episode grouping reduces adjacent-holiday dependence but does not model other cross-episode dependence",
                               "Resampling assumes exchangeable independent blocks; sign flips additionally require symmetric block effects",
                               "No significant difference does not establish equivalence or absence of a holiday effect",
                               "The experiment tests the incremental value of this binary flag, not every possible calendar representation"])
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    draw_figures(metrics,daily,output)
    print(pd.DataFrame(period_rows).to_string(index=False))
    print(days.loc[days.cohort.eq("holiday")].to_string(index=False))
    print(episodes.to_string(index=False))
    print(f"Saved holiday supplement to {output}")


if __name__=="__main__":
    main()
