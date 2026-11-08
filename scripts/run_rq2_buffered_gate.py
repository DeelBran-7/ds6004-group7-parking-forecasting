"""Retrospective RQ2 extension with historical empirical spare-space buffers.

Pre-fixed two-part validation: estimate signal-specific residual quantiles in the
first part, choose a shared (q, k) using CURRENT_BUFFERED_GATE in the second.
Adjacent residuals are dependent: these are empirical buffers, not conformal or
safety guarantees. No test outcomes or forecast-gate outcomes choose parameters.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import recommendation as rec
from evaluate import ranked_choices, read_inputs
from scripts import run_rq2_gate_experiment as base

GATES = ("CURRENT_BUFFERED_GATE", "FORECAST_BUFFERED_GATE")
QGRID = (.90, .95, .975, .99)
KGRID = (3, 5, 10, 20)
CALIBRATION_START = pd.Timestamp("2026-01-17")
SELECTION_START = pd.Timestamp("2026-04-01")
TEST_START = pd.Timestamp("2026-06-12")
EMBARGO = pd.Timedelta(minutes=30)
FALLBACK = (.95, 3)
TO_BASE = dict(zip(GATES, base.GATES))
FROM_BASE = dict(zip(base.GATES, GATES))


def calibration_rows(predictions):
    """Use the same valid, same-segment endpoints for both residual signals."""
    mask = (predictions["split"].eq("validation")
            & predictions["decision_time"].ge(CALIBRATION_START)
            & predictions["decision_time"].lt(SELECTION_START - EMBARGO)
            & predictions["target_time"].lt(SELECTION_START)
            & predictions["forecast_label_eligible"].eq(True)
            & predictions["current_is_valid"].eq(True)
            & predictions["actual_is_observed"].eq(True)
            & predictions["M1"].between(0, 1)
            & np.isfinite(predictions["total_lots"]) & predictions["total_lots"].gt(0)
            & np.isfinite(predictions["current_lots_available"])
            & predictions["current_lots_available"].between(0, predictions["total_lots"])
            & np.isfinite(predictions["actual_total_lots"]) & predictions["actual_total_lots"].gt(0)
            & np.isfinite(predictions["actual_lots_available"])
            & predictions["actual_lots_available"].between(0, predictions["actual_total_lots"]))
    rows = predictions.loc[mask].copy()
    if rows.empty or rows["target_time"].max() >= SELECTION_START:
        raise ValueError("No valid, strictly earlier calibration data")
    return rows


def estimate_margins(rows, required_carparks):
    expected = {(h, str(p)) for h in (15, 30) for p in required_carparks}
    present = {(int(h), str(p)) for h, p in rows.groupby(["horizon_min", "carpark_id"]).groups}
    if not expected.issubset(present):
        raise ValueError(f"Missing calibration samples for {sorted(expected - present)}; never fill from test")
    records = []
    for (horizon, park), part in rows.groupby(["horizon_min", "carpark_id"], sort=True):
        for strategy in GATES:
            raw = (part["current_lots_available"] if strategy == GATES[0]
                   else (1 - part["M1"]) * part["total_lots"])
            residual = (raw - part["actual_lots_available"]).to_numpy(dtype=float)
            for q in QGRID:
                margin = max(0., float(np.quantile(residual, q, method="higher")))
                records.append(dict(horizon_min=horizon, carpark_id=park, strategy=strategy,
                                    quantile=q, margin_spaces=margin, calibration_samples=len(part),
                                    first_decision_time=part.decision_time.min(),
                                    last_decision_time=part.decision_time.max(), last_target_time=part.target_time.max(),
                                    empirical_residual_coverage=float((residual <= margin).mean())))
    return pd.DataFrame(records)


def rank_buffered(candidates, strategy, q, threshold, margins, *, by=()):
    if strategy not in GATES or q not in QGRID or threshold not in KGRID:
        raise ValueError("Use pre-fixed strategy, quantile, and threshold grid")
    selected = margins.loc[margins.strategy.eq(strategy) & margins["quantile"].eq(q),
                           ["horizon_min", "carpark_id", "margin_spaces"]]
    rows = candidates.merge(selected, on=["horizon_min", "carpark_id"], how="left", validate="many_to_one")
    capacity = rows["total_lots"]
    raw = (rows["current_lots_available"] if strategy == GATES[0]
           else (1 - rows["forecast_occupancy"]) * capacity)
    if not (np.isfinite(capacity) & capacity.gt(0) & np.isfinite(raw) & raw.between(0, capacity)
            & np.isfinite(rows["margin_spaces"]) & rows["margin_spaces"].ge(0)).all():
        raise ValueError("Missing calibration margin or invalid departure-time signal")
    estimate = (raw - rows["margin_spaces"]).clip(lower=0, upper=capacity)
    qualifies = estimate.ge(threshold) | np.isclose(estimate, threshold, rtol=0, atol=1e-9)
    rows = rows.assign(raw_estimated_free=raw, estimated_free=estimate, gate_qualifies=qualifies,
                       _gate_fail=~qualifies,
                       _gate_order=np.where(qualifies, rows["distance_m"], -estimate))
    return rows.sort_values([*by, "_gate_fail", "_gate_order", "distance_m", "carpark_id"], kind="stable")


def buffered_rankings(candidates, q, threshold, cohort, margins, *, strategies=GATES):
    if cohort not in base.COHORTS:
        raise ValueError("Unknown cohort")
    # Cohort eligibility always uses true observed departure availability.
    rows = candidates.loc[candidates.current_lots_available.gt(0)].copy() if cohort == "open_at_departure" else candidates.copy()
    if rows.empty:
        raise ValueError("No complete buffered-gate cases")
    nearest = rows.groupby(rec.CASE).distance_m.min().rename("nearest_distance_m")
    counts = rows.groupby(rec.CASE).size().rename("candidate_count")
    results = []
    for strategy in strategies:
        ranked = rank_buffered(rows, strategy, q, threshold, margins, by=rec.CASE)
        ranked["rank"] = ranked.groupby(rec.CASE, sort=False).cumcount() + 1
        top = ranked.loc[ranked["rank"].le(3)].copy()
        top["strategy"], top["cohort"], top["quantile"], top["threshold"] = strategy, cohort, q, threshold
        top = top.join(nearest, on=rec.CASE).join(counts, on=rec.CASE)
        top["extra_distance_m"] = top.distance_m - top.nearest_distance_m
        top["full_at_arrival"] = top.actual_lots_available.eq(0)
        top["low_spaces_at_arrival"] = top.actual_lots_available.lt(3)
        top["near_full_at_arrival"] = top.actual_lots_available.mul(10).lt(top.actual_total_lots)
        feasible = (~top.full_at_arrival).groupby([top[k] for k in rec.CASE]).any()
        top = top.join(feasible.rename("top3_any_available"), on=rec.CASE)
        results.append(top.drop(columns=["_gate_fail", "_gate_order"]))
    return pd.concat(results, ignore_index=True)


def summarize(first):
    rows = first.assign(strategy=first.strategy.replace(TO_BASE))
    summary = base.summarize(rows)
    summary["strategy"] = summary.strategy.replace(FROM_BASE)
    return summary


def paired_tables(first):
    pairs, daily = base.paired_tables(first.assign(strategy=first.strategy.replace(TO_BASE)))
    for old, new in FROM_BASE.items():
        pairs["comparison"] = pairs.comparison.str.replace(old, new, regex=False)
        daily["comparison"] = daily.comparison.str.replace(old, new, regex=False)
    return pairs, daily


def select_parameters(validation_metrics):
    data = validation_metrics
    if (set(data.split) != {"late_validation"} or set(data.cohort) != {"open_at_departure"}
            or set(data.strategy) != {"C0", GATES[0]}):
        raise ValueError("Selection accepts only current-buffer / C0 late-validation metrics")
    baseline = data.loc[data.strategy.eq("C0")].set_index("horizon_min").sort_index()
    current = data.loc[data.strategy.eq(GATES[0])]
    if set(baseline.index) != {15, 30} or not baseline.index.is_unique:
        raise ValueError("Exactly one C0 result for each horizon required")
    if set(zip(current["quantile"], current.threshold)) != {(q, k) for q in QGRID for k in KGRID}:
        raise ValueError("Complete pre-fixed quantile/threshold grid required")
    records = []
    for q in QGRID:
        for k in KGRID:
            rows = current.loc[current["quantile"].eq(q) & current.threshold.eq(k)].set_index("horizon_min").sort_index()
            if set(rows.index) != {15, 30} or not rows.index.is_unique:
                raise ValueError("Exactly one result per horizon / parameter combination required")
            if not rows.matched_cases.eq(baseline.matched_cases).all():
                raise ValueError("Identical matched denominators required for observed constraints")
            feasible = bool(rows.top1_full_count.le(baseline.top1_full_count).all()
                            and rows.top1_low_spaces_count.le(baseline.top1_low_spaces_count).all())
            row = dict(quantile=q, threshold=k, feasible_observed_constraints=feasible,
                       pooled_mean_distance_m=np.average(rows.mean_distance_m, weights=rows.matched_cases))
            for horizon in (15, 30):
                for field, column in (("full", "top1_full_count"), ("low", "top1_low_spaces_count")):
                    row[f"{field}_count_{horizon}"] = int(rows.loc[horizon, column])
                    row[f"{field}_budget_{horizon}"] = int(baseline.loc[horizon, column])
            records.append(row)
    table = pd.DataFrame(records)
    feasible = table.loc[table.feasible_observed_constraints]
    if feasible.empty:
        selected_q, selected_k = FALLBACK
    else:
        best = feasible.sort_values(["pooled_mean_distance_m", "quantile", "threshold"], kind="stable").iloc[0]
        selected_q, selected_k = float(best["quantile"]), int(best.threshold)
    table["selected"] = table["quantile"].eq(selected_q) & table.threshold.eq(selected_k)
    table["diagnostic_fallback_used"] = feasible.empty
    return selected_q, selected_k, table


def check_training_chronology(forecast_audit):
    boundaries = {}
    for horizon in (15, 30):
        train_last = pd.Timestamp(forecast_audit["horizons"][str(horizon)]["train"]["last_decision_time"])
        train_last_target = train_last + pd.Timedelta(minutes=horizon)
        if train_last_target >= CALIBRATION_START:
            raise ValueError("Forecast training labels overlap calibration period")
        boundaries[str(horizon)] = dict(last_training_origin=train_last.isoformat(), last_training_target=train_last_target.isoformat())
    return boundaries


def figures(metrics, pairs, q, k, output, *, validation_feasible):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    strategies = (*base.OLD_STRATEGIES, *GATES)
    names = ["A Nearest", "B Most free now", "C0 Current score", "C Forecast score", "XGB Full risk", "Cat Full risk", "Current + buffer", "Forecast + buffer"]
    colors = ["#93A7B8"] * 6 + ["#087D84", "#D57A29"]
    status = f"observed validation feasible: {validation_feasible} | q={q:g}, k={k}"
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.7))
    fig.subplots_adjust(left=.17, right=.98, top=.85, bottom=.13, wspace=.38, hspace=.43)
    fig.text(.03, .96, "Empirical buffered spare-space gate | retrospective extension", fontsize=18, fontweight="bold", color="#183042")
    fig.text(.03, .912, f"Point estimates | open at departure | {status}", fontsize=11, color="#526B80")
    selected = metrics.loc[metrics.cohort.eq("open_at_departure")]
    panels = [("mean_distance_m", None, "Mean distance (m)"), ("top1_full_rate", "top1_full_count", "Full: 0 spaces"),
              ("top1_low_spaces_rate", "top1_low_spaces_count", "Low: <3 spaces"), ("top1_near_full_rate", "top1_near_full_count", "Near-full: <10% free")]
    for row, horizon in enumerate((15, 30)):
        rows = selected.loc[selected.horizon_min.eq(horizon)].set_index("strategy").loc[list(strategies)]
        for col, (metric, count, label) in enumerate(panels):
            ax = axes[row, col]; values = rows[metric] * (100 if count else 1)
            ax.barh(np.arange(8), values, color=colors, height=.65)
            largest = max(float(values.max()), .01)
            for y, value, strategy in zip(range(8), values, strategies):
                label_text = f"{value:.1f}" if count is None else f"{value:.3f}% ({int(rows.loc[strategy, count])})"
                ax.text(value + largest * .025, y, label_text, va="center", fontsize=7.1)
            ax.set(yticks=np.arange(8), yticklabels=names if col == 0 else [], ylim=(7.65, -.65), xlim=(0, largest * 1.85))
            ax.set_title(f"{horizon} min | {label}", loc="left", fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8)
            for spine in ax.spines.values(): spine.set_visible(False)
    warning = ("Observed constraints are not safety guarantees. Static-buffer selection excludes forecast-buffer and test outcomes." if validation_feasible
               else "No grid pair met observed validation constraints. q=.95, k=3 is diagnostic only; retain original C0.")
    fig.text(.03, .067, warning, fontsize=10, color="#A34824" if not validation_feasible else "#526B80")
    fig.text(.03, .028, "Historical buffers are signal-specific. Full is primary; low-spaces and near-full are separate outcomes. No conformal coverage is claimed.", fontsize=9.5, color="#526B80")
    fig.savefig(output / "buffered_gate_outcomes.png", dpi=200, facecolor="white"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    fig.subplots_adjust(left=.12, right=.98, top=.77, bottom=.24, wspace=.36)
    fig.text(.04, .96, "Prediction increment with empirical historical buffers", fontsize=17, fontweight="bold", color="#183042")
    fig.text(.04, .9, f"Forecast minus current buffer | {status}", fontsize=10.5, color="#526B80")
    subset = pairs.loc[pairs.comparison.eq(f"{GATES[1]}-{GATES[0]}") & pairs.cohort.eq("open_at_departure")]
    for ax, metric, factor, label in zip(axes, ("mean_distance_m", "full_rate"), (1, 100), ("Distance difference (m)", "Full-rate difference (percentage points)")):
        for y, horizon in enumerate((15, 30)):
            r = subset.loc[subset.horizon_min.eq(horizon) & subset.metric.eq(metric)].iloc[0]
            x = r.estimate * factor
            if pd.notna(r.ci_low): ax.plot([r.ci_low * factor, r.ci_high * factor], [y, y], color="#087D84", lw=2)
            ax.scatter([x], [y], color="#D57A29", s=55, zorder=3)
            ax.annotate(f"Holm p={r.p_value_holm:.4f}" if pd.notna(r.p_value_holm) else "No day-level variation", (x, y), xytext=(0, 15), textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(0, color="#8496A4", lw=1, ls="--")
        ax.set(yticks=[0, 1], yticklabels=["15 min", "30 min"], ylim=(1.55, -.55), xlabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.margins(x=.25); ax.grid(axis="x", alpha=.15); ax.spines[["top", "right"]].set_visible(False)
        displayed = subset.loc[subset.metric.eq(metric), ["estimate", "ci_low", "ci_high"]].to_numpy(dtype=float) * factor
        if np.nanmax(np.abs(displayed)) < 1e-9:
            ax.set_xlim(-.03, .03)
            ax.set_xticks([-.02, 0, .02])
    fig.text(.04, .10, "95% paired day-block intervals, not multiplicity-adjusted. Holm correction spans the complete comparison CSV.", fontsize=9.5, color="#526B80")
    fig.text(.04, .055, "Negative favors the forecast buffer. Rare or zero event differences do not demonstrate equivalence or non-inferiority.", fontsize=9.5, color="#526B80")
    fig.savefig(output / "buffered_gate_paired_effects.png", dpi=200, facecolor="white"); plt.close(fig)


def run(study, metadata, destination, output):
    study, output = Path(study), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings_path = study / "recommendation/replay_settings.json"
    forecast_audit_path = study / "forecast/audit.json"
    settings = json.loads(settings_path.read_text())
    if settings["forecast_column"] != "M1" or settings["low_spaces_threshold"] != 3:
        raise ValueError("Pre-specified M1 / 3-space setup required")
    training = check_training_chronology(json.loads(forecast_audit_path.read_text()))
    destination_value = json.loads(Path(destination).read_text())
    if destination_value != settings["destination"]: raise ValueError("Destination differs from original")
    parks = rec.read_carparks(metadata, destination_value)
    forecast_path, risk_path = study / "forecast/predictions.csv", study / "risk/risk_predictions.csv"
    predictions = rec.read_predictions(forecast_path, ["split", "M1", "forecast_label_eligible", *rec.OBSERVATIONS])
    if set(predictions.split) != {"validation", "test"}: raise ValueError("Expected original validation and test only")
    test_mask = predictions.split.eq("test")
    if predictions.loc[test_mask, "decision_time"].min() != TEST_START: raise ValueError("Test boundary changed")
    if predictions.loc[~test_mask, "target_time"].max() >= TEST_START: raise ValueError("Validation labels overlap test")
    predictions["forecast_occupancy"] = predictions.M1
    calibration = calibration_rows(predictions)
    margins = estimate_margins(calibration, predictions.carpark_id.unique())
    margins.to_csv(output / "calibration_margins.csv", index=False, float_format="%.10g")
    late_mask = (predictions.split.eq("validation") & predictions.decision_time.ge(SELECTION_START)
                 & predictions.target_time.lt(TEST_START))
    late, late_case_audit = rec.matched_cases(predictions.loc[late_mask], parks)
    if late.empty or calibration.target_time.max() >= late.decision_time.min(): raise ValueError("Calibration and selection overlap")
    weight = settings["selected_distance_weight"]
    val_base, _ = rec.evaluate(late, ["C0"], weight, 3, "open_at_departure")
    val_tables = [val_base.assign(quantile=np.nan, threshold=np.nan)]
    for q in QGRID:
        for k in KGRID:
            ranks = buffered_rankings(late, q, k, "open_at_departure", margins)
            val_tables.append(summarize(ranks.loc[ranks["rank"].eq(1)]).assign(quantile=q, threshold=k))
    late_metrics = pd.concat(val_tables, ignore_index=True).assign(split="late_validation")
    selected_q, selected_k, selection = select_parameters(late_metrics.loc[late_metrics.strategy.isin(["C0", GATES[0]])])
    feasible = bool(selection.loc[selection.selected, "feasible_observed_constraints"].iat[0])
    late_metrics.to_csv(output / "late_validation_metrics.csv", index=False, float_format="%.10g")
    selection.to_csv(output / "parameter_selection.csv", index=False, float_format="%.10g")
    calibration_audit = {"training": training, "calibration_origin_start": CALIBRATION_START.isoformat(),
                         "calibration_origin_exclusive_end": (SELECTION_START - EMBARGO).isoformat(),
                         "calibration_target_exclusive_end": SELECTION_START.isoformat(), "embargo_minutes": 30,
                         "actual_first_origin": calibration.decision_time.min().isoformat(),
                         "actual_last_origin": calibration.decision_time.max().isoformat(),
                         "actual_last_target": calibration.target_time.max().isoformat(),
                         "calibration_samples": len(calibration), "same_samples_for_both_signals": True,
                         "same_segment_label_eligibility_required": True, "quantile_method": "higher",
                         "margin_definition": "max(0, higher quantile(raw estimated spaces - observed arrival spaces))",
                         "selection_first_origin": late.decision_time.min().isoformat(),
                         "selection_last_target": late.target_time.max().isoformat(),
                         "forecast_training_precedes_calibration": True, "test_origin_start": TEST_START.isoformat()}
    (output / "calibration_audit.json").write_text(json.dumps(calibration_audit, indent=2) + "\n")
    print(json.dumps({"stage": "late_validation_selection_complete", "quantile": selected_q,
                      "threshold": selected_k, "observed_validation_feasible": feasible,
                      "feasible_grid_pairs": int(selection.feasible_observed_constraints.sum())}), flush=True)
    risk = rec.read_predictions(risk_path, ["low_spaces_threshold", *rec.OBSERVATIONS, *rec.RISK_SIGNALS.values()])
    if not risk.low_spaces_threshold.eq(3).all(): raise ValueError("Risk event threshold changed")
    joined = rec.join_risk(predictions, risk)
    test, test_case_audit = rec.matched_cases(joined.loc[test_mask], parks, risk=True)
    old_ranks = pd.concat([rec.evaluate(test, base.OLD_STRATEGIES, weight, 3, cohort)[1] for cohort in base.COHORTS], ignore_index=True)
    original_path = study / "recommendation/recommendation_rankings.csv"
    published_path = study / "evaluation/recommendation_metrics.csv"
    base.verify_original(read_inputs(original_path), old_ranks, pd.read_csv(published_path))
    sensitivity, selected_gates = [], []
    for q in QGRID:
        for k in KGRID:
            ranks = pd.concat([buffered_rankings(test, q, k, cohort, margins) for cohort in base.COHORTS], ignore_index=True)
            sensitivity.append(summarize(ranks.loc[ranks["rank"].eq(1)]).assign(quantile=q, threshold=k))
            if (q, k) == (selected_q, selected_k): selected_gates.append(ranks)
    selected_rankings = pd.concat([old_ranks, *selected_gates], ignore_index=True)
    first = ranked_choices(selected_rankings)
    metrics = summarize(first)
    metrics["quantile"] = np.where(metrics.strategy.isin(GATES), selected_q, np.nan)
    metrics["threshold"] = np.where(metrics.strategy.isin(GATES), selected_k, np.nan)
    metrics["diagnostic_only"] = metrics.strategy.isin(GATES) & (not feasible)
    pairs, daily = paired_tables(first)
    for name, frame in {"recommendation_metrics": metrics, "paired_policy_comparisons": pairs,
                        "parameter_sensitivity": pd.concat(sensitivity, ignore_index=True), "daily_policy_diagnostics": daily}.items():
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.10g")
    columns = [*rec.KEY, "target_time", "cohort", "strategy", "rank", "candidate_count", "quantile", "threshold",
               "distance_m", "extra_distance_m", "current_lots_available", "total_lots", "forecast_occupancy",
               "raw_estimated_free", "margin_spaces", "estimated_free", "gate_qualifies", "actual_lots_available",
               "actual_total_lots", "full_at_arrival", "low_spaces_at_arrival", "near_full_at_arrival"]
    selected_rankings.reindex(columns=columns).to_csv(output / "selected_rankings.csv", index=False, float_format="%.10g")
    inputs = [forecast_path, forecast_audit_path, risk_path, settings_path, original_path, published_path, Path(metadata), Path(destination)]
    audit = {"analysis_type": "Retrospective RQ2 empirical-buffer extension; not independent confirmation",
             "input_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: base.sha256(p) for p in inputs},
             "quantile_grid": QGRID, "threshold_grid": KGRID, "selected_quantile": selected_q, "selected_threshold": selected_k,
             "selected_observed_validation_constraints_satisfied": feasible, "diagnostic_fallback_used": not feasible,
             "accepted_replacement": "CURRENT_BUFFERED_GATE is validation-feasible for further study" if feasible else "None; retain original C0",
             "selection": "CURRENT_BUFFERED_GATE late-validation full and <3 counts no greater than same-pool C0 at each horizon; minimum pooled distance; ties smaller q then k. Forecast-buffer/test outcomes excluded. No feasible pair uses fixed (.95,3) only diagnostically.",
             "gate_rule": "clip(raw spaces minus signal-specific calibration margin, 0, current capacity); >=k nearest first, otherwise largest buffered estimate then distance then ID",
             "forecast_spaces": "(1-M1)*current total_lots; no future capacity for ranking",
             "historical_current_baseline": "Current buffer uses historical change residuals, but no M1 signal; it is not original C0",
             "cohort_filter": "True current_lots_available > 0, never buffer-adjusted availability",
             "quantile_method": "higher", "boundary_numeric_tolerance": 1e-9,
             "warning": "Empirical residual buffers and observed validation constraints are not safety/conformal guarantees; adjacent residuals are dependent",
             "primary_cohort": "open_at_departure", "primary_comparison": f"{GATES[1]}-{GATES[0]}", "primary_event": "full: zero spaces",
             "secondary_events": ["low: <3 spaces", "near-full: <10% free including full"],
             "original_weight": weight, "original_rankings_and_metrics_verified": True,
             "test_start": test.decision_time.min().isoformat(), "test_end": test.decision_time.max().isoformat(),
             "resampling": "10000 paired Singapore departure-date blocks; seed 20261105; observation-weighted differences",
             "inference": "95% percentile intervals not multiplicity adjusted; two-sided day-sign flips; Holm across complete paired_policy_comparisons.csv",
             "assumptions": "Exchangeable independent date blocks; symmetric day effects for sign flips; neighboring-day dependence not modelled",
             "interpretation": "Forecast-vs-current buffer isolates the signal-plus-signal-specific-buffer increment. Current buffer vs C0 is a rule/history increment; forecast buffer vs C0 combines both. Non-significance is not equivalence.",
             "sensitivity": "All pre-fixed q/k test metrics retained; never used for choosing parameters",
             "method_search_stops_after_this_extension": True}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    figures(metrics, pairs, selected_q, selected_k, output, validation_feasible=feasible)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs/history_comparison/two_year_standard")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/chinatown_annual_carparks.csv")
    parser.add_argument("--destination", type=Path, default=ROOT / "data/chinatown_destination.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rq2_buffered_gate")
    args = parser.parse_args()
    print(json.dumps(run(args.study.resolve(), args.metadata.resolve(), args.destination.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__": main()
