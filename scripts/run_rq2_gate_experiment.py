"""Retrospective paired RQ2 extension: sufficient estimated spaces, then nearest.

One common gate is selected using CURRENT_GATE validation outcomes only. The
forecast gate and every test outcome are excluded from threshold selection.
This changes the ranking rule, so comparisons against old C0 do not by
themselves measure the incremental value of forecasting.
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
import recommendation as rec
from evaluate import correct_family, ranked_choices, read_inputs, require_same_values
from evaluation_stats import paired_day_inference

GATES = ("CURRENT_GATE", "FORECAST_GATE")
OLD_STRATEGIES = (*rec.BASE_STRATEGIES, *rec.RISK_SIGNALS)
STRATEGIES = (*OLD_STRATEGIES, *GATES)
GRID = (1, 3, 5, 10, 20)
COHORTS = ("visitor_eligible", "open_at_departure")
COMPARISONS = (("FORECAST_GATE", "CURRENT_GATE"), ("CURRENT_GATE", "C0"), ("FORECAST_GATE", "C0"))
METRICS = {"full_rate": "full_at_arrival", "low_spaces_rate": "low_spaces_at_arrival",
           "near_full_rate": "near_full_at_arrival", "mean_distance_m": "distance_m",
           "top3_feasibility": "top3_any_available"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_gate(candidates, strategy, threshold, *, by=()):
    """The ordering uses current capacity and estimates only, never arrival labels."""
    if strategy not in GATES or isinstance(threshold, bool) or threshold < 1 or int(threshold) != threshold:
        raise ValueError("Recognized gate and positive integer threshold required")
    capacity = candidates["total_lots"]
    estimate = (candidates["current_lots_available"] if strategy == "CURRENT_GATE"
                else (1 - candidates["forecast_occupancy"]) * capacity)
    if not (np.isfinite(capacity) & capacity.gt(0) & np.isfinite(estimate) & estimate.between(0, capacity)).all():
        raise ValueError("Gate needs valid current capacity and estimated availability")
    qualifies = estimate.ge(threshold) | np.isclose(estimate, threshold, rtol=0, atol=1e-9)
    ranked = candidates.assign(estimated_free=estimate, gate_qualifies=qualifies,
                                _gate_fail=~qualifies,
                                _gate_order=np.where(qualifies, candidates["distance_m"], -estimate))
    return ranked.sort_values([*by, "_gate_fail", "_gate_order", "distance_m", "carpark_id"], kind="stable")


def gate_rankings(candidates, threshold, cohort, *, strategies=GATES):
    rows = candidates.loc[candidates["current_lots_available"].gt(0)].copy() if cohort == "open_at_departure" else candidates.copy()
    if rows.empty:
        raise ValueError("No complete gate cases")
    nearest = rows.groupby(rec.CASE)["distance_m"].min().rename("nearest_distance_m")
    counts = rows.groupby(rec.CASE).size().rename("candidate_count")
    rankings = []
    for strategy in strategies:
        ranked = rank_gate(rows, strategy, threshold, by=rec.CASE)
        ranked["rank"] = ranked.groupby(rec.CASE, sort=False).cumcount() + 1
        top = ranked.loc[ranked["rank"].le(3)].copy()
        top["strategy"], top["cohort"], top["threshold"] = strategy, cohort, threshold
        top = top.join(nearest, on=rec.CASE).join(counts, on=rec.CASE)
        top["extra_distance_m"] = top["distance_m"] - top["nearest_distance_m"]
        top["full_at_arrival"] = top["actual_lots_available"].eq(0)
        top["low_spaces_at_arrival"] = top["actual_lots_available"].lt(3)
        top["near_full_at_arrival"] = top["actual_lots_available"].mul(10).lt(top["actual_total_lots"])
        feasible = (~top["full_at_arrival"]).groupby([top[k] for k in rec.CASE]).any()
        top = top.join(feasible.rename("top3_any_available"), on=rec.CASE)
        rankings.append(top.drop(columns=["_gate_fail", "_gate_order"]))
    return pd.concat(rankings, ignore_index=True)


def summarize(first):
    metrics = []
    for (horizon, cohort, strategy), rows in first.groupby(["horizon_min", "cohort", "strategy"], sort=True):
        metrics.append({"horizon_min": horizon, "cohort": cohort, "strategy": strategy,
                        "matched_cases": len(rows), "decision_dates": rows["decision_time"].dt.normalize().nunique(),
                        "top1_full_count": int(rows["full_at_arrival"].sum()),
                        "top1_full_rate": rows["full_at_arrival"].mean(),
                        "top1_low_spaces_count": int(rows["low_spaces_at_arrival"].sum()),
                        "top1_low_spaces_rate": rows["low_spaces_at_arrival"].mean(),
                        "top1_near_full_count": int(rows["near_full_at_arrival"].sum()),
                        "top1_near_full_rate": rows["near_full_at_arrival"].mean(),
                        "top3_any_available_rate": rows["top3_any_available"].mean(),
                        "mean_distance_m": rows["distance_m"].mean(),
                        "mean_extra_vs_nearest_m": rows["extra_distance_m"].mean(),
                        "gate_any_qualifying_rate": rows["gate_qualifies"].astype(bool).mean() if strategy in GATES else None,
                        "gate_fallback_rate": (~rows["gate_qualifies"].astype(bool)).mean() if strategy in GATES else None})
    return pd.DataFrame(metrics)


def select_threshold(validation_metrics):
    """Only a static validation table is admitted; no forecasts or test labels."""
    data = validation_metrics
    if (set(data["split"]) != {"validation"} or set(data["cohort"]) != {"open_at_departure"}
            or set(data["strategy"]) != {"C0", "CURRENT_GATE"}):
        raise ValueError("Selection accepts static-gate / C0 validation metrics only")
    baseline = data.loc[data["strategy"].eq("C0")].set_index("horizon_min")
    current = data.loc[data["strategy"].eq("CURRENT_GATE")]
    if set(baseline.index) != {15, 30} or not baseline.index.is_unique or set(current["threshold"]) != set(GRID):
        raise ValueError("Expected both horizons and the complete fixed threshold grid")
    records = []
    for threshold in GRID:
        rows = current.loc[current["threshold"].eq(threshold)].set_index("horizon_min").sort_index()
        if set(rows.index) != {15, 30} or not rows.index.is_unique:
            raise ValueError("One static-gate validation result per threshold and horizon required")
        if not rows["matched_cases"].eq(baseline["matched_cases"]).all():
            raise ValueError("Validation constraints require matched baseline denominators")
        full_ok = rows["top1_full_count"].le(baseline["top1_full_count"])
        low_ok = rows["top1_low_spaces_count"].le(baseline["top1_low_spaces_count"])
        record = {"threshold": threshold, "feasible_observed_constraints": bool(full_ok.all() and low_ok.all()),
                  "pooled_mean_distance_m": np.average(rows["mean_distance_m"], weights=rows["matched_cases"])}
        for horizon in (15, 30):
            record.update({f"full_count_{horizon}": int(rows.loc[horizon, "top1_full_count"]),
                           f"full_budget_{horizon}": int(baseline.loc[horizon, "top1_full_count"]),
                           f"low_count_{horizon}": int(rows.loc[horizon, "top1_low_spaces_count"]),
                           f"low_budget_{horizon}": int(baseline.loc[horizon, "top1_low_spaces_count"])})
        records.append(record)
    table = pd.DataFrame(records)
    feasible = table.loc[table["feasible_observed_constraints"]]
    selected = (int(feasible.sort_values(["pooled_mean_distance_m", "threshold"]).iloc[0]["threshold"])
                if not feasible.empty else 3)
    table["selected"] = table["threshold"].eq(selected)
    table["fallback_used"] = feasible.empty
    return selected, table


def paired_tables(first):
    comparisons, diagnostics = [], []
    for (horizon, cohort), subset in first.groupby(["horizon_min", "cohort"], sort=True):
        for candidate, reference in COMPARISONS:
            left = subset.loc[subset["strategy"].eq(candidate)].set_index("decision_time").sort_index()
            right = subset.loc[subset["strategy"].eq(reference)].set_index("decision_time").sort_index()
            if not left.index.equals(right.index):
                raise ValueError("Unmatched policy comparison")
            changed = left["carpark_id"].ne(right["carpark_id"])
            common = {"horizon_min": horizon, "cohort": cohort, "comparison": f"{candidate}-{reference}",
                      "changed_first_choices": int(changed.sum()), "changed_first_choice_rate": changed.mean()}
            for metric, column in METRICS.items():
                values, baseline = left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float)
                stats = paired_day_inference(left.index.normalize(), values, baseline,
                                             repetitions=10000, seed=20261105)
                comparisons.append({**common, "metric": metric,
                                    "primary": candidate == "FORECAST_GATE" and reference == "CURRENT_GATE"
                                               and cohort == "open_at_departure" and metric == "full_rate",
                                    "candidate_events": int(values.sum()) if metric not in {"mean_distance_m"} else None,
                                    "reference_events": int(baseline.sum()) if metric not in {"mean_distance_m"} else None,
                                    **stats})
            daily = pd.DataFrame({"date": left.index.normalize(), "changed": changed.to_numpy(),
                                  "full_avoided": (~left["full_at_arrival"] & right["full_at_arrival"]).to_numpy(),
                                  "new_full": (left["full_at_arrival"] & ~right["full_at_arrival"]).to_numpy(),
                                  "distance_delta_m": (left["distance_m"] - right["distance_m"]).to_numpy()})
            daily = daily.groupby("date").agg(matched_cases=("changed", "size"), changed_first_choices=("changed", "sum"),
                                              full_avoided=("full_avoided", "sum"), new_full=("new_full", "sum"),
                                              mean_distance_delta_m=("distance_delta_m", "mean")).reset_index()
            daily["changed_first_choice_rate"] = daily["changed_first_choices"] / daily["matched_cases"]
            diagnostics.append(daily.assign(horizon_min=horizon, cohort=cohort, comparison=common["comparison"]))
    return correct_family(pd.DataFrame(comparisons)), pd.concat(diagnostics, ignore_index=True)


def verify_original(original, rebuilt, published_metrics):
    keys = [*rec.CASE, "cohort", "strategy", "rank"]
    a, b = original.set_index(keys).sort_index(), rebuilt.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Rebuilt original recommendation keys changed")
    for col in ("carpark_id", "candidate_count", "distance_m", "actual_lots_available", "actual_total_lots"):
        require_same_values(a[col], b[col], f"Original rankings changed at {col}")
    metrics = summarize(ranked_choices(rebuilt))
    keys = ["horizon_min", "cohort", "strategy"]
    a, b = published_metrics.set_index(keys).sort_index(), metrics.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Original aggregate keys changed")
    for col in a.columns:
        require_same_values(a[col], b[col], f"Original summary changed at {col}")


def figures(metrics, pairs, threshold, output, *, validation_feasible):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    names = ["A Nearest", "B Most free now", "C0 Current score", "C Forecast score", "XGB Full risk", "Cat Full risk", "Current gate", "Forecast gate"]
    colors = ["#93A7B8"] * 6 + ["#087D84", "#D57A29"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.7))
    fig.subplots_adjust(left=.17, right=.98, top=.85, bottom=.13, wspace=.38, hspace=.43)
    status = f"validation-selected k = {threshold}" if validation_feasible else f"diagnostic fallback k = {threshold}"
    fig.text(.03, .96, f"Nearest after a spare-space gate | {status}", fontsize=18, fontweight="bold", color="#183042")
    fig.text(.03, .912, "Point estimates | same departures and candidates | spaces available at departure", fontsize=11, color="#526B80")
    selected = metrics.loc[metrics["cohort"].eq("open_at_departure")]
    panels = [("mean_distance_m", None, "Mean distance (m)"), ("top1_full_rate", "top1_full_count", "Full: 0 spaces"),
              ("top1_low_spaces_rate", "top1_low_spaces_count", "Low: <3 spaces"), ("top1_near_full_rate", "top1_near_full_count", "Near-full: <10% free")]
    for row, horizon in enumerate((15, 30)):
        rows = selected.loc[selected["horizon_min"].eq(horizon)].set_index("strategy").loc[list(STRATEGIES)]
        for col, (metric, count, label) in enumerate(panels):
            ax = axes[row, col]
            values = rows[metric] * (100 if count else 1)
            ax.barh(np.arange(8), values, color=colors, height=.65)
            largest = max(float(values.max()), .01)
            for y, value, strategy in zip(range(8), values, STRATEGIES):
                text = f"{value:.1f}" if count is None else f"{value:.3f}% ({int(rows.loc[strategy, count])})"
                ax.text(value + largest * .025, y, text, va="center", fontsize=7.1)
            ax.set(yticks=np.arange(8), yticklabels=names if col == 0 else [], ylim=(7.65, -.65), xlim=(0, largest * 1.85))
            ax.set_title(f"{horizon} min | {label}", loc="left", fontsize=10, fontweight="bold")
            ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8)
            for spine in ax.spines.values(): spine.set_visible(False)
    warning = ("The common threshold uses static-gate validation outcomes only; observed constraints are not safety guarantees." if validation_feasible
               else "No grid threshold met the observed validation constraints. k=3 is the pre-fixed diagnostic fallback, not an accepted replacement.")
    fig.text(.03, .067, warning, fontsize=10, color="#A34824" if not validation_feasible else "#526B80")
    fig.text(.03, .028, "Full is the primary event; low-spaces and near-full are separate outcomes. This is a retrospective extension to the original ranking rule.", fontsize=9.5, color="#526B80")
    fig.savefig(output / "gate_outcomes.png", dpi=200, facecolor="white"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    fig.subplots_adjust(left=.12, right=.98, top=.77, bottom=.24, wspace=.36)
    fig.text(.04, .96, "Incremental prediction value under the same gate", fontsize=17, fontweight="bold", color="#183042")
    fig.text(.04, .9, f"Forecast gate minus current gate | {status} | open at departure", fontsize=11, color="#526B80")
    subset = pairs.loc[pairs["comparison"].eq("FORECAST_GATE-CURRENT_GATE") & pairs["cohort"].eq("open_at_departure")]
    for ax, metric, factor, label in zip(axes, ("mean_distance_m", "full_rate"), (1, 100), ("Distance difference (m)", "Full-rate difference (percentage points)")):
        for y, horizon in enumerate((15, 30)):
            r = subset.loc[subset["horizon_min"].eq(horizon) & subset["metric"].eq(metric)].iloc[0]
            x = r["estimate"] * factor
            if pd.notna(r["ci_low"]):
                ax.plot([r["ci_low"] * factor, r["ci_high"] * factor], [y, y], color="#087D84", lw=2)
            ax.scatter([x], [y], color="#D57A29", s=55, zorder=3)
            ax.annotate(f"Holm p={r['p_value_holm']:.4f}" if pd.notna(r["p_value_holm"]) else "No day-level variation", (x, y), xytext=(0, 15), textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(0, color="#8496A4", lw=1, ls="--")
        ax.set(yticks=[0, 1], yticklabels=["15 min", "30 min"], ylim=(1.55, -.55), xlabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f" if metric == "mean_distance_m" else "%.2f"))
        ax.margins(x=.25); ax.grid(axis="x", alpha=.15); ax.spines[["top", "right"]].set_visible(False)
    fig.text(.04, .10, "95% paired day-block intervals, not multiplicity-adjusted. Holm correction spans the complete comparison CSV.", fontsize=9.5, color="#526B80")
    fig.text(.04, .055, "Negative favors the forecast gate. Rare or zero event differences do not demonstrate equivalence or non-inferiority.", fontsize=9.5, color="#526B80")
    fig.savefig(output / "gate_paired_effects.png", dpi=200, facecolor="white"); plt.close(fig)


def run(study, metadata, destination, output):
    study, output = Path(study), Path(output)
    settings_path = study / "recommendation/replay_settings.json"
    settings = json.loads(settings_path.read_text())
    if settings["forecast_column"] != "M1" or settings["low_spaces_threshold"] != 3:
        raise ValueError("This extension is pre-specified for M1 and a low-space threshold of 3")
    destination_value = json.loads(Path(destination).read_text())
    if settings["destination"] != destination_value:
        raise ValueError("Destination changed from the original study")
    parks = rec.read_carparks(metadata, destination_value)
    forecast_path, risk_path = study / "forecast/predictions.csv", study / "risk/risk_predictions.csv"
    predictions = rec.read_predictions(forecast_path, ["split", "M1", *rec.OBSERVATIONS])
    if set(predictions["split"]) != {"validation", "test"}:
        raise ValueError("Original validation and test predictions required")
    val_mask = predictions["split"].eq("validation")
    if predictions.loc[val_mask, "target_time"].max() >= predictions.loc[~val_mask, "decision_time"].min():
        raise ValueError("Validation outcomes overlap test decisions")
    predictions["forecast_occupancy"] = predictions["M1"]
    risk = rec.read_predictions(risk_path, ["low_spaces_threshold", *rec.OBSERVATIONS, *rec.RISK_SIGNALS.values()])
    if not risk["low_spaces_threshold"].eq(3).all(): raise ValueError("Risk event threshold changed")
    predictions = rec.join_risk(predictions, risk)
    validation, val_case_audit = rec.matched_cases(predictions.loc[val_mask], parks)
    test, test_case_audit = rec.matched_cases(predictions.loc[~val_mask], parks, risk=True)
    weight = settings["selected_distance_weight"]
    val_base, _ = rec.evaluate(validation, ["C0"], weight, 3, "open_at_departure")
    val_base["threshold"] = np.nan
    val_tables = [val_base]
    for k in GRID:
        ranks = gate_rankings(validation, k, "open_at_departure", strategies=("CURRENT_GATE",))
        val_tables.append(summarize(ranks.loc[ranks["rank"].eq(1)]).assign(threshold=k))
    validation_metrics = pd.concat(val_tables, ignore_index=True).assign(split="validation")
    selected, selection = select_threshold(validation_metrics)
    old_ranks = pd.concat([rec.evaluate(test, OLD_STRATEGIES, weight, 3, cohort)[1] for cohort in COHORTS], ignore_index=True)
    original = read_inputs(study / "recommendation/recommendation_rankings.csv")
    published = pd.read_csv(study / "evaluation/recommendation_metrics.csv")
    verify_original(original, old_ranks, published)
    sensitivity, selected_gates = [], []
    for threshold in GRID:
        ranks = pd.concat([gate_rankings(test, threshold, cohort) for cohort in COHORTS], ignore_index=True)
        sensitivity.append(summarize(ranks.loc[ranks["rank"].eq(1)]).assign(threshold=threshold))
        if threshold == selected: selected_gates.append(ranks)
    selected_rankings = pd.concat([old_ranks, *selected_gates], ignore_index=True)
    first = ranked_choices(selected_rankings)
    metrics = summarize(first)
    validation_feasible = bool(selection.loc[selection.selected, "feasible_observed_constraints"].iat[0])
    metrics["threshold"] = np.where(metrics["strategy"].isin(GATES), selected, np.nan)
    metrics["diagnostic_only"] = metrics["strategy"].isin(GATES) & (not validation_feasible)
    pairs, daily = paired_tables(first)
    output.mkdir(parents=True, exist_ok=True)
    outputs = {"validation_metrics": validation_metrics, "threshold_selection": selection,
               "recommendation_metrics": metrics, "paired_policy_comparisons": pairs,
               "threshold_sensitivity": pd.concat(sensitivity, ignore_index=True), "daily_policy_diagnostics": daily}
    for name, frame in outputs.items(): frame.to_csv(output / f"{name}.csv", index=False, float_format="%.10g")
    ranking_columns = [*rec.KEY, "target_time", "cohort", "strategy", "rank", "candidate_count", "threshold",
                       "distance_m", "extra_distance_m", "current_lots_available", "total_lots", "forecast_occupancy",
                       "estimated_free", "gate_qualifies", "actual_lots_available", "actual_total_lots",
                       "full_at_arrival", "low_spaces_at_arrival", "near_full_at_arrival"]
    selected_rankings.reindex(columns=ranking_columns).to_csv(output / "selected_rankings.csv", index=False, float_format="%.10g")
    inputs = [forecast_path, risk_path, settings_path, study / "recommendation/recommendation_rankings.csv",
              study / "evaluation/recommendation_metrics.csv", Path(metadata), Path(destination)]
    audit = {"analysis_type": "Retrospective RQ2 ranking-rule extension; not independent confirmation",
             "input_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: sha256(p) for p in inputs},
             "threshold_grid": GRID, "selected_threshold": selected,
             "selected_observed_validation_constraints_satisfied": bool(selection.loc[selection.selected, "feasible_observed_constraints"].iat[0]),
             "fallback_used": bool(selection.fallback_used.iat[0]),
             "selection": "Static CURRENT_GATE only; at each horizon observed full and <3-space rates no greater than old C0; among feasible thresholds minimize pooled distance, ties lower k; no feasible uses fixed k=3",
             "selection_warning": "Observed validation feasibility is not a guaranteed risk bound or a test of non-inferiority",
             "gate_rule": "At least k estimated spaces: nearest first; otherwise descending estimated spaces, then distance and ID",
             "forecast_spaces": "(1 - M1 occupancy) * current total_lots; future capacity never used for ranking",
             "boundary_numeric_tolerance": 1e-9, "cohorts": COHORTS, "primary_cohort": "open_at_departure",
             "primary_comparison": "FORECAST_GATE-CURRENT_GATE", "primary_event": "full at arrival: zero spaces",
             "secondary_events": ["low: <3 spaces", "near-full: <10% free, including full"],
             "original_weight": weight, "original_rankings_and_metrics_verified": True,
             "validation_start": validation.decision_time.min().isoformat(), "validation_last_target": validation.target_time.max().isoformat(),
             "test_start": test.decision_time.min().isoformat(), "test_end": test.decision_time.max().isoformat(),
             "resampling": "10000 paired Singapore departure-date blocks, seed 20261105; observation-weighted differences",
             "inference": "95% percentile intervals not multiplicity adjusted; two-sided day-sign flips; Holm over all comparisons in paired_policy_comparisons.csv",
             "assumptions": "Exchangeable independent date blocks; symmetric day effects for sign flips; neighboring-day dependence not modelled",
             "interpretation": "Forecast-vs-current gate isolates prediction under the new rule. Current gate vs old C0 measures rule change. Forecast gate vs old C0 combines both. No significant difference does not prove equivalence or non-inferiority.",
             "sensitivity": "All pre-fixed test thresholds exported; test results never choose threshold"}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    figures(metrics, pairs, selected, output, validation_feasible=validation_feasible)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs/history_comparison/two_year_standard")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/chinatown_annual_carparks.csv")
    parser.add_argument("--destination", type=Path, default=ROOT / "data/chinatown_destination.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rq2_gate")
    args = parser.parse_args()
    print(json.dumps(run(args.study.resolve(), args.metadata.resolve(), args.destination.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__": main()
