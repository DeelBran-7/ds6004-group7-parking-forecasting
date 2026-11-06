"""Run a recent-year control, an expanded-history comparison and a two-year study."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/history_comparison")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare the recent-year input and manifest")
    args = parser.parse_args()
    source, output = args.history.resolve(), args.output.resolve()
    data = pd.read_csv(source)
    times = pd.to_datetime(data["timestamp_sgt"], format="ISO8601")
    recent_start = times.max().normalize() + pd.Timedelta(days=1) - pd.DateOffset(years=1)
    if times.min() >= recent_start:
        raise ValueError("The comparison needs observations before the most recent year")
    recent = data.loc[times.ge(recent_start)]
    recent_times = pd.DatetimeIndex(times.loc[recent.index].unique()).sort_values()
    validation_start = recent_times[len(recent_times) * 3 // 5].isoformat()
    test_start = recent_times[len(recent_times) * 4 // 5].isoformat()
    output.mkdir(parents=True, exist_ok=True)
    recent_path = output / "one_year_input.csv"
    recent.to_csv(recent_path, index=False)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    manifest = {
        "source_filename": source.name, "source_sha256": digest.hexdigest(),
        "source_rows": len(data), "source_start": times.min().isoformat(), "source_end": times.max().isoformat(),
        "recent_year_start": recent_start.isoformat(), "recent_year_rows": len(recent),
        "shared_validation_start": validation_start, "shared_test_start": test_start,
        "embargo_minutes": 30, "threshold_free_spaces": 3,
        "comparison": "Extend the training prefix while preserving recent-year validation and test dates",
        "additional_run": "Full history with chronological 60/20/20; evaluate separately",
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared inputs and experiment manifest in {output}")
        return
    common = [sys.executable, str(ROOT / "run_study.py"),
              "--holidays", str(ROOT / "data/sg_public_holidays_2020_2027.csv"),
              "--metadata", str(ROOT / "data/chinatown_annual_carparks.csv"),
              "--destination", str(ROOT / "data/chinatown_destination.json")]
    shared = ["--validation-start", validation_start, "--test-start", test_start]
    for name, path, boundaries in (("one_year", recent_path, shared), ("two_year_shared", source, shared),
                                    ("two_year_standard", source, [])):
        subprocess.run([*common, "--history", str(path), "--output", str(output / name), *boundaries],
                       cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/compare_history_results.py"),
                    "--one-year", str(output / "one_year"), "--two-year", str(output / "two_year_shared"),
                    "--output", str(output / "comparison")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
