"""Export benchmark and stress-test result files to CSV tables."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def export_benchmarks() -> Path:
    benchmark_path = RESULTS_DIR / "benchmark_results.npy"
    results = np.load(benchmark_path, allow_pickle=True).item()

    rows = []
    for strategy, metrics in results.items():
        rows.append({"strategy": strategy, **metrics})

    output_path = RESULTS_DIR / "benchmark_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def export_stress_tests() -> Path:
    stress_path = RESULTS_DIR / "stress_test_results.npy"
    results = np.load(stress_path, allow_pickle=True).item()

    rows = []
    for scenario, scenario_results in results.items():
        for strategy, metrics in scenario_results.items():
            rows.append({"scenario": scenario, "strategy": strategy, **metrics})

    output_path = RESULTS_DIR / "stress_test_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    benchmark_csv = export_benchmarks()
    stress_csv = export_stress_tests()
    print(f"[OK] Wrote {benchmark_csv.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Wrote {stress_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
