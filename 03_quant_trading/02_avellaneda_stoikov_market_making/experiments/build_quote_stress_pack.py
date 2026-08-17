from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.quote_stress import spread_compression_scenarios


def main() -> None:
    results = ROOT / "results" / "tables"
    docs = ROOT / "docs"
    results.mkdir(parents=True, exist_ok=True)
    docs.mkdir(exist_ok=True)
    scenarios = spread_compression_scenarios(3.0, 1.4)
    scenarios.to_csv(results / "quote_stress_scenarios.csv", index=False)
    (docs / "QUOTE_STRESS_PACK.md").write_text(
        "\n".join(
            [
                "# Quote Stress Pack",
                "",
                f"- Scenarios tested: {len(scenarios)}",
                f"- Quote-approved scenarios: {int((scenarios['verdict'] == 'quote').sum())}",
                "- Added fill probability, toxic-flow stress, and spread-compression diagnostics.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

