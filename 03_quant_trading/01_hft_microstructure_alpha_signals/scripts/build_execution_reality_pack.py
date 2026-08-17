from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.execution_reality import adverse_selection_report, latency_sensitivity_table, queue_fill_probability


def main() -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    signals = pd.DataFrame({"expected_edge_bps": [3.2, 2.8, 2.4, 1.9, 2.1]})
    latency = latency_sensitivity_table(signals)
    fills = pd.DataFrame(
        {
            "side": ["buy", "sell", "buy", "sell"],
            "fill_price": [100.00, 100.02, 99.98, 100.03],
            "mid_after_fill": [99.99, 100.01, 100.00, 100.04],
        }
    )
    adverse = adverse_selection_report(fills)
    queue_prob = queue_fill_probability(120, 3.5, 1.2, 25)
    latency.to_csv(reports / "latency_sensitivity.csv", index=False)
    (reports / "execution_reality_pack.md").write_text(
        "\n".join(
            [
                "# HFT Execution Reality Pack",
                "",
                f"- Queue-fill probability sample: {queue_prob:.3f}",
                f"- Toxic fill share sample: {adverse['toxic_fill_share']:.2%}",
                f"- Tradable latency buckets: {int(latency['tradable'].sum())}/{len(latency)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
