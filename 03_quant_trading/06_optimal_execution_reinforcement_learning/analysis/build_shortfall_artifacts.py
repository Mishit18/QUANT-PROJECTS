from __future__ import annotations

from pathlib import Path

import pandas as pd

from shortfall import participation_guardrail, shortfall_decomposition


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    fills = pd.DataFrame(
        {
            "price": [100.01, 100.02, 100.05, 100.03],
            "mid_price": [100.00, 100.01, 100.03, 100.02],
            "quantity": [100, 150, 200, 250],
            "market_volume": [1500, 1200, 900, 1600],
        }
    )
    decomp = shortfall_decomposition(fills, arrival_price=100.0, side="buy")
    guarded = participation_guardrail(fills)
    results = ROOT / "results"
    report = ROOT / "report"
    results.mkdir(exist_ok=True)
    report.mkdir(exist_ok=True)
    guarded.to_csv(results / "participation_guardrail_sample.csv", index=False)
    pd.DataFrame([decomp]).to_csv(results / "shortfall_decomposition_sample.csv", index=False)
    (report / "SHORTFALL_EVIDENCE_PACK.md").write_text(
        "\n".join(
            [
                "# Shortfall Evidence Pack",
                "",
                f"- Implementation shortfall: {decomp['implementation_shortfall_bps']:.3f} bps",
                f"- Spread cost: {decomp['spread_cost_bps']:.3f} bps",
                f"- Participation breaches: {int(guarded['participation_breach'].sum())}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
