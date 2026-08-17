import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.quote_stress import quote_fill_probability, spread_compression_scenarios, toxic_flow_stress


def test_quote_stress_outputs_verdicts():
    prob = quote_fill_probability(1.0, 0.8, 2.0)
    stress = toxic_flow_stress(2.0, 1.0, prob)
    scenarios = spread_compression_scenarios(3.0, 1.5)

    assert 0 <= prob <= 1
    assert stress["verdict"] in {"quote", "widen_or_skip"}
    assert len(scenarios) == 4

