# src/research_report.py
"""
Create a concise Markdown research report from generated pipeline artifacts.
"""
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from config import FEATURES_PARQUET, REPORTS_DIR


REPORT_PATH = REPORTS_DIR / "research_report.md"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_research_report(output_path: Path = REPORT_PATH) -> Path:
    summary = _load_json(REPORTS_DIR / "pipeline_summary.json")
    stationarity = _load_json(REPORTS_DIR / "stationarity_tests.json")
    validation = _load_json(REPORTS_DIR / "validation_results.json")

    metrics = summary.get("backtest_metrics", {})
    cleaning = summary.get("cleaning_summary", {})

    feature_snapshot = {}
    if FEATURES_PARQUET.exists():
        df = pd.read_parquet(FEATURES_PARQUET)
        feature_snapshot = {
            "shape": df.shape,
            "start": df.index.min(),
            "end": df.index.max(),
            "missing_pct": float(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
        }

    lines = [
        "# Real Market Data Engineering - Research Report",
        "",
        "## Executive Assessment",
        "",
        "This run rebuilds the India VIX minute-data pipeline with corrected ISO timestamp parsing, bounded OHLC repair, leakage-safe forward targets, and a one-bar-delayed diagnostic backtest.",
        "",
        "## Data Integrity",
        "",
        f"- Rows processed: {summary.get('total_rows', 'n/a')}",
        f"- Date range: {summary.get('date_range', 'n/a')}",
        f"- Raw OHLC-invalid rows: {cleaning.get('raw_ohlc_invalid', 'n/a')}",
        f"- Bad ticks flagged: {cleaning.get('bad_ticks', 'n/a')} ({_fmt_num(cleaning.get('bad_ticks_pct'), 2)}%)",
        f"- Post-clean OHLC-invalid rows: {cleaning.get('post_clean_ohlc_invalid', 'n/a')}",
        f"- Volume informative: {cleaning.get('volume_is_informative', 'n/a')}",
        "",
        "## Stationarity",
        "",
    ]

    for name, result in stationarity.items():
        lines.extend([
            f"- {name}: {result.get('overall_conclusion', 'n/a')}",
            f"- {name} observations tested: {result.get('n_obs_tested', 'n/a')} of {result.get('n_obs_raw', 'n/a')}",
            f"- {name} ADF p-value: {_fmt_num(result.get('adf', {}).get('pvalue'), 6)}",
            f"- {name} KPSS p-value: {_fmt_num(result.get('kpss', {}).get('pvalue'), 6)}",
        ])

    lines.extend([
        "",
        "## Feature Set",
        "",
        f"- Feature columns created: {summary.get('total_features', 'n/a')}",
        f"- Forward target columns created: {summary.get('total_targets', 'n/a')}",
        f"- Feature matrix shape: {feature_snapshot.get('shape', 'n/a')}",
        f"- Feature matrix missing-value share: {_fmt_num(feature_snapshot.get('missing_pct'), 2)}%",
        "",
        "## Diagnostic Backtest",
        "",
        f"- Total return: {_fmt_pct(metrics.get('total_return'))}",
        f"- Annualized return: {_fmt_pct(metrics.get('annualized_return'))}",
        f"- Annualized volatility: {_fmt_pct(metrics.get('annualized_volatility'))}",
        f"- Sharpe ratio: {_fmt_num(metrics.get('sharpe_ratio'), 3)}",
        f"- Max drawdown: {_fmt_pct(metrics.get('max_drawdown'))}",
        f"- Trades: {metrics.get('total_trades', 'n/a')}",
        f"- Total turnover: {_fmt_num(metrics.get('total_turnover'), 2)}",
        f"- Total cost drag: {_fmt_pct(metrics.get('total_cost_sum'))}",
        "",
        "Note: the backtest is a diagnostic data-engineering sanity check, not an investable strategy claim. India VIX is an index, so live deployment would require a mapped tradeable instrument and exchange-specific execution assumptions.",
        "",
        "## Validation",
        "",
        f"- Tests passed: {validation.get('tests_passed', 'n/a')}",
        f"- Tests failed: {validation.get('tests_failed', 'n/a')}",
        f"- Warnings: {validation.get('warnings', 'n/a')}",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    path = generate_research_report()
    print(f"Saved research report to {path}")
