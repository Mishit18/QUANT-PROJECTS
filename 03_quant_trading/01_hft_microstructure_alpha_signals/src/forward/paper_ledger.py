from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


GENESIS_HASH = "0" * 64


def _digest(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_paper_ledger(
    snapshots: Iterable[dict[str, Any]],
    *,
    imbalance_threshold: float = 0.20,
    cost_bps: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(snapshots)
    if len(rows) < 2:
        raise ValueError("at least two chronological snapshots are required")

    ledger: list[dict[str, Any]] = []
    previous_hash = GENESIS_HASH
    position = 0
    cumulative_net_return = 0.0
    turnover = 0.0
    gross_pnl = 0.0
    cost_paid = 0.0

    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        imbalance = float(previous["depth_imbalance"])
        deviation = float(previous["microprice_deviation_bps"])
        signal = 0
        if imbalance > imbalance_threshold and deviation > 0:
            signal = 1
        elif imbalance < -imbalance_threshold and deviation < 0:
            signal = -1

        target_position = signal
        step_turnover = abs(target_position - position)
        midpoint_return = float(current["midpoint"]) / float(previous["midpoint"]) - 1.0
        gross_return = position * midpoint_return
        transaction_cost = step_turnover * cost_bps / 10_000.0
        net_return = gross_return - transaction_cost

        turnover += step_turnover
        gross_pnl += gross_return
        cost_paid += transaction_cost
        cumulative_net_return += net_return
        position = target_position

        body = {
            "sequence": index,
            "signal_time": previous["received_at"],
            "mark_time": current["received_at"],
            "signal": signal,
            "position": position,
            "midpoint": float(current["midpoint"]),
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "cumulative_net_return": cumulative_net_return,
            "turnover": step_turnover,
            "absolute_exposure": abs(position),
            "previous_hash": previous_hash,
        }
        record_hash = _digest(body)
        record = {**body, "record_hash": record_hash}
        ledger.append(record)
        previous_hash = record_hash

    summary = {
        "records": len(ledger),
        "signals_active": sum(record["signal"] != 0 for record in ledger),
        "turnover_units": turnover,
        "mean_absolute_exposure": sum(record["absolute_exposure"] for record in ledger) / len(ledger),
        "gross_return": gross_pnl,
        "transaction_cost": cost_paid,
        "net_return": cumulative_net_return,
        "cost_bps_per_turnover": cost_bps,
        "final_record_hash": previous_hash,
        "study_type": "forward-captured public data with shadow paper positions",
        "execution_boundary": "no orders submitted; midpoint marks and modeled costs only",
    }
    return ledger, summary


def write_ledger(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def verify_ledger(records: Iterable[dict[str, Any]]) -> bool:
    previous_hash = GENESIS_HASH
    for record in records:
        body = {key: value for key, value in record.items() if key != "record_hash"}
        if body.get("previous_hash") != previous_hash:
            return False
        if _digest(body) != record.get("record_hash"):
            return False
        previous_hash = record["record_hash"]
    return True

