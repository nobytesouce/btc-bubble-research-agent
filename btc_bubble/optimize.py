from __future__ import annotations

from itertools import product
import numpy as np
import pandas as pd

from .backtest import performance_summary


def walk_forward_search(results: pd.DataFrame, horizons: tuple[int, ...]) -> dict:
    """Select on early chronological folds and report the sealed final 20% once."""
    if results.empty or len(results) < 20:
        return {"status": "insufficient_data", "minimum_events": 20}
    ordered = results.sort_values("timestamp").reset_index(drop=True)
    holdout_start = max(1, int(len(ordered) * 0.80))
    development = ordered.iloc[:holdout_start]
    holdout = ordered.iloc[holdout_start:]
    candidates = []
    thresholds = product((0.99, 0.995, 0.999), (0.85, 0.90, 0.95))
    for size_gate, volume_gate in thresholds:
        development_subset = development[
            (development["p_size"] >= size_gate)
            & ((development["p_volume"] >= volume_gate) | (development["p_depth"] >= volume_gate))
        ]
        for family, horizon, stop_bps, take_bps in product(
            ("continuation", "reversal"), horizons, (10, 25, 50), (10, 25, 50)
        ):
            column = f"{family}_pnl_{horizon}s_sl{stop_bps}_tp{take_bps}"
            fold_scores = []
            for index_group in np.array_split(np.arange(len(development_subset)), 4):
                fold = development_subset.iloc[index_group]
                summary = performance_summary(fold, column)
                if summary["signals"] >= 3:
                    fold_scores.append(summary.get("net_return", 0.0))
            if len(fold_scores) < 3:
                continue
            stability = float(np.mean(fold_scores) - np.std(fold_scores))
            candidates.append({
                "size_threshold": size_gate,
                "volume_or_depth_threshold": volume_gate,
                "family": family,
                "horizon_seconds": horizon,
                "stop_loss_bps": stop_bps,
                "take_profit_bps": take_bps,
                "score": stability,
                "pnl_column": column,
            })
    if not candidates:
        return {"status": "insufficient_data"}
    champion = max(candidates, key=lambda item: item["score"])
    selected_holdout = holdout[
        (holdout["p_size"] >= champion["size_threshold"])
        & (
            (holdout["p_volume"] >= champion["volume_or_depth_threshold"])
            | (holdout["p_depth"] >= champion["volume_or_depth_threshold"])
        )
    ]
    return {
        "status": "ok",
        "selection_rule": "mean fold net return minus fold standard deviation",
        "development_events": int(len(development)),
        "holdout_events": int(len(holdout)),
        "champion": champion,
        "holdout": performance_summary(selected_holdout, champion["pnl_column"]),
    }
