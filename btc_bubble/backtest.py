from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig


def _window_outcomes(prices: np.ndarray, side_sign: float, entry: float) -> tuple[float, float]:
    signed_returns = side_sign * (prices / entry - 1.0)
    return float(np.max(signed_returns)), float(np.min(signed_returns))


def _barrier_pnl(
    prices: np.ndarray,
    side_sign: float,
    entry: float,
    stop_bps: int,
    take_bps: int,
    cost: float,
) -> float:
    returns = side_sign * (prices / entry - 1.0)
    stop = -stop_bps / 10_000.0
    take = take_bps / 10_000.0
    for value in returns:
        if value <= stop:
            return float(stop - cost)
        if value >= take:
            return float(take - cost)
    return float(returns[-1] - cost)


def backtest_events(trades: pd.DataFrame, events: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    source = trades.sort_values("timestamp").reset_index(drop=True)
    ts = source["timestamp"].to_numpy(dtype=np.int64)
    price = source["price"].to_numpy(dtype=np.float64)
    cvd = source["cvd_60s"].to_numpy(dtype=np.float64)
    total_cost = 2.0 * (
        config.execution.taker_fee_bps_each_side + config.execution.slippage_bps_each_side
    ) / 10_000.0
    rows: list[dict] = []
    max_horizon = max(config.event.horizons_seconds)
    for event in events.itertuples(index=False):
        entry_idx = int(np.searchsorted(ts, event.timestamp, side="right"))
        if entry_idx >= len(source):
            continue
        entry = price[entry_idx]
        sign = 1.0 if event.side == "buy" else -1.0
        end_idx = int(np.searchsorted(ts, event.timestamp + max_horizon * 1000, side="right"))
        path = price[entry_idx:max(entry_idx + 1, end_idx)]
        mfe, mae = _window_outcomes(path, sign, entry)
        result = {
            "event_id": int(event.event_id),
            "timestamp": int(event.timestamp),
            "side": event.side,
            "signal_price": float(event.price),
            "bubble_usd": float(event.cluster_q_usd),
            "rolling_median_bubble_usd": float(event.rolling_median_bubble_usd),
            "entry_price": float(entry),
            "p_size": float(event.p_size),
            "p_volume": float(event.p_volume) if np.isfinite(event.p_volume) else np.nan,
            "p_depth": float(event.p_depth) if np.isfinite(event.p_depth) else np.nan,
            "bbs": float(event.bbs) if np.isfinite(event.bbs) else np.nan,
            "depth_exact": bool(event.depth_exact),
            "frozen_vwap": float(event.frozen_vwap) if np.isfinite(event.frozen_vwap) else np.nan,
            "mfe": mfe,
            "mae": mae,
            "cvd_continued": bool(sign * (cvd[min(end_idx - 1, len(cvd) - 1)] - event.cvd_60s) > 0),
            "liquidation_label": "unknown",
        }
        if "open_interest" in source and np.isfinite(source.loc[entry_idx, "open_interest"]):
            final_oi = source.loc[min(max(end_idx - 1, entry_idx), len(source) - 1), "open_interest"]
            initial_oi = source.loc[entry_idx, "open_interest"]
            result["open_interest_rise"] = bool(np.isfinite(final_oi) and final_oi > initial_oi)
            result["open_interest_change"] = float(final_oi - initial_oi) if np.isfinite(final_oi) else np.nan
        else:
            result["open_interest_rise"] = None
            result["open_interest_change"] = np.nan
        target = result["frozen_vwap"]
        if np.isfinite(target):
            result["reached_frozen_vwap"] = bool(np.min(path) <= target <= np.max(path))
        else:
            result["reached_frozen_vwap"] = False
        for horizon in config.event.horizons_seconds:
            exit_idx = int(np.searchsorted(ts, event.timestamp + horizon * 1000, side="right")) - 1
            exit_idx = min(max(exit_idx, entry_idx), len(price) - 1)
            raw = sign * (price[exit_idx] / entry - 1.0)
            result[f"continuation_pnl_{horizon}s"] = float(raw - total_cost)
            result[f"reversal_pnl_{horizon}s"] = float(-raw - total_cost)
            result[f"continued_{horizon}s"] = bool(raw > 0)
            result[f"reversed_{horizon}s"] = bool(raw < 0)
            horizon_path = price[entry_idx : exit_idx + 1]
            for stop_bps in (10, 25, 50):
                for take_bps in (10, 25, 50):
                    result[f"continuation_pnl_{horizon}s_sl{stop_bps}_tp{take_bps}"] = _barrier_pnl(
                        horizon_path, sign, entry, stop_bps, take_bps, total_cost
                    )
                    result[f"reversal_pnl_{horizon}s_sl{stop_bps}_tp{take_bps}"] = _barrier_pnl(
                        horizon_path, -sign, entry, stop_bps, take_bps, total_cost
                    )
        rows.append(result)
    return pd.DataFrame(rows)


def performance_summary(results: pd.DataFrame, pnl_column: str) -> dict:
    if results.empty or pnl_column not in results:
        return {"signals": 0, "win_rate": None, "profit_factor": None, "sharpe": None, "max_drawdown": None}
    pnl = results[pnl_column].dropna().to_numpy(dtype=np.float64)
    if len(pnl) == 0:
        return {"signals": 0, "win_rate": None, "profit_factor": None, "sharpe": None, "max_drawdown": None}
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    equity = np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    drawdown = equity - peaks
    return {
        "signals": int(len(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "sharpe": float(np.mean(pnl) / np.std(pnl, ddof=1) * np.sqrt(len(pnl))) if len(pnl) > 1 and np.std(pnl, ddof=1) > 0 else None,
        "max_drawdown": float(np.min(drawdown)) if len(drawdown) else 0.0,
        "net_return": float(np.sum(pnl)),
    }
