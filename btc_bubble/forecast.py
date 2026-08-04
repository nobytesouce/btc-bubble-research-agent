from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def forecast_bubble_sizes(events: pd.DataFrame, window: int = 30, minimum_history: int = 5) -> pd.DataFrame:
    """Predict each next qualifying bubble from qualifying bubbles seen earlier.

    The conditional hierarchy mirrors the detector. It prefers side/hour/volatility
    history, then progressively falls back to broader groups. The rolling median is
    used because exceptional trade sizes are strongly right-skewed.
    """
    data = events.sort_values("timestamp").reset_index(drop=True).copy()
    predictions: list[float] = []
    levels: list[str] = []
    for i, event in data.iterrows():
        history = data.iloc[:i]
        choices = (
            (history[(history["side"] == event["side"]) & (history["hour"] == event["hour"]) & (history["vol_regime"] == event["vol_regime"])], "side-hour-vol"),
            (history[(history["side"] == event["side"]) & (history["vol_regime"] == event["vol_regime"])], "side-vol"),
            (history[history["side"] == event["side"]], "side"),
            (history, "global"),
        )
        chosen = next(((sample, level) for sample, level in choices if len(sample) >= minimum_history), None)
        if chosen is None:
            predictions.append(np.nan)
            levels.append("insufficient-history")
            continue
        sample, level = chosen
        predictions.append(float(sample["cluster_q_usd"].tail(window).median()))
        levels.append(level)
    data["predicted_bubble_usd"] = predictions
    data["actual_bubble_usd"] = data["cluster_q_usd"].astype(float)
    data["prediction_level"] = levels
    data["absolute_error_usd"] = (data["predicted_bubble_usd"] - data["actual_bubble_usd"]).abs()
    return data


def forecast_summary(forecasts: pd.DataFrame, next_count: int = 5, window: int = 30) -> tuple[dict, pd.DataFrame]:
    valid = forecasts.dropna(subset=["predicted_bubble_usd"])
    if valid.empty:
        return {"status": "insufficient_history"}, pd.DataFrame()
    actual = valid["actual_bubble_usd"].to_numpy(dtype=float)
    predicted = valid["predicted_bubble_usd"].to_numpy(dtype=float)
    intervals = forecasts["timestamp"].diff().dropna().tail(window)
    median_interval_ms = int(intervals.median()) if len(intervals) else 60_000
    next_size = float(forecasts["actual_bubble_usd"].tail(window).median())
    last_timestamp = int(forecasts["timestamp"].iloc[-1])
    future = pd.DataFrame({
        "forecast_number": np.arange(1, next_count + 1),
        "predicted_timestamp": [last_timestamp + median_interval_ms * step for step in range(1, next_count + 1)],
        "predicted_bubble_usd": next_size,
    })
    summary = {
        "status": "ok",
        "qualifying_bubbles": int(len(forecasts)),
        "evaluated_predictions": int(len(valid)),
        "median_actual_bubble_usd": float(np.median(actual)),
        "median_predicted_bubble_usd": float(np.median(predicted)),
        "mean_absolute_error_usd": float(np.mean(np.abs(predicted - actual))),
        "median_absolute_percentage_error": float(np.median(np.abs(predicted - actual) / actual)),
        "next_five_predicted_bubble_usd": next_size,
        "estimated_median_interval_seconds": median_interval_ms / 1000.0,
    }
    return summary, future


def write_forecast_chart(path: str | Path, forecasts: pd.DataFrame, future: pd.DataFrame) -> Path:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    valid = forecasts.dropna(subset=["predicted_bubble_usd"]).copy()
    times = pd.to_datetime(valid["timestamp"], unit="ms", utc=True)

    fig, price_axis = plt.subplots(figsize=(15, 7.5))
    bubble_axis = price_axis.twinx()
    price_axis.plot(times, valid["price"], color="#2563eb", linewidth=1.8, label="BTC price")
    bubble_axis.plot(times, valid["predicted_bubble_usd"] / 1_000_000, color="#f59e0b", linewidth=2.0, label="Predicted large bubble")
    bubble_axis.plot(times, valid["actual_bubble_usd"] / 1_000_000, color="#10b981", linewidth=1.2, alpha=0.75, label="Actual large bubble")

    if not future.empty:
        future_times = pd.to_datetime(future["predicted_timestamp"], unit="ms", utc=True)
        bridge_times = pd.Index([times.iloc[-1]]).append(pd.Index(future_times))
        bridge_values = [valid["predicted_bubble_usd"].iloc[-1] / 1_000_000] + list(future["predicted_bubble_usd"] / 1_000_000)
        bubble_axis.plot(bridge_times, bridge_values, color="#f59e0b", linestyle="--", linewidth=2.0, label="Next five forecast")

    price_axis.set_title("BTC price vs predicted and actual qualifying bubble size")
    price_axis.set_xlabel("UTC time")
    price_axis.set_ylabel("BTC price (USDT)", color="#2563eb")
    bubble_axis.set_ylabel("Bubble notional (USD millions)")
    price_axis.grid(alpha=0.2)
    price_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=times.dt.tz))
    lines = price_axis.get_lines() + bubble_axis.get_lines()
    price_axis.legend(lines, [line.get_label() for line in lines], loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target
