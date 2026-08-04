from __future__ import annotations

from pathlib import Path
import json

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
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.12, top=0.90)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def aggregate_forecast_frames(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    combined["date"] = pd.to_datetime(combined["timestamp"], unit="ms", utc=True).dt.date.astype(str)
    valid = combined.dropna(subset=["predicted_bubble_usd", "actual_bubble_usd"])
    daily = valid.groupby("date", sort=True).agg(
        btc_price=("price", "median"),
        predicted_bubble_usd=("predicted_bubble_usd", "median"),
        actual_bubble_usd=("actual_bubble_usd", "median"),
        qualifying_bubbles=("actual_bubble_usd", "size"),
    ).reset_index()
    actual = valid["actual_bubble_usd"].to_numpy(dtype=float)
    predicted = valid["predicted_bubble_usd"].to_numpy(dtype=float)
    summary = {
        "status": "ok" if len(valid) else "insufficient_history",
        "days": int(daily["date"].nunique()),
        "qualifying_bubbles": int(len(combined)),
        "evaluated_predictions": int(len(valid)),
        "median_actual_bubble_usd": float(np.median(actual)) if len(actual) else None,
        "median_predicted_bubble_usd": float(np.median(predicted)) if len(predicted) else None,
        "mean_absolute_error_usd": float(np.mean(np.abs(predicted - actual))) if len(actual) else None,
        "median_absolute_percentage_error": float(np.median(np.abs(predicted - actual) / actual)) if len(actual) else None,
    }
    return combined, daily, summary


def write_two_month_chart(path: str | Path, daily: pd.DataFrame) -> Path:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(daily["date"], utc=True)
    fig, price_axis = plt.subplots(figsize=(15, 7.5))
    bubble_axis = price_axis.twinx()
    price_axis.plot(dates, daily["btc_price"], color="#2563eb", linewidth=2.0, label="BTC price")
    bubble_axis.plot(dates, daily["predicted_bubble_usd"] / 1_000_000, color="#f59e0b", linewidth=2.0, label="Predicted large bubble")
    bubble_axis.plot(dates, daily["actual_bubble_usd"] / 1_000_000, color="#10b981", linewidth=2.0, label="Actual large bubble")
    price_axis.set_title("Two-month BTC bubble prediction evaluation (daily medians)")
    price_axis.set_xlabel("UTC date")
    price_axis.set_ylabel("BTC price (USDT)", color="#2563eb")
    bubble_axis.set_ylabel("Bubble notional (USD millions)")
    price_axis.grid(alpha=0.2)
    price_axis.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    price_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    lines = price_axis.get_lines() + bubble_axis.get_lines()
    price_axis.legend(lines, [line.get_label() for line in lines], loc="upper left", ncol=3)
    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.16, top=0.90)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def aggregate_forecast_directory(input_dir: str | Path, output_dir: str | Path) -> dict:
    source = Path(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("**/bubble-forecast-*.csv"))
    if not files:
        raise FileNotFoundError(f"No forecast CSV files found under {source}")
    combined, daily, summary = aggregate_forecast_frames([pd.read_csv(path) for path in files])
    combined_path = output / "two-month-bubble-events.csv"
    daily_path = output / "two-month-daily-comparison.csv"
    summary_path = output / "two-month-summary.json"
    chart_path = output / "two-month-bubble-prediction-chart.png"
    combined.to_csv(combined_path, index=False)
    daily.to_csv(daily_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_two_month_chart(chart_path, daily)
    return {"summary": summary, "chart": str(chart_path), "daily_csv": str(daily_path), "events_csv": str(combined_path)}
