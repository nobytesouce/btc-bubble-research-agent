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


def build_two_hour_samples(
    market: pd.DataFrame,
    bubbles: pd.DataFrame,
    horizon_seconds: int = 7200,
    calibration_max_timestamp: int | None = None,
) -> pd.DataFrame:
    """Create non-overlapping forecasts issued before their complete target window."""
    source = market.sort_values("timestamp").reset_index(drop=True)
    events = bubbles.sort_values("timestamp").reset_index(drop=True)
    horizon_ms = int(horizon_seconds * 1000)
    calibration_max = (
        int(calibration_max_timestamp)
        if calibration_max_timestamp is not None
        else int(source["timestamp"].iloc[0]) - 1
    )
    first_issue = int(source["timestamp"].iloc[0]) + horizon_ms
    last_issue = int(source["timestamp"].iloc[-1]) - horizon_ms
    columns = [
        "issue_timestamp",
        "target_end_timestamp",
        "price_at_issue",
        "past_bubble_count",
        "past_2h_mean_bubble_usd",
        "past_2h_median_bubble_usd",
        "actual_next_2h_mean_bubble_usd",
        "actual_next_2h_bubble_count",
        "calibration_max_timestamp",
        "latest_past_event_timestamp",
        "earliest_future_event_timestamp",
        "lookahead_violation",
    ]
    rows: list[dict] = []
    for issue in range(first_issue, last_issue + 1, horizon_ms):
        past = events[(events["timestamp"] > issue - horizon_ms) & (events["timestamp"] <= issue)]
        future = events[(events["timestamp"] > issue) & (events["timestamp"] <= issue + horizon_ms)]
        if past.empty or future.empty:
            continue
        price_idx = int(np.searchsorted(source["timestamp"].to_numpy(dtype=np.int64), issue, side="right")) - 1
        price_idx = max(price_idx, 0)
        rows.append({
            "issue_timestamp": issue,
            "target_end_timestamp": issue + horizon_ms,
            "price_at_issue": float(source["price"].iloc[price_idx]),
            "past_bubble_count": int(len(past)),
            "past_2h_mean_bubble_usd": float(past["cluster_q_usd"].mean()),
            "past_2h_median_bubble_usd": float(past["cluster_q_usd"].median()),
            "actual_next_2h_mean_bubble_usd": float(future["cluster_q_usd"].mean()),
            "actual_next_2h_bubble_count": int(len(future)),
            "calibration_max_timestamp": calibration_max,
            "latest_past_event_timestamp": int(past["timestamp"].max()),
            "earliest_future_event_timestamp": int(future["timestamp"].min()),
            "lookahead_violation": bool(
                calibration_max >= issue
                or int(past["timestamp"].max()) > issue
                or int(future["timestamp"].min()) <= issue
                or int(future["timestamp"].max()) > issue + horizon_ms
            ),
        })
    samples = pd.DataFrame(rows, columns=columns)
    validate_two_hour_samples(samples)
    return samples


def validate_two_hour_samples(samples: pd.DataFrame) -> None:
    """Fail closed if calibration/features and targets cross the issue boundary."""
    if samples.empty:
        return
    violations = (
        samples["lookahead_violation"].astype(bool)
        | (samples["calibration_max_timestamp"] >= samples["issue_timestamp"])
        | (samples["latest_past_event_timestamp"] > samples["issue_timestamp"])
        | (samples["earliest_future_event_timestamp"] <= samples["issue_timestamp"])
        | (samples["target_end_timestamp"] <= samples["issue_timestamp"])
    )
    if bool(violations.any()):
        raise ValueError(f"Lookahead audit failed for {int(violations.sum())} two-hour samples")


def predict_two_hour_averages(samples: pd.DataFrame, shrinkage_count: float = 20.0) -> tuple[pd.DataFrame, dict]:
    """Sequentially predict the next two-hour mean without using an unfinished target."""
    data = samples.sort_values("issue_timestamp").reset_index(drop=True).copy()
    predictions: list[float] = []
    latest_completed_target: list[float] = []
    for row in data.itertuples(index=False):
        completed = data[
            (data["target_end_timestamp"] <= row.issue_timestamp)
            & np.isfinite(data["actual_next_2h_mean_bubble_usd"])
        ]
        historical = (
            float(completed["actual_next_2h_mean_bubble_usd"].expanding().mean().iloc[-1])
            if len(completed)
            else float(row.past_2h_mean_bubble_usd)
        )
        weight = float(row.past_bubble_count / (row.past_bubble_count + shrinkage_count))
        predictions.append(weight * float(row.past_2h_mean_bubble_usd) + (1.0 - weight) * historical)
        latest_completed_target.append(float(completed["target_end_timestamp"].max()) if len(completed) else np.nan)
    data["predicted_next_2h_mean_bubble_usd"] = predictions
    data["latest_historical_target_end_used"] = latest_completed_target
    history_violation = data["latest_historical_target_end_used"].notna() & (
        data["latest_historical_target_end_used"] > data["issue_timestamp"]
    )
    if bool(history_violation.any()):
        raise ValueError(f"Prediction history audit failed for {int(history_violation.sum())} forecasts")
    actual = data["actual_next_2h_mean_bubble_usd"].to_numpy(dtype=float)
    predicted = data["predicted_next_2h_mean_bubble_usd"].to_numpy(dtype=float)
    summary = {
        "status": "ok" if len(data) else "insufficient_history",
        "forecast_horizon_hours": 2,
        "forecast_samples": int(len(data)),
        "days": int(pd.to_datetime(data["issue_timestamp"], unit="ms", utc=True).dt.date.nunique()) if len(data) else 0,
        "median_actual_next_2h_mean_usd": float(np.median(actual)) if len(actual) else None,
        "median_predicted_next_2h_mean_usd": float(np.median(predicted)) if len(predicted) else None,
        "mean_absolute_error_usd": float(np.mean(np.abs(predicted - actual))) if len(actual) else None,
        "median_absolute_percentage_error": float(np.median(np.abs(predicted - actual) / actual)) if len(actual) else None,
        "correlation": float(np.corrcoef(predicted, actual)[0, 1]) if len(actual) > 1 else None,
        "lookahead_violations": int(data["lookahead_violation"].astype(bool).sum()) if "lookahead_violation" in data else 0,
        "history_boundary_violations": int(history_violation.sum()),
        "causality_audit_passed": bool(
            ("lookahead_violation" not in data or not data["lookahead_violation"].astype(bool).any())
            and not history_violation.any()
        ),
    }
    return data, summary


def write_two_hour_chart(path: str | Path, forecasts: pd.DataFrame) -> Path:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(forecasts["issue_timestamp"], unit="ms", utc=True)
    fig, price_axis = plt.subplots(figsize=(15, 7.5))
    bubble_axis = price_axis.twinx()
    price_axis.plot(dates, forecasts["price_at_issue"], color="#2563eb", linewidth=2.0, label="BTC price at forecast")
    bubble_axis.plot(dates, forecasts["predicted_next_2h_mean_bubble_usd"] / 1_000_000, color="#f59e0b", linewidth=2.0, label="Predicted next-2h average")
    bubble_axis.plot(dates, forecasts["actual_next_2h_mean_bubble_usd"] / 1_000_000, color="#10b981", linewidth=2.0, label="Actual next-2h average")
    price_axis.set_title("BTC average bubble size forecast issued two hours in advance")
    price_axis.set_xlabel("Forecast issue time (UTC)")
    price_axis.set_ylabel("BTC price (USDT)", color="#2563eb")
    bubble_axis.set_ylabel("Average qualifying bubble notional (USD millions)")
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


def aggregate_two_hour_directory(input_dir: str | Path, output_dir: str | Path) -> dict:
    source = Path(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("**/two-hour-samples-*.csv"))
    if not files:
        raise FileNotFoundError(f"No two-hour sample files found under {source}")
    frames = [pd.read_csv(path) for path in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError("All two-hour sample files were empty")
    forecasts, summary = predict_two_hour_averages(pd.concat(frames, ignore_index=True))
    forecast_path = output / "two-hour-ahead-forecasts.csv"
    summary_path = output / "two-hour-ahead-summary.json"
    chart_path = output / "two-hour-ahead-chart.png"
    forecasts.to_csv(forecast_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_two_hour_chart(chart_path, forecasts)
    return {"summary": summary, "chart": str(chart_path), "forecasts_csv": str(forecast_path)}


def build_24_hour_forecasts(events: pd.DataFrame, shrinkage_count: float = 20.0) -> tuple[pd.DataFrame, dict]:
    """Predict each UTC day's average bubble size using completed prior days only."""
    data = events.sort_values("timestamp").reset_index(drop=True).copy()
    value_column = "actual_bubble_usd" if "actual_bubble_usd" in data else "cluster_q_usd"
    price_column = "price" if "price" in data else "signal_price"
    data["date"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True).dt.floor("D")
    daily = data.groupby("date", sort=True).agg(
        price_at_forecast=(price_column, "median"),
        actual_next_24h_mean_bubble_usd=(value_column, "mean"),
        actual_next_24h_bubble_count=(value_column, "size"),
        earliest_target_event_timestamp=("timestamp", "min"),
        latest_target_event_timestamp=("timestamp", "max"),
    ).reset_index()
    rows: list[dict] = []
    for i in range(1, len(daily)):
        target = daily.iloc[i]
        prior = daily.iloc[i - 1]
        completed = daily.iloc[:i]
        issue = int(target["date"].timestamp() * 1000)
        target_end = issue + 24 * 60 * 60 * 1000
        historical = float(completed["actual_next_24h_mean_bubble_usd"].mean())
        weight = float(prior["actual_next_24h_bubble_count"] / (prior["actual_next_24h_bubble_count"] + shrinkage_count))
        prediction = weight * float(prior["actual_next_24h_mean_bubble_usd"]) + (1.0 - weight) * historical
        latest_information = int(prior["latest_target_event_timestamp"])
        earliest_target = int(target["earliest_target_event_timestamp"])
        latest_target = int(target["latest_target_event_timestamp"])
        violation = latest_information >= issue or earliest_target < issue or latest_target >= target_end
        rows.append({
            "issue_timestamp": issue,
            "target_end_timestamp": target_end,
            "price_at_forecast": float(target["price_at_forecast"]),
            "previous_24h_mean_bubble_usd": float(prior["actual_next_24h_mean_bubble_usd"]),
            "previous_24h_bubble_count": int(prior["actual_next_24h_bubble_count"]),
            "predicted_next_24h_mean_bubble_usd": prediction,
            "actual_next_24h_mean_bubble_usd": float(target["actual_next_24h_mean_bubble_usd"]),
            "actual_next_24h_bubble_count": int(target["actual_next_24h_bubble_count"]),
            "latest_information_timestamp": latest_information,
            "earliest_target_event_timestamp": earliest_target,
            "latest_target_event_timestamp": latest_target,
            "lookahead_violation": bool(violation),
        })
    forecasts = pd.DataFrame(rows)
    if not forecasts.empty and bool(forecasts["lookahead_violation"].any()):
        raise ValueError("24-hour forecast lookahead audit failed")
    actual = forecasts["actual_next_24h_mean_bubble_usd"].to_numpy(dtype=float)
    predicted = forecasts["predicted_next_24h_mean_bubble_usd"].to_numpy(dtype=float)
    summary = {
        "status": "ok" if len(forecasts) else "insufficient_history",
        "forecast_horizon_hours": 24,
        "forecast_samples": int(len(forecasts)),
        "median_actual_next_24h_mean_usd": float(np.median(actual)) if len(actual) else None,
        "median_predicted_next_24h_mean_usd": float(np.median(predicted)) if len(predicted) else None,
        "mean_absolute_error_usd": float(np.mean(np.abs(predicted - actual))) if len(actual) else None,
        "median_absolute_percentage_error": float(np.median(np.abs(predicted - actual) / actual)) if len(actual) else None,
        "correlation": float(np.corrcoef(predicted, actual)[0, 1]) if len(actual) > 1 else None,
        "lookahead_violations": int(forecasts["lookahead_violation"].sum()) if len(forecasts) else 0,
        "causality_audit_passed": bool(len(forecasts) and not forecasts["lookahead_violation"].any()),
        "data_scope": "daily qualifying-bubble samples from the supplied event file",
    }
    return forecasts, summary


def write_24_hour_chart(path: str | Path, forecasts: pd.DataFrame) -> Path:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(forecasts["issue_timestamp"], unit="ms", utc=True)
    actual_dates = pd.to_datetime(forecasts["target_end_timestamp"], unit="ms", utc=True)
    fig, price_axis = plt.subplots(figsize=(15, 7.5))
    bubble_axis = price_axis.twinx()
    price_axis.plot(dates, forecasts["price_at_forecast"], color="#2563eb", linewidth=2.0, label="BTC price")
    bubble_axis.plot(dates, forecasts["predicted_next_24h_mean_bubble_usd"] / 1_000_000, color="#f59e0b", linewidth=2.2, label="Predicted next-24h average")
    bubble_axis.plot(actual_dates, forecasts["actual_next_24h_mean_bubble_usd"] / 1_000_000, color="#10b981", linewidth=2.2, label="Actual average at target end (+24h)")
    price_axis.set_title("BTC one-day-ahead bubble forecast (actual shifted +24h)")
    price_axis.set_xlabel("Forecast issue date (UTC)")
    price_axis.set_ylabel("BTC price (USDT)", color="#2563eb")
    bubble_axis.set_ylabel("Average qualifying bubble notional (USD millions)")
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


def create_24_hour_forecast_report(events_csv: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    forecasts, summary = build_24_hour_forecasts(pd.read_csv(events_csv))
    forecast_path = output / "24-hour-ahead-forecasts.csv"
    summary_path = output / "24-hour-ahead-summary.json"
    chart_path = output / "24-hour-ahead-chart.png"
    forecasts.to_csv(forecast_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_24_hour_chart(chart_path, forecasts)
    return {"summary": summary, "chart": str(chart_path), "forecasts_csv": str(forecast_path)}
