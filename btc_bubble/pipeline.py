from __future__ import annotations

from pathlib import Path
import json
from datetime import date as calendar_date, timedelta

from .backtest import backtest_events, performance_summary
from .config import ResearchConfig
from .data import (
    attach_asof_market_context,
    download_binance_aggtrades,
    download_binance_metrics,
    reconstruct_market_orders,
    write_manifest,
)
from .features import ConditionalPercentiles, add_causal_features, add_signal_columns, cluster_signal_events
from .forecast import build_two_hour_samples, forecast_bubble_sizes, forecast_summary, write_forecast_chart
from .optimize import walk_forward_search
from .report import write_report
from .storage import upload_derived_artifact


def run_demo(date: str, max_rows: int, output_dir: str | Path, config_path: str | Path) -> dict:
    config = ResearchConfig.load(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    downloaded = download_binance_aggtrades(config.symbol, date, max_rows=max_rows)
    reconstructed = reconstruct_market_orders(downloaded.frame)
    try:
        metrics = download_binance_metrics(config.symbol, date)
        source_frame = attach_asof_market_context(reconstructed, metrics.frame)
    except Exception:
        source_frame = reconstructed
    write_manifest(output / f"manifest-{date}.json", downloaded)
    featured = add_causal_features(source_frame, config.event.frozen_vwap_seconds)
    split = max(1, int(len(featured) * 0.70))
    training = featured.iloc[:split].copy()
    evaluation = featured.iloc[split:].copy()
    model = ConditionalPercentiles(config.percentile.min_group_rows).fit(training)
    scored = add_signal_columns(model.transform(evaluation), config)
    events = cluster_signal_events(scored, config.event.cluster_ms)
    results = backtest_events(scored, events, config)
    scored["research_candidate"] = (scored["p_size"] >= 0.99) & (
        (scored["p_volume"] >= 0.85) | (scored["p_depth"] >= 0.85)
    )
    candidate_events = cluster_signal_events(scored, config.event.cluster_ms, "research_candidate")
    candidate_results = backtest_events(scored, candidate_events, config)
    horizon = config.event.horizons_seconds[-1]
    continuation = performance_summary(results, f"continuation_pnl_{horizon}s")
    reversal = performance_summary(results, f"reversal_pnl_{horizon}s")
    optimization = walk_forward_search(candidate_results, config.event.horizons_seconds)
    results_path = output / f"events-{date}.csv"
    results.to_csv(results_path, index=False)
    summary = {
        "input_rows": len(downloaded.frame),
        "reconstructed_market_orders": len(reconstructed),
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "detected_events": len(results),
        "median_detected_bubble_usd": float(events["cluster_q_usd"].median()) if not events.empty else None,
        "events_with_exact_depth": int(results["depth_exact"].sum()) if not results.empty else 0,
        "continuation": continuation,
        "reversal": reversal,
    }
    summary_path = output / f"summary-{date}.json"
    summary_path.write_text(json.dumps({"summary": summary, "optimization": optimization}, indent=2), encoding="utf-8")
    report_path = write_report(
        output / f"btc-bubble-{date}.html",
        {"exchange": config.exchange, "product": config.product, "date": date},
        summary,
        optimization,
        results,
    )
    upload_derived_artifact(summary_path, f"summaries/{summary_path.name}")
    upload_derived_artifact(results_path, f"events/{results_path.name}")
    return {"summary": summary, "optimization": optimization, "report": str(report_path)}


def run_forecast(date: str, max_rows: int, output_dir: str | Path, config_path: str | Path) -> dict:
    """Generate a no-trading next-bubble forecast and comparison chart."""
    config = ResearchConfig.load(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    downloaded = download_binance_aggtrades(config.symbol, date, max_rows=max_rows)
    reconstructed = reconstruct_market_orders(downloaded.frame)
    try:
        metrics = download_binance_metrics(config.symbol, date)
        source_frame = attach_asof_market_context(reconstructed, metrics.frame)
    except Exception:
        source_frame = reconstructed
    write_manifest(output / f"manifest-{date}.json", downloaded)
    featured = add_causal_features(source_frame, config.event.frozen_vwap_seconds)
    split = max(1, int(len(featured) * 0.70))
    model = ConditionalPercentiles(config.percentile.min_group_rows).fit(featured.iloc[:split].copy())
    scored = add_signal_columns(model.transform(featured.iloc[split:].copy()), config)
    events = cluster_signal_events(scored, config.event.cluster_ms)
    forecasts = forecast_bubble_sizes(events)
    summary, future = forecast_summary(forecasts)
    chart_path = write_forecast_chart(output / f"bubble-forecast-{date}.png", forecasts, future)
    forecast_path = output / f"bubble-forecast-{date}.csv"
    future_path = output / f"next-five-bubbles-{date}.csv"
    summary_path = output / f"forecast-summary-{date}.json"
    forecasts.to_csv(forecast_path, index=False)
    future.to_csv(future_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    upload_derived_artifact(summary_path, f"forecasts/{summary_path.name}")
    upload_derived_artifact(forecast_path, f"forecasts/{forecast_path.name}")
    return {
        "summary": summary,
        "chart": str(chart_path),
        "forecast_csv": str(forecast_path),
        "next_five_csv": str(future_path),
    }


def run_two_hour_samples(date: str, max_rows: int, output_dir: str | Path, config_path: str | Path) -> dict:
    """Create strictly forward two-hour target windows for later aggregation."""
    config = ResearchConfig.load(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    calibration_date = (calendar_date.fromisoformat(date) - timedelta(days=1)).isoformat()
    calibration_download = download_binance_aggtrades(config.symbol, calibration_date, max_rows=max_rows)
    calibration_frame = reconstruct_market_orders(calibration_download.frame)
    calibration_featured = add_causal_features(calibration_frame, config.event.frozen_vwap_seconds)
    downloaded = download_binance_aggtrades(config.symbol, date, max_rows=max_rows)
    reconstructed = reconstruct_market_orders(downloaded.frame)
    try:
        metrics = download_binance_metrics(config.symbol, date)
        source_frame = attach_asof_market_context(reconstructed, metrics.frame)
    except Exception:
        source_frame = reconstructed
    featured = add_causal_features(source_frame, config.event.frozen_vwap_seconds)
    calibration_max_timestamp = int(calibration_featured["timestamp"].max())
    current_min_timestamp = int(featured["timestamp"].min())
    if calibration_max_timestamp >= current_min_timestamp:
        raise ValueError("Calibration data must end before the forecast day starts")
    model = ConditionalPercentiles(config.percentile.min_group_rows).fit(calibration_featured)
    scored = add_signal_columns(model.transform(featured.copy()), config)
    bubbles = cluster_signal_events(scored, config.event.cluster_ms)
    samples = build_two_hour_samples(
        scored,
        bubbles,
        calibration_max_timestamp=calibration_max_timestamp,
    )
    samples_path = output / f"two-hour-samples-{date}.csv"
    samples.to_csv(samples_path, index=False)
    return {
        "date": date,
        "calibration_date": calibration_date,
        "calibration_max_timestamp": calibration_max_timestamp,
        "forecast_day_min_timestamp": current_min_timestamp,
        "market_rows": len(scored),
        "qualifying_bubbles": len(bubbles),
        "samples": len(samples),
        "lookahead_violations": int(samples["lookahead_violation"].sum()) if len(samples) else 0,
        "samples_csv": str(samples_path),
    }
