from __future__ import annotations

from pathlib import Path
import json

from .backtest import backtest_events, performance_summary
from .config import ResearchConfig
from .data import attach_asof_market_context, download_binance_aggtrades, download_binance_metrics, write_manifest
from .features import ConditionalPercentiles, add_causal_features, add_signal_columns, cluster_signal_events
from .optimize import walk_forward_search
from .report import write_report
from .storage import upload_derived_artifact


def run_demo(date: str, max_rows: int, output_dir: str | Path, config_path: str | Path) -> dict:
    config = ResearchConfig.load(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    downloaded = download_binance_aggtrades(config.symbol, date, max_rows=max_rows)
    try:
        metrics = download_binance_metrics(config.symbol, date)
        source_frame = attach_asof_market_context(downloaded.frame, metrics.frame)
    except Exception:
        source_frame = downloaded.frame
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
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "detected_events": len(results),
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
