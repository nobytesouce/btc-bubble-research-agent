from __future__ import annotations

import argparse
import json

from .forecast import aggregate_forecast_directory, aggregate_two_hour_directory, create_24_hour_forecast_report
from .pipeline import run_demo, run_forecast, run_two_hour_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only BTC bubble research agent")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a no-lookahead official-data demonstration")
    demo.add_argument("--date", default="2024-01-01")
    demo.add_argument("--max-rows", type=int, default=120_000)
    demo.add_argument("--output-dir", default="reports")
    demo.add_argument("--config", default="configs/default.json")
    forecast = sub.add_parser("forecast", help="Predict next qualifying bubble sizes and create a comparison chart")
    forecast.add_argument("--date", default="2024-01-01")
    forecast.add_argument("--max-rows", type=int, default=500_000)
    forecast.add_argument("--output-dir", default="reports")
    forecast.add_argument("--config", default="configs/default.json")
    aggregate = sub.add_parser("aggregate", help="Combine many daily forecasts into one evaluation chart")
    aggregate.add_argument("--input-dir", required=True)
    aggregate.add_argument("--output-dir", default="combined-report")
    two_hour = sub.add_parser("forecast-2h", help="Create strictly forward two-hour average-bubble samples")
    two_hour.add_argument("--date", default="2024-01-01")
    two_hour.add_argument("--max-rows", type=int, default=1_000_000)
    two_hour.add_argument("--output-dir", default="reports")
    two_hour.add_argument("--config", default="configs/default.json")
    aggregate_two_hour = sub.add_parser("aggregate-2h", help="Combine and score two-hour-ahead samples")
    aggregate_two_hour.add_argument("--input-dir", required=True)
    aggregate_two_hour.add_argument("--output-dir", default="combined-report")
    twenty_four = sub.add_parser("forecast-24h", help="Predict each next UTC day's average bubble size")
    twenty_four.add_argument("--events-csv", required=True)
    twenty_four.add_argument("--output-dir", default="combined-report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        result = run_demo(args.date, args.max_rows, args.output_dir, args.config)
        print(json.dumps(result, indent=2))
    elif args.command == "forecast":
        result = run_forecast(args.date, args.max_rows, args.output_dir, args.config)
        print(json.dumps(result, indent=2))
    elif args.command == "aggregate":
        result = aggregate_forecast_directory(args.input_dir, args.output_dir)
        print(json.dumps(result, indent=2))
    elif args.command == "forecast-2h":
        result = run_two_hour_samples(args.date, args.max_rows, args.output_dir, args.config)
        print(json.dumps(result, indent=2))
    elif args.command == "aggregate-2h":
        result = aggregate_two_hour_directory(args.input_dir, args.output_dir)
        print(json.dumps(result, indent=2))
    elif args.command == "forecast-24h":
        result = create_24_hour_forecast_report(args.events_csv, args.output_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
