from __future__ import annotations

import argparse
import json

from .pipeline import run_demo, run_forecast


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        result = run_demo(args.date, args.max_rows, args.output_dir, args.config)
        print(json.dumps(result, indent=2))
    elif args.command == "forecast":
        result = run_forecast(args.date, args.max_rows, args.output_dir, args.config)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
