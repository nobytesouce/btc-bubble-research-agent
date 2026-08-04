# BTC Bubble Research Agent

Cloud-first, read-only research system for detecting and backtesting unusually large BTC trades.

## Safety

- No order placement, signing, wallet, withdrawal, or private exchange endpoints.
- Hyperliquid support is public read-only only.
- Missing depth or liquidation data is reported as missing, never silently approximated as exact.
- The optimizer cannot access a sealed final holdout during candidate selection.

## Quick start

```bash
python -m pip install -e .
python -m btc_bubble forecast --date 2024-01-01 --max-rows 500000
python -m btc_bubble demo --date 2024-01-01 --max-rows 120000
python -m unittest discover -s tests -v
```

The demo downloads one official Binance BTCUSDT USD-M aggregate-trade file, reconstructs same-millisecond/same-side fragments, converts event size and trailing volume to USD notional, fits conditional percentile distributions on the first 70% of events, detects bubbles in the remaining 30%, backtests them, and writes an HTML report under `reports/`.

Binance's free archive does not expose taker order IDs across price levels, so fragmented market-order reconstruction is an auditable proxy rather than an exact order-ID match. Exact `P_depth`/BBS values are emitted only when synchronized 10 bp order-book depth is supplied.

The `forecast` command is the default cloud task. It produces a three-line PNG chart containing BTC price, the prediction made before each qualifying bubble, and the actual qualifying bubble notional. It also appends five dashed size/interval projections. Predictions use only earlier qualifying events and never place trades.

## Cloud operation

- `.github/workflows/daily.yml`: catch-up-safe daily research run.
- `.github/workflows/weekly.yml`: larger weekly search.
- Derived artifacts can be uploaded to a private Hugging Face dataset by setting `HF_TOKEN` and `HF_REPO_ID` GitHub secrets.
- Raw exchange archives are never uploaded.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for coverage and validation rules.
