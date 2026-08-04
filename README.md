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
python -m btc_bubble demo --date 2024-01-01 --max-rows 120000
python -m unittest discover -s tests -v
```

The demo downloads one official Binance BTCUSDT USD-M aggregate-trade file, fits percentile distributions on the first 70% of rows, detects events in the remaining 30%, backtests them, and writes an HTML report under `reports/`.

## Cloud operation

- `.github/workflows/daily.yml`: catch-up-safe daily research run.
- `.github/workflows/weekly.yml`: larger weekly search.
- Derived artifacts can be uploaded to a private Hugging Face dataset by setting `HF_TOKEN` and `HF_REPO_ID` GitHub secrets.
- Raw exchange archives are never uploaded.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for coverage and validation rules.

