from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

import pandas as pd

from .safety import assert_read_only_url


@dataclass(frozen=True)
class DownloadResult:
    frame: pd.DataFrame
    url: str
    sha256: str


def _download_zip_csv(url: str, max_rows: int | None = None) -> tuple[pd.DataFrame, str]:
    assert_read_only_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "btc-bubble-research/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_name) as stream:
            frame = pd.read_csv(stream, nrows=max_rows)
    return frame, digest


def binance_aggtrades_url(symbol: str, date: str) -> str:
    symbol = symbol.upper()
    return (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{date}.zip"
    )


def download_binance_aggtrades(symbol: str, date: str, max_rows: int | None = None) -> DownloadResult:
    url = binance_aggtrades_url(symbol, date)
    frame, digest = _download_zip_csv(url, max_rows=max_rows)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "agg_trade_id" not in frame.columns:
        frame.columns = [
            "agg_trade_id", "price", "qty", "first_trade_id", "last_trade_id", "timestamp", "buyer_maker"
        ][: len(frame.columns)]
    frame = frame.rename(
        columns={
            "transact_time": "timestamp",
            "is_buyer_maker": "buyer_maker",
            "quantity": "qty",
        }
    )
    required = {"price", "qty", "timestamp", "buyer_maker"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Unexpected Binance schema; missing {sorted(missing)}")
    frame = frame.assign(
        timestamp=pd.to_numeric(frame["timestamp"], errors="raise").astype("int64"),
        price=pd.to_numeric(frame["price"], errors="raise").astype("float64"),
        qty=pd.to_numeric(frame["qty"], errors="raise").astype("float64"),
        side=frame["buyer_maker"].map({False: "buy", True: "sell", "False": "buy", "True": "sell"}),
        exchange="binance",
        product="usd-m-perpetual",
        symbol=symbol.upper(),
    )
    if frame["side"].isna().any():
        normalized = frame["buyer_maker"].astype(str).str.lower()
        frame["side"] = normalized.map({"false": "buy", "true": "sell"})
    return DownloadResult(frame.sort_values("timestamp").reset_index(drop=True), url, digest)


def download_binance_metrics(symbol: str, date: str) -> DownloadResult:
    symbol = symbol.upper()
    url = (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        f"{symbol}/{symbol}-metrics-{date}.zip"
    )
    frame, digest = _download_zip_csv(url)
    frame["timestamp"] = pd.to_datetime(frame["create_time"], utc=True).astype("int64") // 1_000_000
    frame["open_interest"] = pd.to_numeric(frame["sum_open_interest"], errors="coerce")
    frame["open_interest_value"] = pd.to_numeric(frame["sum_open_interest_value"], errors="coerce")
    keep = ["timestamp", "open_interest", "open_interest_value"]
    return DownloadResult(frame[keep].sort_values("timestamp").drop_duplicates("timestamp"), url, digest)


def attach_asof_market_context(trades: pd.DataFrame, metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    data = trades.sort_values("timestamp").copy()
    if metrics is not None and not metrics.empty:
        data = pd.merge_asof(
            data,
            metrics.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=10 * 60 * 1000,
        )
    return data


def write_manifest(path: str | Path, result: DownloadResult) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"source_url": result.url, "sha256": result.sha256, "rows": len(result.frame)}, indent=2),
        encoding="utf-8",
    )
