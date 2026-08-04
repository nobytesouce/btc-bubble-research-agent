from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ResearchConfig


def add_causal_features(frame: pd.DataFrame, vwap_seconds: int = 60) -> pd.DataFrame:
    data = frame.sort_values("timestamp").reset_index(drop=True).copy()
    ts = data["timestamp"].to_numpy(dtype=np.int64)
    qty = data["qty"].to_numpy(dtype=np.float64)
    price = data["price"].to_numpy(dtype=np.float64)
    q_usd = data["q_usd"].to_numpy(dtype=np.float64) if "q_usd" in data else price * qty
    signed = q_usd * np.where(data["side"].to_numpy() == "buy", 1.0, -1.0)
    q_prefix = np.concatenate(([0.0], np.cumsum(q_usd)))
    base_prefix = np.concatenate(([0.0], np.cumsum(qty)))
    pq_prefix = np.concatenate(([0.0], np.cumsum(q_usd)))
    cvd_prefix = np.concatenate(([0.0], np.cumsum(signed)))
    left = 0
    volume = np.zeros(len(data), dtype=np.float64)
    vwap = np.full(len(data), np.nan, dtype=np.float64)
    cvd = np.zeros(len(data), dtype=np.float64)
    window_ms = int(vwap_seconds * 1000)
    for i in range(len(data)):
        while left < i and ts[left] < ts[i] - window_ms:
            left += 1
        volume[i] = q_prefix[i] - q_prefix[left]
        notional = pq_prefix[i] - pq_prefix[left]
        base_volume = base_prefix[i] - base_prefix[left]
        if base_volume > 0:
            vwap[i] = notional / base_volume
        cvd[i] = cvd_prefix[i] - cvd_prefix[left]
    log_price = np.log(price)
    returns = np.diff(log_price, prepend=log_price[0])
    volatility = pd.Series(returns).rolling(500, min_periods=50).std().shift(1).to_numpy()
    data["q_usd"] = q_usd
    data["volume_60s_usd"] = volume
    data["volume_60s"] = volume
    data["frozen_vwap"] = vwap
    data["cvd_60s"] = cvd
    data["volatility"] = volatility
    data["log_q"] = np.log(np.maximum(q_usd, np.finfo(float).tiny))
    data["r_volume"] = np.divide(q_usd, volume, out=np.full_like(q_usd, np.nan), where=volume > 0)
    if "depth_opp_10bp" in data:
        depth = data["depth_opp_10bp"].to_numpy(dtype=np.float64)
        data["r_depth"] = np.divide(q_usd, depth, out=np.full_like(q_usd, np.nan), where=depth > 0)
    else:
        data["r_depth"] = np.nan
    dt = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
    data["hour"] = dt.dt.hour.astype("int8")
    return data


@dataclass
class ConditionalPercentiles:
    min_group_rows: int = 250

    def fit(self, frame: pd.DataFrame) -> "ConditionalPercentiles":
        training = frame.copy()
        finite_vol = training["volatility"].dropna()
        self.vol_edges = finite_vol.quantile([1 / 3, 2 / 3]).to_numpy() if len(finite_vol) else np.array([0.0, 0.0])
        training["vol_regime"] = self._vol_regime(training["volatility"])
        self.tables: dict[tuple[str, tuple], np.ndarray] = {}
        self.group_levels = [
            ("exchange", "product", "hour", "side", "vol_regime"),
            ("exchange", "product", "side", "vol_regime"),
            ("exchange", "product", "side"),
            ("exchange", "product"),
        ]
        for feature in ("log_q", "r_volume", "r_depth"):
            valid = training[np.isfinite(training[feature])]
            for columns in self.group_levels:
                for key, group in valid.groupby(list(columns), observed=True, sort=False):
                    key_tuple = key if isinstance(key, tuple) else (key,)
                    values = np.sort(group[feature].to_numpy(dtype=np.float64))
                    if len(values) >= self.min_group_rows or columns == self.group_levels[-1]:
                        self.tables[(feature + "|" + ",".join(columns), key_tuple)] = values
        return self

    def _vol_regime(self, series: pd.Series) -> pd.Series:
        values = series.to_numpy(dtype=np.float64)
        regimes = np.where(values <= self.vol_edges[0], "low", np.where(values <= self.vol_edges[1], "mid", "high"))
        regimes[~np.isfinite(values)] = "unknown"
        return pd.Series(regimes, index=series.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        scored["vol_regime"] = self._vol_regime(scored["volatility"])
        mappings = {"log_q": "p_size", "r_volume": "p_volume", "r_depth": "p_depth"}
        for feature, output in mappings.items():
            percentiles = np.full(len(scored), np.nan)
            fallback = np.full(len(scored), -1, dtype=np.int8)
            for i, row in enumerate(scored.itertuples(index=False)):
                value = getattr(row, feature)
                if not np.isfinite(value):
                    continue
                row_map = row._asdict()
                for level, columns in enumerate(self.group_levels):
                    key = tuple(row_map[column] for column in columns)
                    values = self.tables.get((feature + "|" + ",".join(columns), key))
                    if values is not None and len(values):
                        percentiles[i] = np.searchsorted(values, value, side="right") / len(values)
                        fallback[i] = level
                        break
            scored[output] = percentiles
            scored[output + "_fallback"] = fallback
        return scored


def add_signal_columns(frame: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    data = frame.copy()
    p = config.percentile
    volume_or_depth = (data["p_volume"] >= p.volume_gate) | (data["p_depth"] >= p.depth_gate)
    data["big_bubble"] = (data["p_size"] >= p.very_big) & volume_or_depth
    data["size_class"] = np.select(
        [data["p_size"] >= p.exceptional, data["p_size"] >= p.very_big, data["p_size"] >= p.big],
        ["exceptional", "very_big", "big"],
        default="normal",
    )
    has_depth = np.isfinite(data["p_depth"])
    data["bbs"] = np.where(
        has_depth,
        100.0 * (0.40 * data["p_size"] + 0.30 * data["p_volume"] + 0.30 * data["p_depth"]),
        np.nan,
    )
    data["depth_exact"] = has_depth
    return data


def cluster_signal_events(frame: pd.DataFrame, cluster_ms: int, signal_column: str = "big_bubble") -> pd.DataFrame:
    signals = frame[frame[signal_column]].copy()
    if signals.empty:
        return signals
    new_event = (
        signals["side"].ne(signals["side"].shift())
        | signals["timestamp"].sub(signals["timestamp"].shift()).gt(cluster_ms)
    )
    signals["event_id"] = new_event.cumsum().astype("int64")
    idx = signals.groupby("event_id", sort=False)["qty"].idxmax()
    events = signals.loc[idx].sort_values("timestamp").reset_index(drop=True)
    event_qty = signals.groupby("event_id", sort=False)["qty"].sum()
    event_notional = signals.groupby("event_id", sort=False)["q_usd"].sum()
    events["cluster_qty"] = events["event_id"].map(event_qty)
    events["cluster_q_usd"] = events["event_id"].map(event_notional)
    events["rolling_median_bubble_usd"] = events["cluster_q_usd"].expanding().median()
    return events
