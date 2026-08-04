from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass(frozen=True)
class PercentileConfig:
    min_group_rows: int = 250
    big: float = 0.99
    very_big: float = 0.995
    exceptional: float = 0.999
    volume_gate: float = 0.90
    depth_gate: float = 0.90


@dataclass(frozen=True)
class EventConfig:
    cluster_ms: int = 1_000
    frozen_vwap_seconds: int = 60
    horizons_seconds: tuple[int, ...] = (5, 15, 60, 300)


@dataclass(frozen=True)
class ExecutionConfig:
    taker_fee_bps_each_side: float = 5.0
    slippage_bps_each_side: float = 1.0


@dataclass(frozen=True)
class ResearchConfig:
    exchange: str = "binance"
    product: str = "usd-m-perpetual"
    symbol: str = "BTCUSDT"
    percentile: PercentileConfig = field(default_factory=PercentileConfig)
    event: EventConfig = field(default_factory=EventConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def load(cls, path: str | Path) -> "ResearchConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            exchange=payload.get("exchange", "binance"),
            product=payload.get("product", "usd-m-perpetual"),
            symbol=payload.get("symbol", "BTCUSDT"),
            percentile=PercentileConfig(**payload.get("percentile", {})),
            event=EventConfig(
                **{
                    **payload.get("event", {}),
                    "horizons_seconds": tuple(payload.get("event", {}).get("horizons_seconds", (5, 15, 60, 300))),
                }
            ),
            execution=ExecutionConfig(**payload.get("execution", {})),
        )

