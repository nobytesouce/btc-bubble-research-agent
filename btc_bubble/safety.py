from __future__ import annotations

from urllib.parse import urlparse


READ_ONLY_HOSTS = {
    "data.binance.vision",
    "api.hyperliquid.xyz",
    "api.bybit.com",
    "www.okx.com",
}

FORBIDDEN_TOKENS = (
    "/exchange",
    "order/create",
    "place-order",
    "withdraw",
    "private-key",
    "wallet-seed",
)


def assert_read_only_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS market-data URLs are permitted")
    if parsed.hostname not in READ_ONLY_HOSTS:
        raise ValueError(f"Unapproved market-data host: {parsed.hostname}")
    lowered = url.lower()
    if any(token in lowered for token in FORBIDDEN_TOKENS):
        raise PermissionError("Trading, wallet, and withdrawal endpoints are forbidden")

