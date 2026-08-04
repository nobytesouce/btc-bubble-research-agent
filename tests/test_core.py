from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from btc_bubble.config import ResearchConfig
from btc_bubble.features import ConditionalPercentiles, add_causal_features, add_signal_columns
from btc_bubble.safety import assert_read_only_url


class FeatureTests(unittest.TestCase):
    def synthetic(self, n: int = 1000) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": np.arange(n, dtype=np.int64) * 1000,
            "price": 50_000 + np.sin(np.arange(n) / 30),
            "qty": np.linspace(0.01, 10.0, n),
            "side": np.where(np.arange(n) % 2 == 0, "buy", "sell"),
            "exchange": "binance",
            "product": "usd-m-perpetual",
            "symbol": "BTCUSDT",
        })

    def test_causal_volume_excludes_current_trade(self):
        featured = add_causal_features(self.synthetic(100))
        self.assertEqual(featured.loc[0, "volume_60s"], 0.0)
        self.assertAlmostEqual(featured.loc[1, "volume_60s"], featured.loc[0, "qty"])

    def test_percentile_model_scores_larger_trade_higher(self):
        featured = add_causal_features(self.synthetic())
        model = ConditionalPercentiles(min_group_rows=10).fit(featured.iloc[:800])
        comparison = featured.iloc[[500, 500]].copy().reset_index(drop=True)
        comparison["log_q"] = np.log([0.1, 100.0])
        scored = add_signal_columns(model.transform(comparison), ResearchConfig())
        self.assertGreater(scored.iloc[1]["p_size"], scored.iloc[0]["p_size"])

    def test_missing_depth_is_explicit(self):
        featured = add_causal_features(self.synthetic())
        model = ConditionalPercentiles(min_group_rows=10).fit(featured.iloc[:800])
        scored = add_signal_columns(model.transform(featured.iloc[800:]), ResearchConfig())
        self.assertTrue(scored["bbs"].isna().all())
        self.assertFalse(scored["depth_exact"].any())


class SafetyTests(unittest.TestCase):
    def test_public_data_allowed(self):
        assert_read_only_url("https://data.binance.vision/data/test.zip")

    def test_trading_endpoint_forbidden(self):
        with self.assertRaises(PermissionError):
            assert_read_only_url("https://api.hyperliquid.xyz/exchange")


if __name__ == "__main__":
    unittest.main()
