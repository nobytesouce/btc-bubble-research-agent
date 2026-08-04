from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from btc_bubble.config import ResearchConfig
from btc_bubble.data import reconstruct_market_orders
from btc_bubble.features import ConditionalPercentiles, add_causal_features, add_signal_columns
from btc_bubble.forecast import aggregate_forecast_frames, forecast_bubble_sizes
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
        expected = featured.loc[0, "price"] * featured.loc[0, "qty"]
        self.assertAlmostEqual(featured.loc[1, "volume_60s_usd"], expected)

    def test_reconstructs_fragmented_notional(self):
        raw = self.synthetic(3)
        raw["agg_trade_id"] = [1, 2, 3]
        raw["first_trade_id"] = [10, 11, 12]
        raw["last_trade_id"] = [10, 11, 12]
        raw.loc[1, ["timestamp", "side"]] = raw.loc[0, ["timestamp", "side"]]
        reconstructed = reconstruct_market_orders(raw)
        self.assertEqual(len(reconstructed), 2)
        self.assertEqual(int(reconstructed.loc[0, "fragment_count"]), 2)
        expected = float((raw.loc[:1, "price"] * raw.loc[:1, "qty"]).sum())
        self.assertAlmostEqual(float(reconstructed.loc[0, "q_usd"]), expected)

    def test_percentile_model_scores_larger_trade_higher(self):
        featured = add_causal_features(self.synthetic())
        model = ConditionalPercentiles(min_group_rows=10).fit(featured.iloc[:800])
        comparison = featured.iloc[[500, 500]].copy().reset_index(drop=True)
        comparison["log_q"] = np.log([1_000.0, 1_000_000.0])
        scored = add_signal_columns(model.transform(comparison), ResearchConfig())
        self.assertGreater(scored.iloc[1]["p_size"], scored.iloc[0]["p_size"])

    def test_missing_depth_is_explicit(self):
        featured = add_causal_features(self.synthetic())
        model = ConditionalPercentiles(min_group_rows=10).fit(featured.iloc[:800])
        scored = add_signal_columns(model.transform(featured.iloc[800:]), ResearchConfig())
        self.assertTrue(scored["bbs"].isna().all())
        self.assertFalse(scored["depth_exact"].any())

    def test_bubble_forecast_uses_only_prior_events(self):
        events = pd.DataFrame({
            "timestamp": np.arange(7, dtype=np.int64) * 1000,
            "side": "buy",
            "hour": 1,
            "vol_regime": "mid",
            "cluster_q_usd": [100.0, 200.0, 300.0, 400.0, 500.0, 10_000.0, 20_000.0],
        })
        forecast = forecast_bubble_sizes(events, window=30, minimum_history=5)
        self.assertTrue(np.isnan(forecast.loc[4, "predicted_bubble_usd"]))
        self.assertEqual(float(forecast.loc[5, "predicted_bubble_usd"]), 300.0)
        self.assertEqual(float(forecast.loc[6, "predicted_bubble_usd"]), 350.0)

    def test_aggregate_forecasts_reports_accuracy(self):
        frame = pd.DataFrame({
            "timestamp": [1_704_067_200_000, 1_704_153_600_000],
            "price": [42_000.0, 43_000.0],
            "predicted_bubble_usd": [1_000_000.0, 2_000_000.0],
            "actual_bubble_usd": [2_000_000.0, 2_000_000.0],
        })
        _, daily, summary = aggregate_forecast_frames([frame])
        self.assertEqual(summary["days"], 2)
        self.assertEqual(summary["evaluated_predictions"], 2)
        self.assertAlmostEqual(summary["median_absolute_percentage_error"], 0.25)
        self.assertEqual(len(daily), 2)


class SafetyTests(unittest.TestCase):
    def test_public_data_allowed(self):
        assert_read_only_url("https://data.binance.vision/data/test.zip")

    def test_trading_endpoint_forbidden(self):
        with self.assertRaises(PermissionError):
            assert_read_only_url("https://api.hyperliquid.xyz/exchange")


if __name__ == "__main__":
    unittest.main()
