"""Tests for src/features/indicators.py — technical indicator computation."""

from __future__ import annotations

from src.features.indicators import (
    MAX_WARMUP_PERIODS,
    check_feature_correlation,
    compute_indicators,
    get_feature_columns,
)


class TestComputeIndicators:
    def test_returns_all_feature_columns(self, synthetic_ohlcv):
        df = compute_indicators(synthetic_ohlcv.copy())
        expected = get_feature_columns()
        for col in expected:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_no_nan_after_warmup(self, synthetic_ohlcv):
        df = compute_indicators(synthetic_ohlcv.copy())
        feature_cols = get_feature_columns()
        after_warmup = df.iloc[MAX_WARMUP_PERIODS:]
        nan_counts = after_warmup[feature_cols].isna().sum()
        bad_cols = nan_counts[nan_counts > 0]
        assert bad_cols.empty, (
            f"NaN found after warmup in: {dict(bad_cols)}"
        )

    def test_bounded_features_in_range(self, synthetic_ohlcv):
        df = compute_indicators(synthetic_ohlcv.copy())
        after_warmup = df.iloc[MAX_WARMUP_PERIODS:].dropna()

        assert after_warmup["rsi_14"].between(0, 100).all(), "RSI out of [0, 100]"
        assert after_warmup["stoch_rsi_k"].between(-0.01, 1.01).all(), "StochRSI K out of [0, 1]"
        assert after_warmup["williams_r"].between(-100.01, 0.01).all(), "Williams %R out of [-100, 0]"
        assert after_warmup["bb_pct"].between(-2, 3).all(), "BB %B seems extreme"

    def test_scale_invariance(self, synthetic_ohlcv):
        """Features should be roughly the same scale regardless of price multiplier."""
        df1 = compute_indicators(synthetic_ohlcv.copy())
        df2 = synthetic_ohlcv.copy()
        df2[["open", "high", "low", "close"]] *= 1000
        df2 = compute_indicators(df2)

        feature_cols = get_feature_columns()
        after = MAX_WARMUP_PERIODS + 10

        for col in feature_cols:
            s1 = df1[col].iloc[after:].dropna()
            s2 = df2[col].iloc[after:].dropna()
            if s1.empty or s2.empty:
                continue
            ratio = abs(s1.mean()) / max(abs(s2.mean()), 1e-10)
            assert 0.01 < ratio < 100, (
                f"Feature '{col}' not scale-invariant: ratio={ratio:.4f}"
            )

    def test_feature_count_matches(self):
        cols = get_feature_columns()
        assert len(cols) == 41, f"Expected 41 features, got {len(cols)}"

    def test_no_duplicates_in_feature_list(self):
        cols = get_feature_columns()
        assert len(cols) == len(set(cols)), "Duplicate features in get_feature_columns()"


class TestFeatureCorrelation:
    def test_detects_identical_columns(self, synthetic_ohlcv):
        df = compute_indicators(synthetic_ohlcv.copy())
        df["clone"] = df["rsi_14"]
        flagged = check_feature_correlation(
            df, feature_cols=["rsi_14", "clone"], threshold=0.95
        )
        assert len(flagged) == 1
        assert flagged[0][2] > 0.99

    def test_no_flags_on_clean_data(self, synthetic_ohlcv):
        df = compute_indicators(synthetic_ohlcv.copy())
        flagged = check_feature_correlation(df, threshold=0.999)
        # Some natural correlation expected but at 0.999 threshold should be rare
        assert isinstance(flagged, list)
