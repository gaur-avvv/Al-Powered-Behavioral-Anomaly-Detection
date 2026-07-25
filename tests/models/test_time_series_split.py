"""
Unit tests for AdvancedTimeSeriesSplit strategies.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.splitters.time_series_split import AdvancedTimeSeriesSplit


class TestAdvancedTimeSeriesSplit:
    """Test suite covering time series cross-validation splitters."""

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create sample time series DataFrame."""
        timestamps = pd.date_range("2024-01-01", periods=100, freq="h")
        return pd.DataFrame({
            "timestamp": timestamps,
            "val": np.random.randn(100)
        })

    def test_expanding_window_split(self, sample_dataframe: pd.DataFrame):
        """Test expanding window cross-validation."""
        splitter = AdvancedTimeSeriesSplit(n_splits=3)
        splits = splitter.expanding_window_split(sample_dataframe, time_col="timestamp")

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert max(train_idx) < min(test_idx)

    def test_rolling_window_split(self, sample_dataframe: pd.DataFrame):
        """Test rolling window cross-validation."""
        splitter = AdvancedTimeSeriesSplit(n_splits=3)
        splits = splitter.rolling_window_split(sample_dataframe, time_col="timestamp")

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
