"""
Unit tests for EntityBaselineProfiler.
"""

import pytest
import numpy as np
from src.models.baseline_profiler import EntityBaselineProfiler


class TestEntityBaselineProfiler:
    """Test suite covering baseline profiler creation and incremental updates."""

    def test_create_profile_with_valid_data(self):
        """Test profile creation with valid array data."""
        profiler = EntityBaselineProfiler(seq_length=5)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        profile = profiler.create_profile("entity_001", data)

        assert profile is not None
        assert "mean" in profile
        assert "std" in profile
        assert "percentiles" in profile
        assert profile["mean"] == pytest.approx(3.0)

    def test_update_profile_incrementally(self):
        """Test incremental profile updates."""
        profiler = EntityBaselineProfiler(seq_length=5)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        profiler.create_profile("entity_001", data)
        initial_mean = profiler.get_entity_profile("entity_001")["mean"]

        profiler.update_profile("entity_001", 6.0)
        updated_mean = profiler.get_entity_profile("entity_001")["mean"]

        assert updated_mean > initial_mean
