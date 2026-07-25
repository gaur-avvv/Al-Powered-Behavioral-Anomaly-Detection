"""
Unit tests for L1FeatureSelector, TreeFeatureSelector, and NeuralFeatureSelector.
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_selection.l1_feature_selection import L1FeatureSelector
from src.feature_selection.tree_feature_selection import TreeFeatureSelector
from src.feature_selection.neural_feature_selection import NeuralFeatureSelector


class TestFeatureSelection:
    """Test suite for feature selection classes."""

    @pytest.fixture
    def dataset(self):
        """Create synthetic DataFrame dataset."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f'f_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 50))
        return X, y

    def test_l1_feature_selection(self, dataset):
        """Test L1 feature selection fit and transform."""
        X, y = dataset
        selector = L1FeatureSelector(model_type='classification', max_iter=200)
        res = selector.fit(X, y)

        assert res is not None
        assert len(selector.get_support()) > 0
        X_trans = selector.transform(X)
        assert X_trans.shape[1] == len(selector.get_support())

    def test_tree_feature_selection(self, dataset):
        """Test Tree stability feature selection."""
        X, y = dataset
        selector = TreeFeatureSelector(n_estimators=10)
        res = selector.fit(X, y, n_repeats=2)

        assert res is not None
        assert len(selector.get_support()) > 0

    def test_neural_feature_selection(self, dataset):
        """Test Neural feature selection."""
        X, y = dataset
        selector = NeuralFeatureSelector(input_dim=5, n_iterations=5)
        res = selector.fit(X, y, model_type='classification', n_bootstrap=2)

        assert res is not None
        X_trans = selector.transform(X)
        assert X_trans.shape[1] > 0
