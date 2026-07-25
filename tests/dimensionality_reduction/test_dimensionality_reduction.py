"""
Unit tests for PCAReducer, TSNEDimensionReducer, and UMAPReducer.
"""

import pytest
import numpy as np
import pandas as pd
from src.dimensionality_reduction.pca_reducer import PCAReducer
from src.dimensionality_reduction.tsne_reducer import TSNEDimensionReducer
from src.dimensionality_reduction.umap_reducer import UMAPReducer


class TestDimensionalityReduction:
    """Test suite covering dimensionality reduction classes."""

    @pytest.fixture
    def dataset(self):
        """Create synthetic DataFrame dataset."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(30, 6), columns=[f'f_{i}' for i in range(6)])
        y = pd.Series(np.random.randint(0, 2, 30))
        return X, y

    def test_pca_reducer(self, dataset):
        """Test PCA fit, transform, and inverse_transform."""
        X, y = dataset
        reducer = PCAReducer(n_components=2)
        exp_df = reducer.fit(X)

        assert exp_df is not None
        X_pca = reducer.transform(X)
        assert X_pca.shape == (30, 2)

        X_inv = reducer.inverse_transform(X_pca)
        assert X_inv.shape == X.shape

    def test_tsne_reducer(self, dataset):
        """Test t-SNE fit_transform."""
        X, y = dataset
        reducer = TSNEDimensionReducer(n_components=2, perplexity=5, n_iter=250)
        X_tsne = reducer.fit_transform(X, y)

        assert X_tsne.shape[0] == 30
        assert 'tsne_x' in X_tsne.columns
        assert 'tsne_y' in X_tsne.columns

    def test_umap_reducer(self, dataset):
        """Test UMAP fit_transform."""
        X, y = dataset
        reducer = UMAPReducer(n_components=2, n_neighbors=5)
        X_umap = reducer.fit_transform(X, y)

        assert X_umap.shape[0] == 30
        assert 'umap_x' in X_umap.columns
