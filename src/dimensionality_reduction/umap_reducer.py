"""
UMAP (Uniform Manifold Approximation and Projection) Dimensionality Reducer.
Includes PCA fallback if optional umap-learn package is not installed.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class UMAPReducer:
    """UMAP for nonlinear manifold projection with PCA fallback."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: Optional[int] = 42
    ) -> None:
        """Initialize UMAPReducer."""
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

        self.umap_model = None
        self.scaler = StandardScaler()
        self.embedding_ = None

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Fit UMAP model and return low-dimensional projections.

        :param X: Input DataFrame
        :param y: Optional target Series
        :return: Transformed embedding DataFrame
        """
        X_scaled = self.scaler.fit_transform(X)

        try:
            import umap
            self.umap_model = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state
            )
            self.embedding_ = self.umap_model.fit_transform(X_scaled)
        except Exception:
            # Fallback to PCA if umap is unavailable
            self.umap_model = PCA(n_components=min(self.n_components, X.shape[1]), random_state=self.random_state)
            self.embedding_ = self.umap_model.fit_transform(X_scaled)

        cols = ['umap_x', 'umap_y'] if self.n_components == 2 else [f'umap_{i+1}' for i in range(self.n_components)]
        res_df = pd.DataFrame(self.embedding_, columns=cols, index=X.index)

        if y is not None:
            res_df['label'] = y.values

        return res_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new dataset into lower-dimensional space."""
        if self.umap_model is None:
            raise ValueError("Must fit UMAP before transform")

        X_scaled = self.scaler.transform(X)
        emb = self.umap_model.transform(X_scaled)

        cols = ['umap_x', 'umap_y'] if self.n_components == 2 else [f'umap_{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(emb, columns=cols, index=X.index)
