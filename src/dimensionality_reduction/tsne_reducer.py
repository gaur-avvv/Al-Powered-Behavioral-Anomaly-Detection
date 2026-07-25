"""
t-Distributed Stochastic Neighbor Embedding (t-SNE) Dimensionality Reducer.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TSNEDimensionReducer:
    """t-SNE for nonlinear 2D/3D manifold reduction."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: Optional[int] = 42
    ) -> None:
        """Initialize TSNEDimensionReducer."""
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

        self.tsne = None
        self.scaler = StandardScaler()
        self.embedding_ = None

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        metric: str = 'euclidean',
        init: str = 'pca',
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Fit t-SNE model and return low-dimensional embedding.

        :param X: Input feature DataFrame
        :param y: Optional labels Series
        :return: Transformed embedding DataFrame
        """
        X_scaled = self.scaler.fit_transform(X)

        perp = min(self.perplexity, max(1.0, (X.shape[0] - 1) / 3.0))

        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=perp,
            early_exaggeration=self.early_exaggeration,
            learning_rate=self.learning_rate,
            metric=metric,
            random_state=self.random_state
        )

        self.embedding_ = self.tsne.fit_transform(X_scaled)

        cols = ['tsne_x', 'tsne_y'] if self.n_components == 2 else [f'tsne_{i+1}' for i in range(self.n_components)]
        res_df = pd.DataFrame(self.embedding_, columns=cols, index=X.index)

        if y is not None:
            res_df['label'] = y.values

        return res_df
