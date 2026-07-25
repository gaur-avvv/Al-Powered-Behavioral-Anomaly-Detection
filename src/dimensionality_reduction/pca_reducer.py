"""
Principal Component Analysis (PCA) Dimensionality Reducer.
"""

from typing import Optional, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PCAReducer:
    """PCA for linear dimensionality reduction and explained variance analysis."""

    def __init__(
        self,
        n_components: Optional[Any] = None,
        whiten: bool = False,
        random_state: int = 42
    ) -> None:
        """Initialize PCAReducer."""
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

        self.pca = None
        self.scaler = StandardScaler()
        self.explained_variance_ratio_ = None
        self.components_ = None

    def fit(self, X: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Fit PCA reducer over DataFrame.

        :param X: Feature DataFrame
        :param verbose: Verbosity boolean
        :return: DataFrame of explained variance analysis
        """
        X_scaled = self.scaler.fit_transform(X)

        n_comp = self.n_components
        if n_comp is None or n_comp == 'mle' or n_comp == 'auto':
            n_comp = min(X.shape[0], X.shape[1])

        self.pca = PCA(
            n_components=n_comp,
            whiten=self.whiten,
            random_state=self.random_state
        )
        self.pca.fit(X_scaled)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_

        return pd.DataFrame({
            'component': range(1, len(self.explained_variance_ratio_) + 1),
            'explained_variance': self.pca.explained_variance_,
            'explained_variance_ratio': self.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.explained_variance_ratio_)
        })

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data into principal component space."""
        if self.pca is None:
            raise ValueError("Must fit PCA before transform")

        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=cols, index=X.index)

    def inverse_transform(self, X_pca: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform principal components back to feature space."""
        if self.pca is None:
            raise ValueError("Must fit PCA before inverse transform")

        X_scaled = self.pca.inverse_transform(X_pca.values)
        X_orig = self.scaler.inverse_transform(X_scaled)

        cols = getattr(self.scaler, "feature_names_in_", [f"feature_{i}" for i in range(X_orig.shape[1])])
        return pd.DataFrame(X_orig, columns=cols, index=X_pca.index)
