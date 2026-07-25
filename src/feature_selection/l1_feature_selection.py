"""
Feature selection using L1 regularization (LASSO / L1 Logistic Regression).
"""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class L1FeatureSelector:
    """Feature selection using L1 regularization."""

    def __init__(
        self,
        model_type: str = 'regression',
        alpha_range: tuple = (-3.0, 1.0),
        cv_folds: int = 5,
        max_iter: int = 1000,
        random_state: int = 42
    ) -> None:
        """Initialize L1 Feature Selector."""
        self.model_type = model_type
        self.alpha_range = alpha_range
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.random_state = random_state

        self.selector = None
        self.selected_features = None
        self.feature_importances_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold_strategy: str = '1se',
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Fit L1 feature selector.

        :param X: Feature matrix DataFrame
        :param y: Target Series
        :param threshold_strategy: '1se' or 'min'
        :param verbose: Verbosity boolean flag
        :return: DataFrame of selected features and importances
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.model_type == 'regression':
            model = LassoCV(
                alphas=np.logspace(*self.alpha_range, 10),
                cv=self.cv_folds,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
        else:
            model = LogisticRegressionCV(
                Cs=10,
                cv=self.cv_folds,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver='liblinear'
            )

        model.fit(X_scaled, y)

        self.selector = SelectFromModel(model, threshold='mean', prefit=True)
        support = self.selector.get_support()

        if not any(support):
            support = np.ones(X.shape[1], dtype=bool)

        self.selected_features = X.columns[support].tolist()
        coefs = model.coef_ if self.model_type == 'regression' else model.coef_[0]
        self.feature_importances_ = pd.Series(
            np.abs(coefs),
            index=X.columns
        ).sort_values(ascending=False)

        return pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.feature_importances_[self.selected_features].values
        })

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected feature columns."""
        if self.selected_features is None:
            raise ValueError("Must fit selector before transform")
        return X[self.selected_features]

    def get_support(self) -> List[str]:
        """Get list of selected feature column names."""
        return self.selected_features or []

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances series."""
        return self.feature_importances_

    def get_selected_count(self) -> int:
        """Get count of selected features."""
        return len(self.selected_features or [])
