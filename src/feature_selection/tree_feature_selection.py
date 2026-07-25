"""
Tree-Based Feature Selection with Bootstrap Stability Selection.
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')


class TreeFeatureSelector:
    """Feature selection using tree ensembles and stability scoring."""

    def __init__(
        self,
        model_type: str = 'random_forest',
        n_estimators: int = 50,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42
    ) -> None:
        """Initialize TreeFeatureSelector."""
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.selected_features = None
        self.feature_importances_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
        cv_folds: int = 5,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Fit tree feature selector using stability selection.

        :param X: Feature DataFrame
        :param y: Target Series
        :param n_repeats: Number of bootstrap iterations
        :return: Selected features DataFrame
        """
        n_samples, n_feats = X.shape
        feature_importances = np.zeros((n_repeats, n_feats))
        is_classification = (y.nunique() <= 10)

        for i in range(n_repeats):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]

            if self.model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state + i
                ) if is_classification else RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state + i
                )
            else:
                model = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    random_state=self.random_state + i
                ) if is_classification else DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    random_state=self.random_state + i
                )

            model.fit(X_boot, y_boot)
            feature_importances[i] = model.feature_importances_

        mean_imp = np.mean(feature_importances, axis=0)
        thresh = float(np.percentile(mean_imp, 50)) if len(mean_imp) > 1 else 0.0
        selected_mask = mean_imp >= thresh

        if not any(selected_mask):
            selected_mask[0] = True

        self.selected_features = X.columns[selected_mask].tolist()
        self.feature_importances_ = pd.Series(mean_imp[selected_mask], index=self.selected_features).sort_values(ascending=False)

        return pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.feature_importances_.values
        })

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame to selected feature columns."""
        if self.selected_features is None:
            raise ValueError("Must fit selector before transform")
        return X[self.selected_features]

    def get_support(self) -> List[str]:
        """Get list of selected feature column names."""
        return self.selected_features or []

    def get_feature_importance(self) -> pd.Series:
        """Get feature importances series."""
        return self.feature_importances_
