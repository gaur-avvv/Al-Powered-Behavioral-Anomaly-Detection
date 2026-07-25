"""
Regularization Manager for finding optimal regularization strengths (alpha, L1 ratio) and Random Forest tuning.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')


class RegularizationManager:
    """Utilities for searching optimal regularization parameters."""

    @staticmethod
    def find_optimal_alpha(
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'ridge',
        cv_folds: int = 5,
        random_state: int = 42
    ) -> float:
        """
        Find optimal alpha parameter using cross-validation.

        :param X: Feature DataFrame
        :param y: Target Series
        :param model_type: 'ridge', 'lasso', or 'elastic_net'
        :return: Optimal alpha float
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        alphas = np.logspace(-3, 3, 20)

        if model_type == 'ridge':
            model = RidgeCV(alphas=alphas, cv=cv_folds)
        elif model_type == 'lasso':
            model = LassoCV(alphas=alphas, cv=cv_folds, random_state=random_state)
        else:
            model = ElasticNetCV(alphas=alphas, cv=cv_folds, random_state=random_state)

        model.fit(X_scaled, y)
        return float(model.alpha_)

    @staticmethod
    def find_optimal_l1_ratio(
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        random_state: int = 42
    ) -> float:
        """Find optimal L1 ratio for ElasticNet models."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=cv_folds,
            random_state=random_state
        )
        model.fit(X_scaled, y)
        return float(model.l1_ratio_)

    @staticmethod
    def tune_random_forest(
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int = 10,
        cv_folds: int = 3,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Tune Random Forest hyperparameter distributions."""
        param_dist = {
            'n_estimators': [20, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        model = RandomForestRegressor(random_state=random_state)
        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            random_state=random_state
        )
        search.fit(X, y)
        return search.best_params_
