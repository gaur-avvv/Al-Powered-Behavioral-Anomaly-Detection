"""
Cross Validation Manager for model comparison and strategy selection.
"""

from typing import List, Tuple, Any, Dict
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')


class CrossValidationManager:
    """Manages cross-validation splitter selection and multi-model benchmarking."""

    @staticmethod
    def get_cv_strategy(
        data_type: str = 'standard',
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ) -> Any:
        """Get cross-validation splitter instance."""
        if data_type == 'stratified':
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif data_type == 'time_series':
            return TimeSeriesSplit(n_splits=n_splits)
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @staticmethod
    def evaluate_model_with_cv(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_type: str = 'standard',
        scoring: str = 'accuracy'
    ) -> pd.DataFrame:
        """Evaluate model across cross-validation splits."""
        cv = CrossValidationManager.get_cv_strategy(data_type=cv_type, n_splits=3)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return pd.DataFrame({
            'fold': range(1, len(scores) + 1),
            'score': scores
        })

    @staticmethod
    def compare_models(
        models: List[Tuple[str, Any]],
        X: pd.DataFrame,
        y: pd.Series,
        cv_type: str = 'standard'
    ) -> pd.DataFrame:
        """Compare multiple models using cross-validation scores."""
        records = []
        for name, model in models:
            cv_df = CrossValidationManager.evaluate_model_with_cv(model, X, y, cv_type=cv_type)
            records.append({
                'model': name,
                'mean_score': float(cv_df['score'].mean()),
                'std_score': float(cv_df['score'].std())
            })
        return pd.DataFrame(records)
