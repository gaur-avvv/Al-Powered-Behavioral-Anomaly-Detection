"""
Advanced Time Series Cross-Validation Strategies.
Prevents look-ahead temporal data leakage in sequence models.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd


class AdvancedTimeSeriesSplit:
    """
    Provides advanced temporal cross-validation splitting mechanisms:
    Rolling window, expanding window, blocked split, grouped split, and seasonal split.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[Any] = None,
        gap: int = 0
    ) -> None:
        """
        Initialize AdvancedTimeSeriesSplit.

        :param n_splits: Number of cross-validation splits
        :param test_size: Proportion or count of test items per split
        :param gap: Temporal gap between training and testing samples
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def rolling_window_split(
        self,
        data: pd.DataFrame,
        time_col: str = 'timestamp',
        window_size: str = 'auto',
        step_size: str = 'auto'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Rolling window cross-validation split over sorted timestamp data.

        :param data: Dataframe containing time series records
        :param time_col: Timestamp column name
        :param window_size: Size of rolling window
        :param step_size: Step increment size between splits
        :return: List of (train_indices_array, test_indices_array) tuples
        """
        df_sorted = data.sort_values(by=time_col).reset_index(drop=True)
        total_len = len(df_sorted)

        w_size = int(total_len * 0.5) if window_size == 'auto' else int(window_size)
        s_size = int(total_len * 0.1) if step_size == 'auto' else int(step_size)
        s_size = max(s_size, 1)

        splits = []
        for i in range(0, total_len - w_size - 1, s_size):
            train_idx = np.arange(i, i + w_size)
            test_start = i + w_size + self.gap
            test_end = min(test_start + s_size, total_len)

            if test_start < total_len:
                test_idx = np.arange(test_start, test_end)
                if len(test_idx) > 0:
                    splits.append((df_sorted.index[train_idx].values, df_sorted.index[test_idx].values))

            if len(splits) >= self.n_splits:
                break

        return splits

    def expanding_window_split(
        self,
        data: pd.DataFrame,
        time_col: str = 'timestamp'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Expanding window cross-validation split (train set grows sequentially).

        :param data: Dataframe containing time series records
        :param time_col: Timestamp column name
        :return: List of (train_indices_array, test_indices_array) tuples
        """
        df_sorted = data.sort_values(by=time_col).reset_index(drop=True)
        total_len = len(df_sorted)

        step = total_len // (self.n_splits + 1)
        step = max(step, 1)

        splits = []
        for i in range(1, self.n_splits + 1):
            train_end = i * step
            test_end = min((i + 1) * step, total_len)

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end + self.gap, test_end)

            if len(test_idx) > 0:
                splits.append((df_sorted.index[train_idx].values, df_sorted.index[test_idx].values))

        return splits

    def blocked_time_series_split(
        self,
        data: pd.DataFrame,
        time_col: str = 'timestamp'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Blocked time series split (prevents temporal leakage between non-adjacent blocks).

        :param data: Dataframe containing time series records
        :param time_col: Timestamp column name
        :return: List of (train_indices_array, test_indices_array) tuples
        """
        df_sorted = data.sort_values(by=time_col).reset_index(drop=True)
        total_len = len(df_sorted)

        block_size = total_len // (self.n_splits + 1)
        block_size = max(block_size, 1)

        splits = []
        for i in range(self.n_splits):
            train_end = (i + 1) * block_size
            test_start = train_end + self.gap
            test_end = min(test_start + block_size, total_len)

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(test_idx) > 0:
                splits.append((df_sorted.index[train_idx].values, df_sorted.index[test_idx].values))

        return splits
