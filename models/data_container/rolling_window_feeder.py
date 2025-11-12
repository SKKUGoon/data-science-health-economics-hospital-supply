from __future__ import annotations

from datetime import timedelta, datetime
from typing import Iterable, Iterator, Tuple

import pandas as pd


class RollingWindowFeeder:
    """
    Generate expanding training windows with a fixed-width test horizon.

    The feeder yields chronological (train_df, test_df) pairs where the training
    portion spans the previous ``window_size_days`` days and the test portion
    covers the next ``forward_window_days`` days. Each subsequent batch advances
    by the forward window length, enabling weekly backtesting or retraining
    triggers.
    """

    def __init__(
        self,
        window_size_days: int = 730,
        forward_window_days: int = 7,
        drop_empty_batches: bool = True,
    ) -> None:
        if window_size_days <= 0:
            raise ValueError("window_size_days must be a positive integer")
        if forward_window_days <= 0:
            raise ValueError("forward_window_days must be a positive integer")

        self.window_size_days = window_size_days
        self.forward_window_days = forward_window_days
        self.drop_empty_batches = drop_empty_batches

        self._window_delta = timedelta(days=window_size_days)
        self._forward_delta = timedelta(days=forward_window_days)

    def feed(
        self,
        data: pd.DataFrame,
        datetime_col: str,
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, Tuple[datetime, datetime], Tuple[datetime, datetime]]]:
        """
        Yield sequential expanding-window splits.

        Args:
            data: DataFrame containing a datetime column.
            datetime_col: Column name identifying event timestamps.

        Yields:
            Tuple of (train_df, test_df) for each window.
        """
        if datetime_col not in data.columns:
            raise KeyError(f"Column '{datetime_col}' not found in DataFrame")

        if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
            raise TypeError(f"Column '{datetime_col}' must be datetime-like")

        minimum_day = data[datetime_col].min()
        maximum_day = data[datetime_col].max()

        if pd.isna(minimum_day) or pd.isna(maximum_day):
            return

        current_day = minimum_day + self._window_delta
        while current_day <= maximum_day:
            train_start = current_day - self._window_delta
            train_mask = (data[datetime_col] >= train_start) & (data[datetime_col] < current_day)

            test_end = current_day + self._forward_delta
            test_mask = (data[datetime_col] >= current_day) & (data[datetime_col] < test_end)

            train_df = data.loc[train_mask]
            test_df = data.loc[test_mask]

            if not self.drop_empty_batches or (not train_df.empty and not test_df.empty):
                yield (
                    train_df.sort_values(datetime_col).drop(columns=[datetime_col]),
                    test_df.sort_values(datetime_col).drop(columns=[datetime_col]),
                    (train_start, current_day),
                    (current_day, test_end)
                )

            current_day += self._forward_delta


class RollingWindowFeederW:
    """
    Generate expanding training windows with a fixed-width test horizon for weekly data.

    The feeder yields chronological (train_df, test_df) pairs where the training
    portion spans the previous ``window_size_weeks`` weeks and the test portion
    covers the next ``forward_window_weeks`` weeks. Each subsequent batch advances
    by the forward window length, enabling weekly backtesting or retraining
    triggers.
    """

    def __init__(
        self,
        window_size_weeks: int = 104,  # ~2 years
        forward_window_weeks: int = 2,  # 2 weeks ahead
        drop_empty_batches: bool = True,
    ) -> None:
        if window_size_weeks <= 0:
            raise ValueError("window_size_weeks must be a positive integer")
        if forward_window_weeks <= 0:
            raise ValueError("forward_window_weeks must be a positive integer")

        self.window_size_weeks = window_size_weeks
        self.forward_window_weeks = forward_window_weeks
        self.drop_empty_batches = drop_empty_batches

        self._window_delta = timedelta(weeks=window_size_weeks)
        self._forward_delta = timedelta(weeks=forward_window_weeks)

    def feed(
        self,
        data: pd.DataFrame,
        datetime_col: str,
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, Tuple[datetime, datetime], Tuple[datetime, datetime]]]:
        """
        Yield sequential expanding-window splits for weekly data.

        Args:
            data: DataFrame containing a datetime column with weekly frequency.
            datetime_col: Column name identifying event timestamps.

        Yields:
            Tuple of (train_df, test_df, train_range, test_range) for each window.
            train_range and test_range are (start, end) datetime tuples.
        """
        if datetime_col not in data.columns:
            raise KeyError(f"Column '{datetime_col}' not found in DataFrame")

        if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
            raise TypeError(f"Column '{datetime_col}' must be datetime-like")

        minimum_week = data[datetime_col].min()
        maximum_week = data[datetime_col].max()

        if pd.isna(minimum_week) or pd.isna(maximum_week):
            return

        current_week = minimum_week + self._window_delta
        while current_week <= maximum_week:
            train_start = current_week - self._window_delta
            train_mask = (data[datetime_col] >= train_start) & (data[datetime_col] < current_week)

            test_end = current_week + self._forward_delta
            test_mask = (data[datetime_col] >= current_week) & (data[datetime_col] < test_end)

            train_df = data.loc[train_mask]
            test_df = data.loc[test_mask]

            if not self.drop_empty_batches or (not train_df.empty and not test_df.empty):
                yield (
                    train_df.sort_values(datetime_col).drop(columns=[datetime_col]),
                    test_df.sort_values(datetime_col).drop(columns=[datetime_col]),
                    (train_start, current_week),
                    (current_week, test_end)
                )

            current_week += self._forward_delta
