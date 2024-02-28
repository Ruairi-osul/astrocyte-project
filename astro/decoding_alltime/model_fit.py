from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    BaseCrossValidator,
    train_test_split,
)
from typing import Callable, Tuple
import pandas as pd
import numpy as np


class ModelFitter:
    """ModelFitter base class for fitting and evaluating models"""

    def fit_model(*args, **kwargs):
        raise NotImplementedError

    def eval_model(*args, **kwargs):
        raise NotImplementedError


class ModelFitterFrac(ModelFitter):
    """ModelFitterFrac fits and evaluates a model using a train/test split

    Methods:
        fit_model: Fit a model using a train/test split
        eval_model: Evaluate a model using a train/test split

    """

    def __init__(
        self,
        metric: Callable[[np.ndarray], float],
        train_frac: float = 0.75,
        shuffle=False,
    ):
        """Initialize ModelFitterFrac

        Args:
            metric (Callable[[np.ndarray], float]): Metric to use for evaluation
            train_frac (float, optional): Fraction of data to use for training. Defaults to 0.75.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to False.
        """
        self.metric = metric
        self.train_frac = train_frac
        self.shuffle = shuffle

    def split_data(
        self,
        df_predictors: pd.DataFrame,
        target: pd.Series,
        shuffle: bool | None = None,
        train_frac: float | None = None,
    ) -> Tuple[
        pd.DataFrame | np.ndarray,
        pd.DataFrame | np.ndarray,
        pd.Series | np.ndarray,
        pd.Series | np.ndarray,
    ]:
        """Split data into train and test sets

        Args:
            df_predictors (pd.DataFrame): DataFrame of predictors
            target (pd.Series): Series of target values
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to None (value from initialization)
            train_frac (float, optional): Fraction of data to use for training. Defaults to None (value from initialization)
        """
        if shuffle is None:
            shuffle = self.shuffle
        if train_frac is None:
            train_frac = self.train_frac
        return train_test_split(
            df_predictors, target, train_size=train_frac, shuffle=shuffle
        )

    def fit_model(
        self,
        df_predictors: pd.DataFrame | np.ndarray,
        model: BaseEstimator,
        target: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray | None = None,
        train_frac: float | None = None,
        shuffle: bool | None = None,
        **kwargs,
    ) -> BaseEstimator:
        """Fit a model using a train/test split

        Args:
            df_predictors (pd.DataFrame | np.ndarray): DataFrame of predictors
            model (BaseEstimator): Model to fit
            target (pd.Series | np.ndarray): Series of target values
            train_frac (float, optional): Fraction of data to use for training. Defaults to None (value from initialization)
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to None (value from initialization)

        Returns:
            BaseEstimator: Fitted model
        """
        X_train, _, y_train, _ = self.split_data(
            df_predictors, target, shuffle=shuffle, train_frac=train_frac
        )

        model.fit(X_train, y_train)
        return model

    def eval_model(
        self,
        df_predictors: pd.DataFrame | np.ndarray,
        model: BaseEstimator,
        target: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray | None = None,
        train_frac: float | None = None,
        shuffle: bool | None = None,
        **kwargs,
    ) -> float:
        """Evaluate a model using a train/test split

        Args:
            df_predictors (pd.DataFrame | np.ndarray): DataFrame of predictors
            model (BaseEstimator): Model to evaluate
            target (pd.Series | np.ndarray): Series of target values
            train_frac (float, optional): Fraction of data to use for training. Defaults to None (value from initialization)
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to None (value from initialization)

        Returns:
            float: Score of the model
        """
        _, X_test, _, y_test = self.split_data(
            df_predictors, target, shuffle=shuffle, train_frac=train_frac
        )

        preds = model.predict(X_test)
        return self.metric(y_test, preds)


class ModelFitterCV(ModelFitter):
    """ModelFitterCV fits and evaluates a model using cross validation


    Methods:
        fit_model: Fit a model using cross validation to obtain the best model
        eval_model: Evaluate a model using cross validation
    """

    def __init__(
        self,
        metric: Callable[[np.ndarray], float],
        cv: BaseCrossValidator,
    ):
        """Initialize ModelFitterCV

        Args:
            metric (Callable[[np.ndarray], float]): Metric to use for evaluation
            cv (BaseCrossValidator): Cross validation strategy
        """

        self.metric = metric
        self.cv = cv

    def fit_model(
        self,
        df_predictors: pd.DataFrame | np.ndarray,
        model: BaseEstimator,
        target: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray | None = None,
    ) -> BaseEstimator:
        """Obtain the best model from cross validation

        Args:
            df_predictors (pd.DataFrame | np.ndarray): DataFrame of predictors
            model (BaseEstimator): Model to evaluate
            target (pd.Series | np.ndarray): Series of target values
            groups (pd.Series | np.ndarray | None, optional): Optional series of group values. Defaults to None.

        Returns:
            BaseEstimator: Best model from cross validation
        """

        models = np.empty(self.cv.get_n_splits(groups=groups), dtype=object)
        scores = np.empty(self.cv.get_n_splits(groups=groups))

        X_arr = np.asarray(df_predictors)
        y_arr = np.asarray(target)
        for i, (train_idx, test_idx) in enumerate(
            self.cv.split(X_arr, y_arr, groups=groups)
        ):
            X_train, X_test = (
                X_arr[train_idx],
                X_arr[test_idx],
            )
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = self.metric(y_test, preds)

            models[i] = model
            scores[i] = score

        best_model = models[np.argmax(scores)]
        return best_model

    def eval_model(
        self,
        df_predictors: pd.DataFrame | np.ndarray,
        model: BaseEstimator,
        target: pd.Series | np.ndarray,
        groups: pd.Series | np.ndarray | None = None,
    ) -> float:
        """Evaluate a model using cross validation

        Args:
            df_predictors (pd.DataFrame | np.ndarray): DataFrame of predictors
            model (BaseEstimator): Model to evaluate
            target (pd.Series | np.ndarray): Series of target values
            groups (pd.Series | np.ndarray | None, optional): Optional series of group values. Defaults to None.

        Returns:
            float: Mean score of the model across cross validation folds
        """

        scores = np.empty(self.cv.get_n_splits(groups=groups))

        X_arr = np.asarray(df_predictors)
        y_arr = np.asarray(target)
        for i, (train_idx, test_idx) in enumerate(
            self.cv.split(X_arr, y_arr, groups=groups)
        ):
            X_train, X_test = (
                X_arr[train_idx],
                X_arr[test_idx],
            )
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = self.metric(y_test, preds)

            scores[i] = score

        return scores.mean()
