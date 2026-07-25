"""
Early Stopping and Model Overfitting Monitor.
"""

from typing import List, Dict, Optional


class EarlyStopping:
    """Monitors metric progression to trigger early stopping."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        restore_best: bool = True
    ) -> None:
        """Initialize EarlyStopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.wait = 0
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.stopped_epoch = 0

    def should_stop(self, current_score: float) -> bool:
        """
        Check if metric progression has plateaued.

        :param current_score: Current epoch score float
        :return: True if training should stop
        """
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                return True
            return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.wait = 0
        self.best_score = float('inf') if self.mode == 'min' else -float('inf')
        self.stopped_epoch = 0


class ModelMonitor:
    """Diagnoses overfitting or underfitting from loss trajectories."""

    @staticmethod
    def diagnose_overfitting_underfitting(
        train_loss: List[float],
        val_loss: List[float]
    ) -> Dict[str, str]:
        """
        Analyze loss histories to diagnose capacity issues.

        :param train_loss: List of training losses
        :param val_loss: List of validation losses
        :return: Diagnosis dict
        """
        if not train_loss or not val_loss:
            return {"status": "insufficient_data"}

        train_final = train_loss[-1]
        val_final = val_loss[-1]

        diagnosis = {}
        if val_final > train_final * 1.5:
            diagnosis['overfitting'] = "Model is overfitting"
            diagnosis['recommendation'] = "Increase regularization, add dropout, or use early stopping"
        elif train_final > 1.0 and val_final > 1.0:
            diagnosis['underfitting'] = "Model is underfitting"
            diagnosis['recommendation'] = "Increase model capacity or train for more epochs"
        else:
            diagnosis['good_fit'] = "Model appears well-fitted"
            diagnosis['recommendation'] = "Maintain current architecture"

        return diagnosis
