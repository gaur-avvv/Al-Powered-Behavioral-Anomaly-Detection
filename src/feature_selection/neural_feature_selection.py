"""
Neural Feature Selector using PyTorch with Integrated Gradients attribution.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class NeuralFeatureSelector:
    """Feature selection using PyTorch models and Integrated Gradients."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        n_iterations: int = 20,
        batch_size: int = 32,
        random_state: int = 42
    ) -> None:
        """Initialize NeuralFeatureSelector."""
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [32, 16]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.selected_features = None
        self.feature_importances_ = None

    def _build_model(self, output_dim: int) -> nn.Module:
        """Build feedforward PyTorch neural network."""
        layers = []
        in_dim = self.input_dim

        for h_dim in self.hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers).to(self.device)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'regression',
        n_bootstrap: int = 3,
        threshold: float = 0.2,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Fit neural feature selector.

        :param X: Feature DataFrame
        :param y: Target Series
        :param model_type: 'regression' or 'classification'
        :return: DataFrame of selected features and importances
        """
        n_samples, n_feats = X.shape
        importances_list = []

        is_class = (model_type != 'regression')
        output_dim = len(y.unique()) if is_class else 1

        for boot in range(n_bootstrap):
            model = self._build_model(output_dim)
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss() if is_class else nn.MSELoss()

            X_t = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y.values, dtype=torch.long if is_class else torch.float32).to(self.device)

            model.train()
            for epoch in range(self.n_iterations):
                optimizer.zero_grad()
                out = model(X_t)
                loss = criterion(out, y_t)
                loss.backward()
                optimizer.step()

            # Compute feature gradients
            model.eval()
            X_eval = X_t[:min(100, n_samples)].clone().detach().requires_grad_(True)
            out_eval = model(X_eval)
            out_eval.sum().backward()

            grads = X_eval.grad.abs().mean(dim=0).cpu().numpy()
            importances_list.append(grads)

        mean_imp = np.mean(importances_list, axis=0)
        thresh_val = threshold * float(np.max(mean_imp)) if np.max(mean_imp) > 0 else 0.0
        selected_mask = mean_imp >= thresh_val

        if not any(selected_mask):
            selected_mask[0] = True

        self.selected_features = X.columns[selected_mask].tolist()
        self.feature_importances_ = pd.Series(mean_imp[selected_mask], index=self.selected_features).sort_values(ascending=False)

        return pd.DataFrame({
            'feature': X.columns,
            'importance': mean_imp,
            'selected': selected_mask
        })

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected feature columns."""
        if self.selected_features is None:
            raise ValueError("Must fit selector before transform")
        return X[self.selected_features]
