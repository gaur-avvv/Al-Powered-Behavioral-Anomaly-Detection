"""
Trainer for LSTM Autoencoder with comprehensive training, validation, and checkpointing.
"""

from typing import Dict, Any, Optional, Tuple
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

from src.models.autoencoder.lstm_autoencoder import LSTMAutoencoder


class AutoencoderTrainer:
    """Trainer for LSTM Autoencoder models."""

    def __init__(
        self,
        model: LSTMAutoencoder,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize AutoencoderTrainer.

        :param model: Target LSTMAutoencoder instance
        :param device: Target PyTorch device string ('cpu' or 'cuda')
        :param config: Configuration parameters dict
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.config = config or self._default_config()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Training history
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')

    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration parameters."""
        return {
            'batch_size': 64,
            'epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'clip_grad': 1.0,
            'patience': 10,
            'sequence_length': 10,
            'validation_split': 0.2,
            'early_stopping': True,
            'checkpoint_path': 'models/autoencoder.pth'
        }

    def prepare_data(
        self,
        data: np.ndarray,
        sequence_length: int = 10
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare sequence data for training.

        :param data: 2D numpy array of shape (samples, features)
        :param sequence_length: Sequence window length
        :return: Tuple of (train_loader, val_loader)
        """
        data_arr = np.asarray(data, dtype=np.float32)
        sequences = []
        for i in range(len(data_arr) - sequence_length):
            sequences.append(data_arr[i:i + sequence_length])

        if len(sequences) == 0:
            sequences = [data_arr]

        sequences_arr = np.array(sequences, dtype=np.float32)

        # Split preserving temporal order
        train_data, val_data = train_test_split(
            sequences_arr,
            test_size=self.config.get('validation_split', 0.2),
            shuffle=False
        )

        train_dataset = TensorDataset(torch.from_numpy(train_data))
        val_dataset = TensorDataset(torch.from_numpy(val_data))

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.config.get('batch_size', 64), len(train_data)),
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(self.config.get('batch_size', 64), len(val_data)),
            shuffle=False
        )

        return train_loader, val_loader

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train model for one single epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            x_batch = batch[0].to(self.device)

            self.optimizer.zero_grad()
            reconstructed, _ = self.model(x_batch)
            loss = self.model.compute_loss(x_batch, reconstructed)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('clip_grad', 1.0)
            )
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def validate(self, dataloader: DataLoader) -> float:
        """Validate model over evaluation dataset."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(self.device)
                reconstructed, _ = self.model(x_batch)
                loss = self.model.compute_loss(x_batch, reconstructed)
                total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def train(self, data: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
        """Execute complete training and validation pipeline."""
        train_loader, val_loader = self.prepare_data(
            data,
            self.config.get('sequence_length', 10)
        )

        epochs = self.config.get('epochs', 20)
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint()

            if self.config.get('early_stopping', True) and self._should_stop(epoch):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        return {
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'best_val_loss': self.best_val_loss
        }

    def _should_stop(self, epoch: int) -> bool:
        """Evaluate early stopping criterion."""
        patience = self.config.get('patience', 10)
        if epoch < patience + 5:
            return False
        recent_losses = self.val_loss_history[-patience:]
        return all(loss >= self.best_val_loss for loss in recent_losses)

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save training state checkpoint."""
        save_path = path or self.config.get('checkpoint_path', 'models/autoencoder.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': len(self.train_loss_history),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss_history[-1] if self.train_loss_history else 0.0,
            'val_loss': self.val_loss_history[-1] if self.val_loss_history else 0.0,
            'best_val_loss': self.best_val_loss
        }, save_path)

    def load_checkpoint(self, path: str) -> int:
        """Load model state checkpoint."""
        if not os.path.exists(path):
            return 0
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint.get('epoch', 0)
