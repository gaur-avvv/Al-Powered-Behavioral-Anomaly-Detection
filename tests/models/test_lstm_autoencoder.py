"""
Unit tests for PyTorch LSTM Autoencoder and AutoencoderTrainer.
"""

import pytest
import torch
import numpy as np
from src.models.autoencoder.lstm_autoencoder import LSTMAutoencoder, create_lstm_autoencoder
from src.models.autoencoder.trainer import AutoencoderTrainer


class TestLSTMAutoencoder:
    """Test suite covering LSTM Autoencoder forward pass, loss, and trainer."""

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Synthetic feature sequence array."""
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 10))

    @pytest.fixture
    def model(self) -> LSTMAutoencoder:
        """Create test model."""
        return LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=16)

    def test_model_creation(self, model: LSTMAutoencoder):
        """Test model layer initialization."""
        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'bottleneck')

    def test_forward_pass(self, model: LSTMAutoencoder):
        """Test tensor dimensions through encoder and decoder."""
        batch_size = 8
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 10)

        reconstructed, latent = model(x)

        assert reconstructed.shape == (batch_size, seq_len, 10)
        assert latent.shape == (batch_size, 16)

    def test_loss_computation(self, model: LSTMAutoencoder):
        """Test MSE reconstruction loss."""
        x = torch.randn(4, 10, 10)
        rec, _ = model(x)
        loss = model.compute_loss(x, rec)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0

    def test_training_pipeline(self, sample_data: np.ndarray):
        """Test training workflow."""
        model = LSTMAutoencoder(input_dim=10, hidden_dim=16, latent_dim=8)
        trainer = AutoencoderTrainer(model, config={'epochs': 3, 'batch_size': 16})

        res = trainer.train(sample_data, verbose=False)

        assert len(res['train_loss_history']) > 0
        assert res['best_val_loss'] < float('inf')
