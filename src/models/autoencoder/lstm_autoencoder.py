"""
LSTM Autoencoder for sequence anomaly detection.
Configurable architecture supporting bidirectional, attention mechanisms, and bottleneck representations.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for multi-dimensional sequence anomaly detection.
    Configurable architecture with optional bidirectional layers and multi-head attention.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        attention: bool = False
    ) -> None:
        """
        Initialize LSTM Autoencoder.

        :param input_dim: Dimension of input feature sequence
        :param hidden_dim: Hidden dimension size of LSTM layers
        :param latent_dim: Bottleneck latent space dimension
        :param num_layers: Number of LSTM layers
        :param dropout: Dropout probability
        :param bidirectional: Whether to use bidirectional LSTM encoder
        :param attention: Whether to apply multihead attention
        """
        super(LSTMAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = attention

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Bottleneck representation
        encoder_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.bottleneck = nn.Sequential(
            nn.Linear(encoder_output_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.decoder_linear = nn.Linear(
            hidden_dim * (2 if bidirectional else 1),
            input_dim
        )

        # Attention mechanism (optional)
        if attention:
            self.attn_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim * (2 if bidirectional else 1),
                num_heads=2 if bidirectional else 1,
                batch_first=True
            )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param x: Input sequence tensor (batch, seq_len, input_dim)
        :param hidden: Optional initial hidden state
        :return: Tuple of (reconstructed sequence tensor, latent embedding tensor)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Encoder
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)

        encoder_out, (h_n, c_n) = self.encoder(x, hidden)

        # Extract representation from last timestep
        if self.bidirectional:
            last_out = encoder_out[:, -1, :]
        else:
            last_out = encoder_out[:, -1, :]

        # Bottleneck
        latent = self.bottleneck(last_out)

        # Decoder
        decoder_in = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder(decoder_in)

        # Apply attention if enabled
        if self.use_attention:
            decoder_out, _ = self.attn_layer(decoder_out, encoder_out, encoder_out)

        # Final reconstruction projection
        reconstructed = self.decoder_linear(decoder_out)

        return reconstructed, latent

    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize zero hidden state."""
        bidirectional_mult = 2 if self.bidirectional else 1
        h_0 = torch.zeros(
            self.num_layers * bidirectional_mult,
            batch_size,
            self.hidden_dim,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * bidirectional_mult,
            batch_size,
            self.hidden_dim,
            device=device
        )
        return h_0, c_0

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction Mean Squared Error (MSE) loss."""
        return F.mse_loss(reconstructed, x, reduction='mean')


def create_lstm_autoencoder(
    input_dim: int,
    config: Dict[str, Any]
) -> LSTMAutoencoder:
    """
    Factory function to construct LSTMAutoencoder from config dictionary.

    :param input_dim: Feature dimension
    :param config: Configuration dict
    :return: Instantiated LSTMAutoencoder instance
    """
    return LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=config.get('hidden_dim', 64),
        latent_dim=config.get('latent_dim', 32),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.2),
        bidirectional=config.get('bidirectional', False),
        attention=config.get('attention', False)
    )
