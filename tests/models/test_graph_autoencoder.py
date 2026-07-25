"""
Unit tests for GraphAutoencoder and GraphDataPreprocessor.
"""

import pytest
import torch
import numpy as np
from src.models.gnn.graph_neural_network import GraphAutoencoder, create_graph_autoencoder
from src.models.gnn.data_preprocessor import GraphDataPreprocessor


class TestGraphAutoencoder:
    """Test suite covering GNN Graph Autoencoder."""

    def test_model_creation(self):
        """Test model creation."""
        model = create_graph_autoencoder(10, {'gnn_type': 'gcn', 'hidden_dim': 32, 'latent_dim': 16})
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass over node features and edge index."""
        model = GraphAutoencoder(node_feature_dim=10, hidden_dim=32, latent_dim=16)
        x = torch.randn(15, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        reconstructed, latent = model(x, edge_index)

        assert reconstructed.shape == (15, 10)
        assert latent.shape == (1, 16)

    def test_graph_preprocessor(self):
        """Test adjacency matrix graph conversion."""
        preprocessor = GraphDataPreprocessor()
        adj = np.eye(5, dtype=np.float32)
        features = np.random.randn(5, 8)

        x, edge_index = preprocessor.create_graph_from_adjacency(adj, features)

        assert x.shape == (5, 8)
        assert edge_index.shape[0] == 2
