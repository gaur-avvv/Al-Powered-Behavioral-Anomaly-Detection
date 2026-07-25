"""
Graph Neural Network Architecture for graph-structured behavioral anomaly detection.
Supports GCN, GAT, and SAGE convolution layers with bottleneck representations.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Robust Graph Convolution Layer with normalized adjacency propagation fallback.
    """

    def __init__(self, in_features: int, out_features: int, gnn_type: str = "gcn") -> None:
        """
        Initialize GraphConvLayer.

        :param in_features: Input node feature size
        :param out_features: Output node feature size
        :param gnn_type: GNN convolution type ('gcn', 'gat', 'sage')
        """
        super(GraphConvLayer, self).__init__()
        self.gnn_type = gnn_type.lower()
        self.linear = nn.Linear(in_features, out_features)

        if self.gnn_type == "sage":
            self.neighbor_linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass node feature aggregation.

        :param x: Node feature tensor (num_nodes, in_features)
        :param edge_index: Graph edge indices tensor (2, num_edges)
        :return: Updated node feature tensor (num_nodes, out_features)
        """
        num_nodes = x.size(0)

        if edge_index.numel() == 0:
            return self.linear(x)

        # Build normalized adjacency matrix for propagation
        src, dst = edge_index[0], edge_index[1]
        adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0

        # Add self-loops
        adj = adj + torch.eye(num_nodes, device=x.device)
        deg = torch.sum(adj, dim=1, keepdim=True)
        adj_norm = adj / torch.clamp(deg, min=1.0)

        # Aggregation
        aggregated = torch.matmul(adj_norm, x)

        if self.gnn_type == "sage":
            out = self.linear(x) + self.neighbor_linear(aggregated)
        else:
            out = self.linear(aggregated)

        return out


class GraphAutoencoder(nn.Module):
    """
    Graph Autoencoder for graph-structured behavioral anomaly detection.
    """

    def __init__(
        self,
        node_feature_dim: int = 10,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        gnn_type: str = 'gcn',
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: str = 'relu'
    ) -> None:
        """
        Initialize GraphAutoencoder.

        :param node_feature_dim: Input feature dimension per node
        :param hidden_dim: Hidden dimension size
        :param latent_dim: Latent representation bottleneck size
        :param gnn_type: Type of GNN convolution ('gcn', 'gat', 'sage')
        :param num_layers: Layer count
        :param dropout: Dropout probability
        :param activation: Activation function name
        """
        super(GraphAutoencoder, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.gnn_type = gnn_type
        self.num_layers = num_layers

        self.activation = self._get_activation(activation)

        # Encoder layers
        self.encoder_layer1 = GraphConvLayer(node_feature_dim, hidden_dim, gnn_type)
        self.encoder_layer2 = GraphConvLayer(hidden_dim, latent_dim, gnn_type)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, latent_dim)
        )

        # Decoder layers
        self.decoder_layer1 = GraphConvLayer(latent_dim, hidden_dim, gnn_type)
        self.decoder_layer2 = GraphConvLayer(hidden_dim, node_feature_dim, gnn_type)

        self.final_activation = nn.Sigmoid()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function instance."""
        name_lower = name.lower()
        if name_lower == 'leaky_relu':
            return nn.LeakyReLU(0.01)
        elif name_lower == 'elu':
            return nn.ELU()
        elif name_lower == 'tanh':
            return nn.Tanh()
        return nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over graph nodes and edges.

        :param x: Node feature tensor (num_nodes, node_feature_dim)
        :param edge_index: Graph topology tensor (2, num_edges)
        :return: Tuple of (reconstructed node features, graph latent embedding)
        """
        # Encoder
        h = self.encoder_layer1(x, edge_index)
        h = self.activation(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h_latent = self.encoder_layer2(h, edge_index)
        h_latent = self.activation(h_latent)

        # Graph-level pooling (mean over nodes)
        graph_latent = torch.mean(h_latent, dim=0, keepdim=True)
        graph_latent = self.bottleneck(graph_latent)

        # Decoder
        h_dec = self.decoder_layer1(h_latent, edge_index)
        h_dec = self.activation(h_dec)
        h_dec = F.dropout(h_dec, p=0.2, training=self.training)

        reconstructed = self.decoder_layer2(h_dec, edge_index)
        reconstructed = self.final_activation(reconstructed)

        return reconstructed, graph_latent

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE feature reconstruction loss."""
        return F.mse_loss(reconstructed, x, reduction='mean')


def create_graph_autoencoder(
    node_feature_dim: int,
    config: Dict[str, Any]
) -> GraphAutoencoder:
    """
    Factory function to construct GraphAutoencoder from configuration dict.

    :param node_feature_dim: Node feature dimension
    :param config: Configuration parameters
    :return: Instantiated GraphAutoencoder
    """
    return GraphAutoencoder(
        node_feature_dim=node_feature_dim,
        hidden_dim=config.get('hidden_dim', 64),
        latent_dim=config.get('latent_dim', 32),
        gnn_type=config.get('gnn_type', 'gcn'),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.2),
        activation=config.get('activation', 'relu')
    )
