"""
Graph Data Preprocessor for constructing edge indices and PyTorch node feature matrices.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class GraphDataPreprocessor:
    """Preprocesses entity interaction logs into graph tensors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GraphDataPreprocessor."""
        self.config = config or {}
        self.scaler = StandardScaler()

    def create_graph_from_edges(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        node_features: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct node feature tensor and edge index tensor from node and edge DataFrames.

        :param nodes: DataFrame containing node feature columns
        :param edges: DataFrame containing 'source' and 'target' index columns
        :param node_features: List of column names to extract as features
        :return: Tuple of (x_node_tensor, edge_index_tensor)
        """
        features_arr = nodes[node_features].values
        features_scaled = self.scaler.fit_transform(features_arr)
        x = torch.tensor(features_scaled, dtype=torch.float32)

        if not edges.empty and "source" in edges.columns and "target" in edges.columns:
            src = torch.tensor(edges["source"].values, dtype=torch.long)
            dst = torch.tensor(edges["target"].values, dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return x, edge_index

    def create_graph_from_adjacency(
        self,
        adjacency_matrix: np.ndarray,
        node_features: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct graph tensors from 2D adjacency matrix and feature matrix.

        :param adjacency_matrix: 2D square adjacency matrix
        :param node_features: 2D node feature matrix
        :return: Tuple of (x_node_tensor, edge_index_tensor)
        """
        features_scaled = self.scaler.fit_transform(node_features)
        x = torch.tensor(features_scaled, dtype=torch.float32)

        edges = np.argwhere(adjacency_matrix > 0).T
        edge_index = torch.tensor(edges, dtype=torch.long)

        return x, edge_index
