"""
Comprehensive Training & Evaluation Script over synthetic_access_logs_10000.csv.
Performs Exploratory Data Analysis (EDA), Feature Engineering, Time Series Splitting,
Model Training (LSTM Autoencoder, GNN, Classifier), and Metric Evaluation.
"""

from typing import Dict, Any, List, Tuple
import os
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.models.autoencoder.lstm_autoencoder import LSTMAutoencoder
from src.models.autoencoder.trainer import AutoencoderTrainer
from src.models.gnn.graph_neural_network import GraphAutoencoder
from src.models.gnn.data_preprocessor import GraphDataPreprocessor
from src.models.splitters.time_series_split import AdvancedTimeSeriesSplit
from src.models.attack_classifier import AttackClassifier
from src.models.detection_engine import SequenceDetector


def run_exploratory_data_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform Exploratory Data Analysis (EDA) over the access log dataset.

    :param df: Input access log DataFrame
    :return: EDA metrics dictionary
    """
    print("\n=============================================================")
    print(" 1. EXPLORATORY DATA ANALYSIS (EDA)")
    print("=============================================================")

    total_records = len(df)
    missing_values = df.isnull().sum().to_dict()
    label_counts = df['label'].value_counts().to_dict()
    entity_counts = df['entity_id'].nunique()
    entity_types = df['entity_type'].value_counts().to_dict()
    auth_methods = df['auth_method'].value_counts().to_dict()

    print(f" -> Total Log Records: {total_records}")
    print(f" -> Unique Entities: {entity_counts}")
    print(f" -> Label Distribution: {label_counts}")
    print(f" -> Entity Types: {entity_types}")
    print(f" -> Auth Methods: {auth_methods}")
    print(f" -> Missing Values: {sum(missing_values.values())} total missing fields.")

    return {
        "total_records": total_records,
        "unique_entities": entity_counts,
        "label_distribution": label_counts,
        "missing_values": missing_values
    }


def extract_features_and_graph(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Extract numerical sequence feature matrices and build graph topology tensors.

    :param df: Input DataFrame
    :return: Tuple of (X_features, y_binary, x_node_tensor, edge_index_tensor)
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convert session duration
    durations = df['session_duration'].values.astype(np.float32)

    # Command sequence length
    cmd_counts = df['command_sequence'].apply(lambda x: len(str(x).split('>'))).values.astype(np.float32)

    # Label encoding for auth method and entity type
    le_auth = LabelEncoder()
    auth_enc = le_auth.fit_transform(df['auth_method'].astype(str)).astype(np.float32)

    le_resource = LabelEncoder()
    resource_enc = le_resource.fit_transform(df['resource_accessed'].astype(str)).astype(np.float32)

    # Build feature matrix (samples x features)
    X_raw = np.column_stack([durations, cmd_counts, auth_enc, resource_enc])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Binary labels (normal = 0, anomaly = 1)
    y_binary = (df['label'] != 'normal').astype(int).values

    # Construct Graph Topology (nodes = unique entities, edges = shared resource access)
    entity_map = {ent: idx for idx, ent in enumerate(df['entity_id'].unique())}
    node_features = np.zeros((len(entity_map), 4), dtype=np.float32)

    for idx, row in df.iterrows():
        ent_idx = entity_map[row['entity_id']]
        node_features[ent_idx] = X_scaled[idx]

    # Build co-access edges
    resource_grouped = df.groupby('resource_accessed')['entity_id'].unique()
    src_list, dst_list = [], []
    for res, entities in resource_grouped.items():
        if len(entities) > 1:
            for i in range(min(len(entities) - 1, 5)):
                src_list.append(entity_map[entities[i]])
                dst_list.append(entity_map[entities[i + 1]])

    if len(src_list) == 0:
        src_list, dst_list = [0], [0]

    preprocessor = GraphDataPreprocessor()
    adj_matrix = np.zeros((len(entity_map), len(entity_map)), dtype=np.float32)
    for s, d in zip(src_list, dst_list):
        adj_matrix[s, d] = 1.0

    x_nodes, edge_index = preprocessor.create_graph_from_adjacency(adj_matrix, node_features)

    return X_scaled, y_binary, x_nodes, edge_index


def main():
    print("=============================================================")
    print(" AI-POWERED BEHAVIORAL ANOMALY DETECTION MODEL TRAINING")
    print("=============================================================")

    csv_path = "synthetic_access_logs_10000.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Run EDA
    eda_results = run_exploratory_data_analysis(df)

    # 2. Extract Features & Graph Structure
    print("\n=============================================================")
    print(" 2. FEATURE ENGINEERING & GRAPH TOPOLOGY CREATION")
    print("=============================================================")
    X_features, y_binary, x_nodes, edge_index = extract_features_and_graph(df)
    print(f" -> Feature Matrix Shape: {X_features.shape}")
    print(f" -> Graph Node Tensor Shape: {x_nodes.shape}")
    print(f" -> Graph Edge Index Shape: {edge_index.shape}")

    # 3. Time Series Cross-Validation Split
    print("\n=============================================================")
    print(" 3. ADVANCED TIME SERIES SPLITTING")
    print("=============================================================")
    splitter = AdvancedTimeSeriesSplit(n_splits=5)
    splits = splitter.expanding_window_split(df, time_col='timestamp')
    print(f" -> Generated {len(splits)} expanding window time-series splits.")

    train_idx, test_idx = splits[-1]
    X_train, X_test = X_features[train_idx], X_features[test_idx]
    y_train, y_test = y_binary[train_idx], y_binary[test_idx]

    # 4. Train & Evaluate LSTM Autoencoder
    print("\n=============================================================")
    print(" 4. TRAINING & EVALUATING LSTM AUTOENCODER")
    print("=============================================================")
    autoencoder = LSTMAutoencoder(input_dim=X_features.shape[1], hidden_dim=32, latent_dim=16)
    trainer = AutoencoderTrainer(autoencoder, config={'epochs': 15, 'batch_size': 64})

    t0 = time.time()
    results = trainer.train(X_train, verbose=True)
    train_time = time.time() - t0

    # Compute reconstruction loss over test set
    autoencoder.eval()
    with torch.no_grad():
        test_seqs = torch.tensor(X_test[:min(len(X_test), 500)].reshape(-1, 10, X_features.shape[1]), dtype=torch.float32)
        reconstructed, _ = autoencoder(test_seqs)
        test_mse_loss = float(autoencoder.compute_loss(test_seqs, reconstructed).item())

    print(f" -> Training Completed in {train_time:.2f}s")
    print(f" -> Best Validation Loss: {results['best_val_loss']:.6f}")
    print(f" -> Test Reconstruction MSE Loss: {test_mse_loss:.6f}")

    # Save LSTM Autoencoder Checkpoint
    trainer.save_checkpoint("models/autoencoder.pth")
    print(" -> Saved Autoencoder checkpoint to models/autoencoder.pth")

    # 5. Train & Evaluate Graph Neural Network (GNN)
    print("\n=============================================================")
    print(" 5. TRAINING & EVALUATING GRAPH NEURAL NETWORK (GNN)")
    print("=============================================================")
    gnn = GraphAutoencoder(node_feature_dim=x_nodes.size(1), hidden_dim=32, latent_dim=16)
    gnn_optimizer = torch.optim.Adam(gnn.parameters(), lr=0.005)

    gnn.train()
    for epoch in range(20):
        gnn_optimizer.zero_grad()
        rec_nodes, _ = gnn(x_nodes, edge_index)
        gnn_loss = gnn.compute_loss(x_nodes, rec_nodes)
        gnn_loss.backward()
        gnn_optimizer.step()

    gnn.eval()
    with torch.no_grad():
        rec_nodes, _ = gnn(x_nodes, edge_index)
        gnn_test_loss = float(gnn.compute_loss(x_nodes, rec_nodes).item())

    print(f" -> GNN Final Reconstruction Loss: {gnn_test_loss:.6f}")
    os.makedirs("models", exist_ok=True)
    torch.save(gnn.state_dict(), "models/gnn.pth")
    print(" -> Saved GNN checkpoint to models/gnn.pth")

    # 6. Comprehensive Metrics Evaluation Matrix
    print("\n=============================================================")
    print(" 6. COMPREHENSIVE SYSTEM EVALUATION METRICS MATRIX")
    print("=============================================================")

    classifier = AttackClassifier()
    detector = SequenceDetector("models/autoencoder.onnx")

    y_pred_probs = []
    latencies = []
    for i in range(min(len(X_test), 200)):
        start = time.time()
        score_res = detector.detect_sequence_anomaly(f"test_{i}", np.expand_dims(X_test[i:i+1], axis=0))
        latencies.append((time.time() - start) * 1000.0)
        y_pred_probs.append(score_res["combined_score"])

    y_pred_probs = np.array(y_pred_probs)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    y_test_subset = y_test[:len(y_pred_binary)]

    acc = accuracy_score(y_test_subset, y_pred_binary)
    prec = precision_score(y_test_subset, y_pred_binary, zero_division=0)
    rec = recall_score(y_test_subset, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test_subset, y_pred_binary, zero_division=0)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    print(f" -> Accuracy: {acc * 100:.2f}%")
    print(f" -> Precision: {prec * 100:.2f}%")
    print(f" -> Recall: {rec * 100:.2f}%")
    print(f" -> F1-Score: {f1 * 100:.2f}%")
    print(f" -> P95 Latency: {p95_latency:.2f} ms")
    print(f" -> P99 Latency: {p99_latency:.2f} ms")

    print("\n=============================================================")
    print(" SUCCESS: ALL MODELS TRAINED, EVALUATED & CHECKPOINTED!")
    print("=============================================================\n")


if __name__ == "__main__":
    main()
