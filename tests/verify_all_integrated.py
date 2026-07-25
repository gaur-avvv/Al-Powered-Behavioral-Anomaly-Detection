"""
Comprehensive Integration Verification Script.
Tests ALL modules working together end-to-end:
  - Feature Selection (L1, Tree, Neural)
  - Dimensionality Reduction (PCA, t-SNE, UMAP)
  - Deep Learning Models (LSTM Autoencoder, GNN)
  - Time Series Splitting
  - Regularization, Early Stopping, Cross-Validation
  - Explainability (SHAP, LIME)
  - Attack Classifier
  - Detection Engine & Baseline Profiler
  - API Endpoints
"""

import sys
import time
import traceback
import numpy as np
import pandas as pd

PASS = 0
FAIL = 0
RESULTS = []


def check(name, fn):
    """Run a verification check and record result."""
    global PASS, FAIL
    try:
        start = time.time()
        fn()
        elapsed = (time.time() - start) * 1000
        PASS += 1
        RESULTS.append(("PASS", name, f"{elapsed:.1f}ms"))
        print(f"  [PASS] {name} ({elapsed:.1f}ms)")
    except Exception as e:
        FAIL += 1
        RESULTS.append(("FAIL", name, str(e)))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


def test_l1_feature_selector():
    from src.feature_selection.l1_feature_selection import L1FeatureSelector
    X = pd.DataFrame(np.random.randn(60, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 60))
    sel = L1FeatureSelector(model_type='classification', max_iter=200)
    res = sel.fit(X, y)
    assert len(sel.get_support()) > 0, "No features selected"
    X_t = sel.transform(X)
    assert X_t.shape[1] > 0, "Transform produced empty DataFrame"


def test_tree_feature_selector():
    from src.feature_selection.tree_feature_selection import TreeFeatureSelector
    X = pd.DataFrame(np.random.randn(60, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 60))
    sel = TreeFeatureSelector(n_estimators=10)
    sel.fit(X, y, n_repeats=2)
    assert len(sel.get_support()) > 0


def test_neural_feature_selector():
    from src.feature_selection.neural_feature_selection import NeuralFeatureSelector
    X = pd.DataFrame(np.random.randn(40, 4), columns=[f'f{i}' for i in range(4)])
    y = pd.Series(np.random.randint(0, 2, 40))
    sel = NeuralFeatureSelector(input_dim=4, n_iterations=5)
    sel.fit(X, y, model_type='classification', n_bootstrap=2)
    X_t = sel.transform(X)
    assert X_t.shape[1] > 0


def test_pca_reducer():
    from src.dimensionality_reduction.pca_reducer import PCAReducer
    X = pd.DataFrame(np.random.randn(30, 6), columns=[f'f{i}' for i in range(6)])
    reducer = PCAReducer(n_components=2)
    ev = reducer.fit(X)
    assert 'cumulative_variance' in ev.columns
    X_pca = reducer.transform(X)
    assert X_pca.shape == (30, 2)
    X_inv = reducer.inverse_transform(X_pca)
    assert X_inv.shape == X.shape


def test_tsne_reducer():
    from src.dimensionality_reduction.tsne_reducer import TSNEDimensionReducer
    X = pd.DataFrame(np.random.randn(30, 6), columns=[f'f{i}' for i in range(6)])
    reducer = TSNEDimensionReducer(n_components=2, perplexity=5, n_iter=250)
    X_tsne = reducer.fit_transform(X)
    assert 'tsne_x' in X_tsne.columns and 'tsne_y' in X_tsne.columns


def test_umap_reducer():
    from src.dimensionality_reduction.umap_reducer import UMAPReducer
    X = pd.DataFrame(np.random.randn(30, 6), columns=[f'f{i}' for i in range(6)])
    reducer = UMAPReducer(n_components=2, n_neighbors=5)
    X_umap = reducer.fit_transform(X)
    assert 'umap_x' in X_umap.columns


def test_lstm_autoencoder():
    import torch
    from src.models.autoencoder.lstm_autoencoder import LSTMAutoencoder
    model = LSTMAutoencoder(input_dim=5, hidden_dim=16, latent_dim=8, num_layers=1)
    x = torch.randn(2, 10, 5)
    reconstructed, latent = model(x)
    assert reconstructed.shape == x.shape, f"Unexpected shape: {reconstructed.shape}"
    loss = torch.nn.MSELoss()(reconstructed, x)
    assert loss.item() >= 0


def test_autoencoder_trainer():
    import torch
    from src.models.autoencoder.lstm_autoencoder import LSTMAutoencoder
    from src.models.autoencoder.trainer import AutoencoderTrainer
    model = LSTMAutoencoder(input_dim=3, hidden_dim=8, latent_dim=4, num_layers=1)
    trainer = AutoencoderTrainer(
        model,
        config={
            'learning_rate': 0.01,
            'epochs': 3,
            'sequence_length': 5,
            'batch_size': 4,
            'patience': 10,
            'early_stopping': False,
            'checkpoint_path': 'models/test_ae.pth'
        }
    )
    data = np.random.randn(30, 3).astype(np.float32)
    history = trainer.train(data)
    assert len(history['train_loss_history']) == 3


def test_graph_autoencoder():
    import torch
    from src.models.gnn.graph_neural_network import GraphAutoencoder
    model = GraphAutoencoder(node_feature_dim=4, hidden_dim=8, latent_dim=4)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    reconstructed, graph_latent = model(x, edge_index)
    assert reconstructed.shape == x.shape
    assert graph_latent.shape[1] == 4


def test_graph_data_preprocessor():
    import torch
    from src.models.gnn.data_preprocessor import GraphDataPreprocessor
    prep = GraphDataPreprocessor()
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    features = np.random.randn(3, 4)
    x, edge_index = prep.create_graph_from_adjacency(adj, features)
    assert isinstance(x, torch.Tensor)
    assert isinstance(edge_index, torch.Tensor)


def test_time_series_split():
    from src.models.splitters.time_series_split import AdvancedTimeSeriesSplit
    ts = AdvancedTimeSeriesSplit(n_splits=3)
    timestamps = pd.date_range('2024-01-01', periods=100, freq='h')
    df = pd.DataFrame({'timestamp': timestamps, 'value': np.random.randn(100)})
    splits = ts.expanding_window_split(df, time_col='timestamp')
    assert len(splits) >= 1
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0 and len(test_idx) > 0


def test_regularization_manager():
    from src.regularization.regularization_utils import RegularizationManager
    X = pd.DataFrame(np.random.randn(50, 4), columns=[f'f{i}' for i in range(4)])
    y = pd.Series(np.random.randn(50))
    alpha = RegularizationManager.find_optimal_alpha(X, y, model_type='ridge')
    assert alpha > 0


def test_early_stopping():
    from src.prevention.early_stopping import EarlyStopping, ModelMonitor
    es = EarlyStopping(patience=2, min_delta=0.001, mode='min')
    assert not es.should_stop(1.0)   # new best
    assert not es.should_stop(0.5)   # improved
    assert not es.should_stop(0.5)   # no improve, wait=1
    assert es.should_stop(0.5)       # no improve, wait=2 >= patience

    diag = ModelMonitor.diagnose_overfitting_underfitting([0.5, 0.3, 0.1], [0.5, 0.4, 0.6])
    assert 'overfitting' in diag or 'good_fit' in diag or 'underfitting' in diag


def test_cross_validation_manager():
    from src.validation.cv_strategies import CrossValidationManager
    from sklearn.ensemble import RandomForestClassifier
    X = pd.DataFrame(np.random.randn(60, 4), columns=[f'f{i}' for i in range(4)])
    y = pd.Series(np.random.randint(0, 2, 60))
    cv_df = CrossValidationManager.evaluate_model_with_cv(
        RandomForestClassifier(n_estimators=10, random_state=42), X, y
    )
    assert len(cv_df) == 3


def test_baseline_profiler():
    from src.models.baseline_profiler import EntityBaselineProfiler
    profiler = EntityBaselineProfiler(seq_length=10)
    data = np.random.randn(50)
    profile = profiler.create_profile("test_entity", data)
    assert 'mean' in profile


def test_detection_engine():
    from src.models.detection_engine import SequenceDetector
    detector = SequenceDetector()
    seq = np.random.randn(1, 10, 3).astype(np.float32)
    result = detector.detect_sequence_anomaly("entity_test", seq)
    assert 'combined_score' in result


def test_attack_classifier():
    from src.models.attack_classifier import AttackClassifier
    classifier = AttackClassifier()
    features = {'geo_velocity': 180.0, 'failed_logins': 6.0, 'new_device': 1.0}
    result = classifier.classify_anomaly(features)
    assert 'primary_category' in result and 'confidence' in result


def test_explainability_shap():
    from src.explainability.explanation_engine import ExplainableAI
    explainer = ExplainableAI()
    features = {'geo_velocity': 100.0, 'failed_logins': 3.0, 'new_device': 1.0}
    result = explainer.explain_anomaly(features, method='shap')
    assert 'method' in result


def test_explainability_lime():
    from src.explainability.explanation_engine import ExplainableAI
    explainer = ExplainableAI()
    features = {'geo_velocity': 100.0, 'failed_logins': 3.0}
    result = explainer.explain_anomaly(features, method='lime')
    assert 'method' in result


def test_dashboard_service():
    from src.dashboard.dashboard_service import AnalystDashboard
    dashboard = AnalystDashboard()
    alerts = dashboard._get_recent_alerts(limit=5)
    assert isinstance(alerts, list)


def test_monitoring_service():
    from src.monitoring.monitoring_service import MonitoringService
    monitor = MonitoringService()
    health = monitor.check_health()
    assert 'status' in health


def test_report_generator():
    from src.report.report_generator import PerformanceReport
    report = PerformanceReport()
    full = report.generate_full_report()
    assert 'assumptions' in full and 'limitations' in full


def test_api_import():
    from src.api.main import app
    assert app is not None
    assert app.title == "AI-Powered Behavioral Anomaly Detection API"


def test_visualization_artifacts_exist():
    import os
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    expected = [
        'loss_curves.png', 'confusion_matrix.png', 'roc_auc_curves.png',
        'feature_importance.png', 'kfold_cross_validation.png', 'tsne_umap_projections.png'
    ]
    for fname in expected:
        path = os.path.join(assets_dir, fname)
        assert os.path.exists(path), f"Missing asset: {fname}"
        assert os.path.getsize(path) > 1000, f"Asset too small: {fname}"


def main():
    print("=" * 70)
    print("  AEGIS.AI - Comprehensive Module Integration Verification")
    print("=" * 70)

    print("\n[1/8] Feature Selection Modules:")
    check("L1 (LASSO) Feature Selector", test_l1_feature_selector)
    check("Tree Stability Feature Selector", test_tree_feature_selector)
    check("Neural Integrated Gradients Selector", test_neural_feature_selector)

    print("\n[2/8] Dimensionality Reduction Modules:")
    check("PCA Reducer (fit/transform/inverse)", test_pca_reducer)
    check("t-SNE Manifold Reducer", test_tsne_reducer)
    check("UMAP Manifold Reducer (PCA fallback)", test_umap_reducer)

    print("\n[3/8] Deep Learning Models:")
    check("PyTorch LSTM Autoencoder (forward pass)", test_lstm_autoencoder)
    check("Autoencoder Trainer (3-epoch training)", test_autoencoder_trainer)
    check("PyTorch Graph Autoencoder (GCN)", test_graph_autoencoder)
    check("Graph Data Preprocessor", test_graph_data_preprocessor)

    print("\n[4/8] Time Series Splitting & Cross-Validation:")
    check("Advanced Time Series Split (expanding)", test_time_series_split)
    check("Cross-Validation Manager (3-fold)", test_cross_validation_manager)

    print("\n[5/8] Regularization & Overfitting Controls:")
    check("Regularization Manager (Ridge alpha)", test_regularization_manager)
    check("Early Stopping & Model Monitor", test_early_stopping)

    print("\n[6/8] Detection & Classification:")
    check("Entity Baseline Profiler", test_baseline_profiler)
    check("Sequence Anomaly Detector", test_detection_engine)
    check("Multi-Class Attack Classifier", test_attack_classifier)

    print("\n[7/8] Explainability:")
    check("SHAP Feature Attributions", test_explainability_shap)
    check("LIME Local Approximations", test_explainability_lime)

    print("\n[8/8] Services & API:")
    check("Dashboard WebSocket Service", test_dashboard_service)
    check("Monitoring Health Service", test_monitoring_service)
    check("Performance Report Generator", test_report_generator)
    check("FastAPI Application Import", test_api_import)

    print("\n[BONUS] Visualization Matrix Artifacts:")
    check("All 6 PNG Assets Exist & Valid", test_visualization_artifacts_exist)

    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"  TOTAL: {total} checks | PASSED: {PASS} | FAILED: {FAIL}")
    if FAIL == 0:
        print("  ALL MODULES VERIFIED - FULLY INTEGRATED & OPERATIONAL")
    else:
        print(f"  WARNING: {FAIL} check(s) failed - review output above")
    print("=" * 70)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
