#!/usr/bin/env python3
"""Offline release checks; no download, training, or modification of saved results."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, brier_score_loss
from pipeline import compute_topological_summary, compute_persistence_entropy
from generate_persistence_images import compute_persistence_image

ROOT = Path(__file__).resolve().parent

def main():
    uniform = compute_topological_summary(np.zeros((3, 3), dtype=np.float32))
    assert uniform['h0_count'] == uniform['h1_count'] == 0
    ring = np.zeros((3, 3), dtype=np.float32)
    ring[1, 1] = 1
    features = compute_topological_summary(ring)
    assert features['h1_count'] == 1 and np.isclose(features['h1_total_persistence'], 1)
    assert np.isclose(compute_persistence_entropy(np.array([1., 1.])), 1)
    assert compute_persistence_entropy(np.array([])) == 0
    assert not compute_persistence_image(np.empty((0, 2))).any()
    assert np.isfinite(compute_persistence_image(np.array([[.2, .6]]))).all()
    saved = np.load(ROOT/'results/topo_cnn_predictions.npz', allow_pickle=False)
    y, prob = saved['y_true'], saved['y_prob']
    assert y.shape == prob.shape == (4000,)
    assert np.array_equal(np.bincount(y), [2000, 2000])
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    pred = prob >= .5
    metrics = {
        'Test_Accuracy': accuracy_score(y, pred),
        'Test_ROC_AUC': roc_auc_score(y, prob),
        'Test_F1_Score': f1_score(y, pred),
        'Test_Recall': recall_score(y, pred),
        'Test_Brier_Score': brier_score_loss(y, prob),
    }
    table = pd.read_csv(ROOT/'figures/classification_metrics_benchmark_with_cnn.csv')
    row = table.loc[table.Model == 'TopoCNN (2D Persistence Images)'].iloc[0]
    for name, value in metrics.items():
        assert np.isclose(value, row[name], atol=1e-7), (name, value, row[name])
    print('Passed: cubical feature checks and all five saved TopoCNN prediction metrics.')
    print(metrics)

if __name__ == '__main__':
    main()
