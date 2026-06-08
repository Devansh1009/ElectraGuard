"""
LSTM / Neural Network Detection Engine

Lightweight implementation using scikit-learn's MLPClassifier
for Streamlit Cloud compatibility (no TensorFlow dependency).
Follows the same pipeline as Kocaman & Tümen (2020).
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def clean_data(matrix):
    """Clean: NaN → column mean, negatives → 0."""
    result = matrix.copy().astype(float)
    for j in range(result.shape[1]):
        col = result[:, j]
        valid = col[~np.isnan(col)]
        mean_val = valid.mean() if len(valid) > 0 else 0
        col[np.isnan(col)] = mean_val
        col[col < 0] = 0
        result[:, j] = col
    return result


def normalize_data(matrix):
    """Min-Max normalize each row to [0, 1]."""
    result = matrix.copy()
    for i in range(result.shape[0]):
        row = result[i]
        mn, mx = row.min(), row.max()
        if mx - mn > 0:
            result[i] = (row - mn) / (mx - mn)
        else:
            result[i] = np.zeros_like(row)
    return result


def create_sequences(data, labels, window_size=7):
    """Create sliding window sequences."""
    X, y = [], []
    for i in range(data.shape[0]):
        series = data[i]
        label = labels[i]
        for j in range(len(series) - window_size + 1):
            X.append(series[j:j + window_size])
            y.append(label)
    return np.array(X), np.array(y)


def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, F1."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def find_optimal_threshold(y_true, y_probs):
    """Find optimal threshold via Youden's J statistic."""
    best_t, best_j = 0.5, -1
    for t in np.arange(0.1, 0.91, 0.05):
        preds = (y_probs >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = t
    return best_t


def train_and_evaluate(time_series_data, labels, window_size=7, n_folds=5,
                       progress_callback=None):
    """
    5-fold cross-validated neural network training.

    Uses scikit-learn MLPClassifier with architecture similar to the
    LSTM paper: hidden layers (128, 64) with relu activation.

    Args:
        time_series_data: numpy array (n_consumers, n_days)
        labels: numpy array (n_consumers,) of 0/1
        progress_callback: callable(fold, n_folds, metrics_dict)

    Returns:
        dict with fold_metrics, avg_metrics, consumer_probs, threshold
    """
    data = clean_data(time_series_data)
    data = normalize_data(data)

    if data.shape[1] < window_size:
        return None

    n_consumers = data.shape[0]
    consumer_probs = np.full(n_consumers, 0.5)

    X_all, y_all = create_sequences(data, labels, window_size)
    seqs_per_consumer = data.shape[1] - window_size + 1

    if len(X_all) < 10:
        return None

    # Ensure both classes present
    if len(np.unique(y_all)) < 2:
        return None

    actual_folds = min(n_folds, len(X_all))
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        # Skip folds with only one class
        if len(np.unique(y_train)) < 2:
            continue

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=42 + fold_idx,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val).astype(float)
        threshold = find_optimal_threshold(y_val, val_probs)
        val_preds = (val_probs >= threshold).astype(int)
        metrics = compute_metrics(y_val, val_preds)
        metrics["threshold"] = threshold
        fold_metrics.append(metrics)

        if progress_callback:
            progress_callback(fold_idx + 1, actual_folds, metrics)

    if not fold_metrics:
        return None

    # Final model on all data
    final_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    final_model.fit(X_all, y_all)
    all_probs = final_model.predict_proba(X_all)[:, 1] if hasattr(final_model, "predict_proba") else final_model.predict(X_all).astype(float)

    # Average per consumer
    for i in range(n_consumers):
        start = i * seqs_per_consumer
        end = start + seqs_per_consumer
        if end <= len(all_probs):
            consumer_probs[i] = all_probs[start:end].mean()

    avg_metrics = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        avg_metrics[key] = np.mean([m[key] for m in fold_metrics])
    avg_metrics["threshold"] = np.mean([m["threshold"] for m in fold_metrics])

    return {
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics,
        "consumer_probs": consumer_probs,
        "threshold": avg_metrics["threshold"],
    }
