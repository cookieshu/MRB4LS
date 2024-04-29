import numpy as np
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool


def recall(y_trues, y_scores, k):
    assert y_trues.shape == y_scores.shape
    assert len(y_trues.shape) == 2
    orders = np.argsort(y_scores, axis=-1)[:, ::-1][:, :k]
    return np.mean(
        np.sum(np.take_along_axis(y_trues, orders, axis=-1), axis=-1) /
        np.sum(y_trues, axis=-1))

def MAP(y_trues, y_scores, k):
    assert y_trues.shape == y_scores.shape
    assert len(y_trues.shape) == 2
    average_precisions = []
    for y_true, y_score in zip(y_trues, y_scores):
        sorted_indices = np.argsort(y_score)[::-1]
        sorted_y_true = y_true[sorted_indices]
        precision_at_k = np.cumsum(sorted_y_true[:k]) / np.arange(1, k + 1)
        average_precision = np.sum(precision_at_k * sorted_y_true[:k]) / np.sum(sorted_y_true)

        average_precisions.append(average_precision)

    mean_average_precision = np.mean(average_precisions)
    return mean_average_precision


def mrr(y_trues, y_scores):
    assert y_trues.shape == y_scores.shape
    assert len(y_trues.shape) == 2
    orders = np.argsort(y_scores, axis=-1)[:, ::-1]
    y_trues = np.take_along_axis(y_trues, orders, axis=-1)
    rr_scores = y_trues / (np.arange(y_trues.shape[1]) + 1)
    return np.mean(np.sum(rr_scores, axis=-1) / np.sum(y_trues, axis=-1))


def fast_roc_auc_score(y_trues, y_scores, num_processes=None):
    with Pool(processes=num_processes) as pool:
        return np.mean(pool.starmap(roc_auc_score, zip(y_trues, y_scores)))
