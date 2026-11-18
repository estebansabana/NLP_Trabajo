"""Métricas de recuperación de información (sección 9.7 del notebook)."""
import numpy as np


def precision_at_k(ranking, relevantes, k=10):
    return sum(d in relevantes for d in ranking[:k]) / k


def ap_at_k(ranking, relevantes, k=10):
    """AP@k normalizada por mín(R, k), R = relevantes conocidos."""
    aciertos, s = 0, 0.0
    for i, d in enumerate(ranking[:k], 1):
        if d in relevantes:
            aciertos += 1
            s += aciertos / i
    return s / min(len(relevantes), k) if relevantes else 0.0


def ndcg_at_k(ranking, relevantes, k=10):
    dcg = sum(1 / np.log2(i + 1) for i, d in enumerate(ranking[:k], 1) if d in relevantes)
    idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(relevantes), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def rr_at_k(ranking, relevantes, k=10):
    return next((1 / i for i, d in enumerate(ranking[:k], 1) if d in relevantes), 0.0)
