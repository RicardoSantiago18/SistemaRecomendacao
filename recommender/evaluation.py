"""
evaluation.py – Avaliação do sistema de recomendação.

Métricas implementadas:
  - MAE (Mean Absolute Error): erro médio absoluto das predições.
  - RMSE (Root Mean Squared Error): raiz do erro quadrático médio.
  - Precision@K: proporção de itens recomendados que são relevantes.
  - Recall@K: proporção de itens relevantes que foram recomendados.
  - Coverage: proporção de itens que o sistema consegue recomendar.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .user_cf import UserCFRecommender


def mae(predictions: List[float], actuals: List[float]) -> float:
    """Mean Absolute Error."""
    if not predictions:
        return 0.0
    return float(np.mean(np.abs(np.array(predictions) - np.array(actuals))))


def rmse(predictions: List[float], actuals: List[float]) -> float:
    """Root Mean Squared Error."""
    if not predictions:
        return 0.0
    return float(np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2)))


def precision_at_k(
    recommended: List[int], relevant: List[int], k: int = 10
) -> float:
    """
    Precision@K: dos K itens recomendados, quantos são relevantes?
    Um item é 'relevante' se o usuário o avaliou com nota >= 4.0.
    """
    if k == 0:
        return 0.0
    rec_at_k = recommended[:k]
    if not rec_at_k:
        return 0.0
    hits = len(set(rec_at_k) & set(relevant))
    return hits / len(rec_at_k)


def recall_at_k(
    recommended: List[int], relevant: List[int], k: int = 10
) -> float:
    """
    Recall@K: dos itens relevantes, quantos foram recomendados nos top-K?
    """
    if not relevant:
        return 0.0
    rec_at_k = recommended[:k]
    hits = len(set(rec_at_k) & set(relevant))
    return hits / len(relevant)


def coverage(
    recommender: UserCFRecommender,
    user_ids: List[int],
    n: int = 10,
    movies_df: Optional[pd.DataFrame] = None,
) -> float:
    """
    Cobertura: proporção de itens únicos que aparecem nas recomendações
    sobre o total de itens do catálogo.
    """
    all_recommended = set()
    total_items = len(recommender.user_item_matrix.columns)

    for uid in user_ids:
        recs = recommender.recommend(uid, n=n, movies_df=movies_df)
        all_recommended.update(recs["movieId"].tolist())

    return len(all_recommended) / total_items if total_items > 0 else 0.0


def evaluate_recommender(
    recommender: UserCFRecommender,
    test_data: pd.DataFrame,
    movies_df: Optional[pd.DataFrame] = None,
    k: int = 10,
    n_sample_users: Optional[int] = None,
    relevance_threshold: float = 4.0,
) -> Dict:
    """
    Avaliação completa do sistema de recomendação.

    Parâmetros
    ----------
    recommender : UserCFRecommender
        Modelo já treinado.
    test_data : pd.DataFrame
        Dados de teste com colunas [userId, movieId, rating].
    movies_df : pd.DataFrame, optional
        DataFrame de filmes.
    k : int
        Valor de K para Precision@K e Recall@K.
    n_sample_users : int, optional
        Número de usuários para avaliar (amostragem). None = todos.
    relevance_threshold : float
        Nota mínima para considerar um item como 'relevante'.

    Retorna
    -------
    dict com todas as métricas.
    """
    # Obter usuários no teste que também estão no modelo
    test_users = test_data["userId"].unique()
    model_users = set(recommender.user_item_matrix.index)
    valid_users = [u for u in test_users if u in model_users]

    if n_sample_users and n_sample_users < len(valid_users):
        rng = np.random.RandomState(42)
        valid_users = rng.choice(valid_users, size=n_sample_users, replace=False).tolist()

    all_predictions = []
    all_actuals = []
    precision_scores = []
    recall_scores = []
    all_recommended_items = set()

    per_user_results = []

    for user_id in valid_users:
        user_test = test_data[test_data["userId"] == user_id]

        # ---- Métricas de predição (MAE, RMSE) ----
        preds = []
        acts = []
        for _, row in user_test.iterrows():
            pred = recommender.predict(user_id, row["movieId"])
            preds.append(pred)
            acts.append(row["rating"])

        all_predictions.extend(preds)
        all_actuals.extend(acts)

        # ---- Métricas de ranking (Precision@K, Recall@K) ----
        # Itens relevantes no teste
        relevant_items = user_test[
            user_test["rating"] >= relevance_threshold
        ]["movieId"].tolist()

        # Gerar recomendações
        recs = recommender.recommend(user_id, n=k, movies_df=movies_df)
        recommended_items = recs["movieId"].tolist()
        all_recommended_items.update(recommended_items)

        p_at_k = precision_at_k(recommended_items, relevant_items, k)
        r_at_k = recall_at_k(recommended_items, relevant_items, k)

        precision_scores.append(p_at_k)
        recall_scores.append(r_at_k)

        user_mae = mae(preds, acts)
        user_rmse = rmse(preds, acts)

        per_user_results.append(
            {
                "userId": user_id,
                "n_test_items": len(user_test),
                "MAE": round(user_mae, 4),
                "RMSE": round(user_rmse, 4),
                "Precision@K": round(p_at_k, 4),
                "Recall@K": round(r_at_k, 4),
            }
        )

    # Métricas globais
    overall_mae = mae(all_predictions, all_actuals)
    overall_rmse = rmse(all_predictions, all_actuals)
    avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0

    # Cobertura calculada com as recomendações já geradas
    total_items = len(recommender.user_item_matrix.columns)
    cov = len(all_recommended_items) / total_items if total_items > 0 else 0.0

    results = {
        "overall": {
            "MAE": round(overall_mae, 4),
            "RMSE": round(overall_rmse, 4),
            "Precision@K": round(avg_precision, 4),
            "Recall@K": round(avg_recall, 4),
            "Coverage": round(cov, 4),
            "K": k,
            "n_users_evaluated": len(valid_users),
            "n_predictions": len(all_predictions),
        },
        "per_user": pd.DataFrame(per_user_results),
    }

    return results
