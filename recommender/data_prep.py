"""
data_prep.py – Carregamento e preparação dos dados do MovieLens.

Responsabilidades:
  - Ler os CSVs (ratings, movies, tags, links).
  - Filtrar usuários e filmes com poucas interações.
  - Construir a matriz usuário-item (pivot).
  - Dividir os dados em treino/teste.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

# ──────────────────────────────────────────────
# Caminhos padrão
# ──────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data", "raw")


def load_data(
    data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os quatro CSVs do MovieLens e retorna DataFrames."""
    d = data_dir or _DATA_DIR
    ratings = pd.read_csv(os.path.join(d, "ratings.csv"))
    movies = pd.read_csv(os.path.join(d, "movies.csv"))
    tags = pd.read_csv(os.path.join(d, "tags.csv"))
    links = pd.read_csv(os.path.join(d, "links.csv"))
    return ratings, movies, tags, links


def filter_sparse_data(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 10,
) -> pd.DataFrame:
    """
    Remove usuários e filmes com poucas avaliações para reduzir esparsidade
    e melhorar a qualidade das recomendações.
    """
    # Filtrar usuários
    user_counts = ratings["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings["userId"].isin(valid_users)]

    # Filtrar filmes
    movie_counts = ratings["movieId"].value_counts()
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    ratings = ratings[ratings["movieId"].isin(valid_movies)]

    return ratings.reset_index(drop=True)


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói a matriz usuário × item (pivot table).
    Valores ausentes ficam como NaN para diferenciar
    'não assistido' de 'avaliação zero'.
    """
    matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    return matrix


def train_test_split(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide as avaliações em treino e teste por usuário
    (amostragem estratificada por usuário).
    Garante que cada usuário tenha pelo menos uma avaliação no treino.
    """
    rng = np.random.RandomState(random_state)
    train_indices = []
    test_indices = []

    for _, group in ratings.groupby("userId"):
        indices = group.index.tolist()
        n_test = max(1, int(len(indices) * test_size))
        # Garantir pelo menos 1 no treino
        if n_test >= len(indices):
            n_test = len(indices) - 1
        test_idx = rng.choice(indices, size=n_test, replace=False).tolist()
        train_idx = [i for i in indices if i not in test_idx]
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    train = ratings.loc[train_indices].reset_index(drop=True)
    test = ratings.loc[test_indices].reset_index(drop=True)
    return train, test


def get_movie_info(movies: pd.DataFrame) -> dict:
    """Retorna um dicionário movieId → {title, genres} para lookups rápidos."""
    info = {}
    for _, row in movies.iterrows():
        info[row["movieId"]] = {
            "title": row["title"],
            "genres": row["genres"].split("|") if pd.notna(row["genres"]) else [],
        }
    return info
