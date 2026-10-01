"""
user_cf.py – Filtragem Colaborativa baseada em Usuário com KNN + distância Manhattan.

Implementa o algoritmo completo:
  1. Normaliza a matriz (subtrai média por usuário).
  2. Calcula distância Manhattan entre usuários.
  3. Encontra os K vizinhos mais próximos.
  4. Prediz avaliações via média ponderada (peso = 1/distância).
  5. Gera recomendações excluindo itens já avaliados.
  6. Fallback por popularidade para cold-start.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class UserCFRecommender:
    """
    Sistema de Recomendação por Filtragem Colaborativa Baseada em Usuário.
    Usa K-Nearest Neighbors com distância Manhattan.
    """

    def __init__(self, k: int = 20, min_common: int = 3):
        """
        Parâmetros
        ----------
        k : int
            Número de vizinhos mais próximos a considerar.
        min_common : int
            Número mínimo de itens em comum para considerar dois usuários
            como vizinhos válidos.
        """
        self.k = k
        self.min_common = min_common
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.user_means: Optional[pd.Series] = None
        self.normalized_matrix: Optional[pd.DataFrame] = None
        self.global_mean: float = 0.0
        self._popularity_ranking: Optional[pd.Series] = None
        self._neighbors_cache: Dict[int, List[Tuple[int, float]]] = {}

    def fit(self, user_item_matrix: pd.DataFrame) -> "UserCFRecommender":
        """
        Treina o modelo com a matriz usuário-item.

        Parâmetros
        ----------
        user_item_matrix : pd.DataFrame
            Matriz com userId como índice, movieId como colunas, ratings como valores.
        """
        self.user_item_matrix = user_item_matrix.copy()
        self._neighbors_cache = {}

        # Média de cada usuário (ignora NaN)
        self.user_means = self.user_item_matrix.mean(axis=1)

        # Média global (para cold-start)
        self.global_mean = self.user_item_matrix.values[
            ~np.isnan(self.user_item_matrix.values)
        ].mean()

        # Matriz normalizada (rating - média do usuário)
        self.normalized_matrix = self.user_item_matrix.sub(
            self.user_means, axis=0
        )

        # Ranking de popularidade (para cold-start)
        # Média ponderada: filmes com mais avaliações e médias altas
        movie_mean = self.user_item_matrix.mean(axis=0)
        movie_count = self.user_item_matrix.count(axis=0)
        # Score bayesiano simples: (count / (count + m)) * mean + (m / (count + m)) * global
        m = 50  # parâmetro de regularização
        self._popularity_ranking = (
            (movie_count / (movie_count + m)) * movie_mean
            + (m / (movie_count + m)) * self.global_mean
        )
        self._popularity_ranking = self._popularity_ranking.sort_values(
            ascending=False
        )

        return self

    def _manhattan_distance(
        self, user_a: pd.Series, user_b: pd.Series
    ) -> Tuple[float, int]:
        """
        Calcula a distância Manhattan entre dois usuários,
        considerando apenas os itens avaliados por ambos.

        Retorna (distância, n_common).
        """
        # Máscara de itens avaliados por ambos
        mask = user_a.notna() & user_b.notna()
        n_common = mask.sum()

        if n_common == 0:
            return np.inf, 0

        dist = np.abs(user_a[mask] - user_b[mask]).sum()
        # Normalizar pela quantidade de itens em comum
        dist = dist / n_common

        return dist, n_common

    def _get_neighbors(self, target_user_id: int) -> List[Tuple[int, float]]:
        """
        Encontra os K vizinhos mais próximos do usuário alvo.

        Retorna lista de (userId, distância) ordenada por distância.
        """
        if target_user_id in self._neighbors_cache:
            return self._neighbors_cache[target_user_id]

        if target_user_id not in self.user_item_matrix.index:
            return []

        target = self.user_item_matrix.loc[target_user_id]
        distances = []

        for other_id in self.user_item_matrix.index:
            if other_id == target_user_id:
                continue
            other = self.user_item_matrix.loc[other_id]
            dist, n_common = self._manhattan_distance(target, other)

            if n_common >= self.min_common and dist < np.inf:
                distances.append((other_id, dist))

        # Ordenar por distância (menor = mais similar)
        distances.sort(key=lambda x: x[1])

        neighbors = distances[: self.k]
        self._neighbors_cache[target_user_id] = neighbors
        return neighbors

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prediz a avaliação do usuário para um filme específico.

        Usa média ponderada dos vizinhos (peso = 1 / distância).
        """
        if self.user_item_matrix is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        # Se o usuário já avaliou, retorna a avaliação real
        if (
            user_id in self.user_item_matrix.index
            and movie_id in self.user_item_matrix.columns
        ):
            val = self.user_item_matrix.loc[user_id, movie_id]
            if pd.notna(val):
                return val

        # Cold-start: usuário desconhecido
        if user_id not in self.user_item_matrix.index:
            return self.global_mean

        neighbors = self._get_neighbors(user_id)

        if not neighbors:
            return self.user_means.get(user_id, self.global_mean)

        # Média ponderada com normalização
        weighted_sum = 0.0
        weight_total = 0.0

        for neighbor_id, dist in neighbors:
            if movie_id not in self.normalized_matrix.columns:
                continue
            neighbor_rating = self.normalized_matrix.loc[neighbor_id, movie_id]
            if pd.isna(neighbor_rating):
                continue

            # Peso inversamente proporcional à distância
            weight = 1.0 / (dist + 1e-8)  # epsilon para evitar divisão por zero
            weighted_sum += weight * neighbor_rating
            weight_total += weight

        if weight_total == 0:
            return self.user_means.get(user_id, self.global_mean)

        prediction = self.user_means[user_id] + (weighted_sum / weight_total)

        # Clip para o intervalo válido [0.5, 5.0]
        return float(np.clip(prediction, 0.5, 5.0))

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Gera N recomendações para o usuário, excluindo itens já avaliados.

        Parâmetros
        ----------
        user_id : int
            ID do usuário.
        n : int
            Número de recomendações.
        movies_df : pd.DataFrame, optional
            DataFrame de filmes para incluir títulos e gêneros.

        Retorna
        -------
        pd.DataFrame com colunas [movieId, predicted_rating, title, genres].
        """
        if self.user_item_matrix is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")

        # Cold-start: retorna filmes populares
        if user_id not in self.user_item_matrix.index:
            return self._cold_start_recommendations(n, movies_df)

        # Itens já avaliados pelo usuário
        user_ratings = self.user_item_matrix.loc[user_id]
        seen_movies = set(user_ratings.dropna().index.tolist())
        candidate_movies = [
            m for m in self.user_item_matrix.columns if m not in seen_movies
        ]

        if not candidate_movies:
            return pd.DataFrame(
                columns=["movieId", "predicted_rating", "title", "genres"]
            )

        neighbors = self._get_neighbors(user_id)
        if neighbors:
            neighbor_ids, distances = zip(*neighbors)
            weights = pd.Series(
                [1.0 / (dist + 1e-8) for dist in distances],
                index=neighbor_ids,
            )
            neighbor_ratings = self.normalized_matrix.loc[
                list(neighbor_ids), candidate_movies
            ]
            weighted_sum = neighbor_ratings.mul(weights, axis=0).sum(axis=0)
            weight_total = neighbor_ratings.notna().mul(weights, axis=0).sum(axis=0)
            predictions = (
                weighted_sum.div(weight_total.replace(0, np.nan))
                .add(self.user_means[user_id])
                .fillna(self.user_means[user_id])
                .clip(0.5, 5.0)
            )
        else:
            predictions = pd.Series(
                self.user_means[user_id], index=candidate_movies
            )

        top_n = predictions.sort_values(ascending=False, kind="stable").head(n)
        result = top_n.rename_axis("movieId").reset_index(name="predicted_rating")
        result["predicted_rating"] = result["predicted_rating"].round(2)

        # Adicionar info dos filmes
        if movies_df is not None:
            result = result.merge(
                movies_df[["movieId", "title", "genres"]], on="movieId", how="left"
            )

        return result

    def _cold_start_recommendations(
        self, n: int, movies_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Recomendações para cold-start (usuário sem histórico).
        Usa ranking de popularidade bayesiano.
        """
        top_movies = self._popularity_ranking.head(n)

        result = pd.DataFrame(
            {
                "movieId": top_movies.index,
                "predicted_rating": top_movies.values.round(2),
            }
        )

        if movies_df is not None:
            result = result.merge(
                movies_df[["movieId", "title", "genres"]], on="movieId", how="left"
            )

        return result

    def get_user_history(
        self, user_id: int, movies_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Retorna o histórico de avaliações do usuário.
        """
        if (
            self.user_item_matrix is None
            or user_id not in self.user_item_matrix.index
        ):
            return pd.DataFrame(columns=["movieId", "rating", "title", "genres"])

        user_ratings = self.user_item_matrix.loc[user_id].dropna()
        result = pd.DataFrame(
            {"movieId": user_ratings.index, "rating": user_ratings.values}
        )
        result = result.sort_values("rating", ascending=False).reset_index(drop=True)

        if movies_df is not None:
            result = result.merge(
                movies_df[["movieId", "title", "genres"]], on="movieId", how="left"
            )

        return result

    def get_neighbors_info(
        self, user_id: int
    ) -> List[dict]:
        """
        Retorna informações sobre os vizinhos mais próximos do usuário.
        Útil para debug e demonstração.
        """
        neighbors = self._get_neighbors(user_id)
        info = []
        for neighbor_id, dist in neighbors:
            similarity = 1.0 / (1.0 + dist)  # Converter distância em similaridade [0,1]
            info.append(
                {
                    "neighbor_id": neighbor_id,
                    "distance": round(dist, 4),
                    "similarity": round(similarity, 4),
                }
            )
        return info
