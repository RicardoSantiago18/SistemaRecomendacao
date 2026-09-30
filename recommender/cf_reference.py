"""
cf_reference.py - Script de referencia para execucao offline.

Treina o modelo, avalia e mostra exemplos de recomendacao
para 5 usuarios de teste + 1 caso cold-start.
Pode ser rodado independentemente do Streamlit.
"""

import sys
import os
import io

# Forcar UTF-8 no stdout para evitar erros de encoding no Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Garantir que o modulo recommender esteja no path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.data_prep import (
    load_data,
    filter_sparse_data,
    build_user_item_matrix,
    train_test_split,
)
from recommender.user_cf import UserCFRecommender
from recommender.evaluation import evaluate_recommender


def main():
    print("=" * 70)
    print("SISTEMA DE RECOMENDACAO DE FILMES")
    print("Filtragem Colaborativa - KNN com Distancia Manhattan")
    print("=" * 70)

    # 1. Carregar dados
    print("\n[1/5] Carregando dados...")
    ratings, movies, tags, links = load_data()
    print(f"  -> {len(ratings)} avaliacoes, {ratings['userId'].nunique()} usuarios, "
          f"{ratings['movieId'].nunique()} filmes")

    # 2. Filtrar dados esparsos
    print("\n[2/5] Filtrando dados esparsos...")
    ratings_filtered = filter_sparse_data(ratings, min_user_ratings=20, min_movie_ratings=10)
    print(f"  -> {len(ratings_filtered)} avaliacoes apos filtro, "
          f"{ratings_filtered['userId'].nunique()} usuarios, "
          f"{ratings_filtered['movieId'].nunique()} filmes")

    # 3. Dividir treino/teste
    print("\n[3/5] Dividindo treino/teste (80/20)...")
    train, test = train_test_split(ratings_filtered, test_size=0.2)
    print(f"  -> Treino: {len(train)} | Teste: {len(test)}")

    # 4. Treinar modelo
    print("\n[4/5] Treinando modelo (K=20, distancia Manhattan)...")
    matrix = build_user_item_matrix(train)
    print(f"  -> Matriz: {matrix.shape[0]} usuarios x {matrix.shape[1]} filmes")
    print(f"  -> Esparsidade: {(matrix.isna().sum().sum() / matrix.size) * 100:.1f}%")

    model = UserCFRecommender(k=20, min_common=3)
    model.fit(matrix)
    print("  -> Modelo treinado com sucesso!")

    # 5. Avaliar
    print("\n[5/5] Avaliando modelo...")
    results = evaluate_recommender(
        model, test, movies_df=movies, k=10, n_sample_users=50
    )

    print("\n" + "=" * 70)
    print("RESULTADOS DA AVALIACAO")
    print("=" * 70)
    overall = results["overall"]
    print(f"  MAE:          {overall['MAE']}")
    print(f"  RMSE:         {overall['RMSE']}")
    print(f"  Precision@10: {overall['Precision@K']}")
    print(f"  Recall@10:    {overall['Recall@K']}")
    print(f"  Coverage:     {overall['Coverage']}")
    print(f"  Usuarios avaliados: {overall['n_users_evaluated']}")
    print(f"  Total de predicoes: {overall['n_predictions']}")

    # 6. Exemplos para 5 usuarios
    print("\n" + "=" * 70)
    print("EXEMPLOS DE RECOMENDACAO (5 usuarios)")
    print("=" * 70)

    # Selecionar 5 usuarios com historicos variados
    sample_users = sorted(matrix.index.tolist())[:5]

    for uid in sample_users:
        print(f"\n{'-' * 60}")
        print(f"USUARIO {uid}")
        print(f"{'-' * 60}")

        # Historico
        history = model.get_user_history(uid, movies_df=movies)
        print(f"\n  [Historico] ({len(history)} filmes avaliados):")
        for _, row in history.head(5).iterrows():
            print(f"     * {row['rating']:.1f}  {row['title']}")
        if len(history) > 5:
            print(f"     ... e mais {len(history) - 5} filmes")

        # Vizinhos
        neighbors = model.get_neighbors_info(uid)
        print(f"\n  [Vizinhos mais proximos] ({len(neighbors)}):")
        for nb in neighbors[:3]:
            print(f"     User {nb['neighbor_id']}: dist={nb['distance']}, "
                  f"sim={nb['similarity']}")

        # Recomendacoes
        recs = model.recommend(uid, n=10, movies_df=movies)
        print(f"\n  [Top 10 Recomendacoes]:")
        for _, row in recs.iterrows():
            print(f"     * {row['predicted_rating']:.2f}  {row['title']}")

    # 7. Caso cold-start
    print(f"\n{'-' * 60}")
    print("CASO COLD-START (usuario novo, ID=99999)")
    print(f"{'-' * 60}")
    cold_recs = model.recommend(99999, n=10, movies_df=movies)
    print("\n  [Recomendacoes por popularidade]:")
    for _, row in cold_recs.iterrows():
        print(f"     * {row['predicted_rating']:.2f}  {row['title']}")

    print(f"\n{'=' * 70}")
    print("Execucao concluida com sucesso!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
