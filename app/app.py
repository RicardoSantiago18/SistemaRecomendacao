"""
app.py – Interface Streamlit para o Sistema de Recomendação de Filmes.

Funcionalidades:
  - Visão geral do dataset.
  - Consulta de histórico de usuários.
  - Geração de recomendações personalizadas.
  - Avaliação interativa de filmes (simulada).
  - Caso cold-start (usuário sem histórico).
  - Resultados de avaliação do modelo.
"""

import sys
import os

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

from recommender.data_prep import (
    load_data,
    filter_sparse_data,
    build_user_item_matrix,
    train_test_split,
)
from recommender.user_cf import UserCFRecommender
from recommender.evaluation import evaluate_recommender


# ──────────────────────────────────────────────
# Configuração da página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema de Recomendação de Filmes",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# CSS customizado
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Tema geral */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: #a8b2d1;
        font-size: 1.1rem;
    }

    /* Cards de métrica */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(233, 69, 96, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card .label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        color: #e94560;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* Card de filme */
    .movie-card {
        background: linear-gradient(145deg, #1e1e3a, #252550);
        border: 1px solid rgba(233, 69, 96, 0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.2s ease;
    }
    .movie-card:hover {
        border-color: rgba(233, 69, 96, 0.5);
        transform: translateX(4px);
    }
    .movie-title {
        color: #ccd6f6;
        font-size: 1rem;
        font-weight: 600;
    }
    .movie-genres {
        color: #8892b0;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }
    .movie-rating {
        color: #ffd700;
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* Seção de vizinhos */
    .neighbor-badge {
        background: rgba(233, 69, 96, 0.15);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        display: inline-block;
        margin: 0.2rem;
        color: #ccd6f6;
        font-size: 0.85rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }

    /* Info boxes */
    .info-box {
        background: rgba(233, 69, 96, 0.1);
        border-left: 4px solid #e94560;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .info-box p {
        color: #ccd6f6;
        margin: 0;
    }

    /* Cold start banner */
    .cold-start-banner {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .cold-start-banner h3 {
        color: #e94560;
    }
    .cold-start-banner p {
        color: #a8b2d1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Cache: carregar e processar dados
# ──────────────────────────────────────────────
@st.cache_data
def load_all_data():
    """Carrega e processa todos os dados."""
    ratings, movies, tags, links = load_data()
    ratings_filtered = filter_sparse_data(ratings, min_user_ratings=20, min_movie_ratings=10)
    train, test = train_test_split(ratings_filtered, test_size=0.2)
    return ratings, ratings_filtered, movies, tags, links, train, test


@st.cache_resource
def train_model(_train: pd.DataFrame, k: int = 20):
    """Treina o modelo de recomendação (cached)."""
    matrix = build_user_item_matrix(_train)
    model = UserCFRecommender(k=k, min_common=3)
    model.fit(matrix)
    return model, matrix


@st.cache_data
def run_evaluation(_model_placeholder, _test: pd.DataFrame, _movies: pd.DataFrame, k: int):
    """Executa a avaliação (cached)."""
    # Re-treinar o modelo aqui pois _model não é hashable
    ratings, movies, tags, links = load_data()
    ratings_filtered = filter_sparse_data(ratings, min_user_ratings=20, min_movie_ratings=10)
    train, test = train_test_split(ratings_filtered, test_size=0.2)
    matrix = build_user_item_matrix(train)
    model = UserCFRecommender(k=20, min_common=3)
    model.fit(matrix)
    return evaluate_recommender(model, test, movies_df=movies, k=k, n_sample_users=50)


# ──────────────────────────────────────────────
# Helpers de renderização
# ──────────────────────────────────────────────
def render_stars(rating: float) -> str:
    """Renderiza estrelas baseado na nota."""
    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def render_metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """


def render_movie_card(title: str, rating: float, genres: str, is_prediction: bool = False) -> str:
    stars = render_stars(rating)
    rating_label = "Predição" if is_prediction else "Avaliação"
    return f"""
    <div class="movie-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="movie-title">{title}</div>
                <div class="movie-genres">🎭 {genres}</div>
            </div>
            <div style="text-align: right;">
                <div class="movie-rating">{stars} {rating:.1f}</div>
                <div style="color: #8892b0; font-size: 0.75rem;">{rating_label}</div>
            </div>
        </div>
    </div>
    """


# ──────────────────────────────────────────────
# Carregamento de dados
# ──────────────────────────────────────────────
ratings, ratings_filtered, movies, tags, links, train, test = load_all_data()
model, matrix = train_model(train, k=20)

user_ids = sorted(matrix.index.tolist())


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🎬 Sistema de Recomendação de Filmes</h1>
        <p>Filtragem Colaborativa baseada em Usuário · KNN · Distância Manhattan</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Painel de Controle")
    st.markdown("---")

    page = st.radio(
        "📌 Navegação",
        [
            "📊 Visão Geral",
            "🔍 Consultar Usuário",
            "🎬 Recomendações",
            "❄️ Cold Start",
            "⭐ Avaliar Filmes",
            "📈 Resultados da Avaliação",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros do Modelo")
    st.markdown(f"- **K vizinhos:** 20")
    st.markdown(f"- **Distância:** Manhattan")
    st.markdown(f"- **Min. itens comuns:** 3")
    st.markdown(f"- **Dataset:** MovieLens Small")



# ──────────────────────────────────────────────
# Páginas
# ──────────────────────────────────────────────

if page == "📊 Visão Geral":
    st.markdown("## 📊 Visão Geral do Dataset")

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            render_metric_card("Total Avaliações", f"{len(ratings):,}"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_metric_card("Usuários", f"{ratings['userId'].nunique():,}"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_metric_card("Filmes", f"{ratings['movieId'].nunique():,}"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            render_metric_card("Nota Média", f"{ratings['rating'].mean():.2f}"),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Após filtro
    st.markdown("### 🔧 Após Filtragem de Dados Esparsos")
    st.markdown(
        """
        <div class="info-box">
            <p>Filtramos usuários com menos de 20 avaliações e filmes com menos de 10 avaliações
            para melhorar a qualidade das recomendações e reduzir a esparsidade da matriz.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            render_metric_card("Avaliações (filtro)", f"{len(ratings_filtered):,}"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_metric_card(
                "Usuários (filtro)",
                f"{ratings_filtered['userId'].nunique():,}",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_metric_card(
                "Filmes (filtro)",
                f"{ratings_filtered['movieId'].nunique():,}",
            ),
            unsafe_allow_html=True,
        )
    with col4:
        sparsity = (matrix.isna().sum().sum() / matrix.size) * 100
        st.markdown(
            render_metric_card("Esparsidade", f"{sparsity:.1f}%"),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Divisão treino/teste
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📚 Conjunto de Treino")
        st.markdown(
            render_metric_card("Avaliações Treino", f"{len(train):,}"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("### 🧪 Conjunto de Teste")
        st.markdown(
            render_metric_card("Avaliações Teste", f"{len(test):,}"),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Distribuição das notas
    st.markdown("### 📊 Distribuição das Avaliações")
    rating_dist = ratings["rating"].value_counts().sort_index()
    st.bar_chart(rating_dist, use_container_width=True)

    # Top gêneros
    st.markdown("### 🎭 Top 15 Gêneros mais Frequentes")
    all_genres = movies["genres"].str.split("|").explode()
    genre_counts = all_genres.value_counts().head(15)
    st.bar_chart(genre_counts, use_container_width=True)


elif page == "🔍 Consultar Usuário":
    st.markdown("## 🔍 Consultar Histórico de Usuário")

    selected_user = st.selectbox(
        "Selecione um usuário:",
        user_ids,
        format_func=lambda x: f"Usuário {x}",
    )

    if selected_user:
        history = model.get_user_history(selected_user, movies_df=movies)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                render_metric_card("Filmes Avaliados", str(len(history))),
                unsafe_allow_html=True,
            )
        with col2:
            avg = history["rating"].mean() if len(history) > 0 else 0
            st.markdown(
                render_metric_card("Nota Média", f"{avg:.2f}"),
                unsafe_allow_html=True,
            )
        with col3:
            neighbors = model.get_neighbors_info(selected_user)
            st.markdown(
                render_metric_card("Vizinhos Encontrados", str(len(neighbors))),
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Vizinhos
        st.markdown("### 👥 Vizinhos Mais Próximos (KNN)")
        if neighbors:
            cols = st.columns(min(5, len(neighbors)))
            for i, nb in enumerate(neighbors[:5]):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="neighbor-badge" style="display: block; text-align: center;">
                            👤 User {nb['neighbor_id']}<br>
                            📏 Dist: {nb['distance']}<br>
                            🎯 Sim: {nb['similarity']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("Nenhum vizinho encontrado para este usuário.")

        st.markdown("---")

        # Histórico de avaliações
        st.markdown("### 📋 Histórico de Avaliações")

        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox(
                "Ordenar por:",
                ["Maior nota", "Menor nota", "Título (A-Z)"],
            )
        with col2:
            filter_genre = st.multiselect(
                "Filtrar por gênero:",
                sorted(
                    set(
                        g
                        for genres in history["genres"].dropna()
                        for g in genres.split("|")
                    )
                )
                if len(history) > 0
                else [],
            )

        display_history = history.copy()
        if filter_genre:
            display_history = display_history[
                display_history["genres"].apply(
                    lambda g: any(fg in g for fg in filter_genre) if pd.notna(g) else False
                )
            ]

        if sort_by == "Maior nota":
            display_history = display_history.sort_values("rating", ascending=False)
        elif sort_by == "Menor nota":
            display_history = display_history.sort_values("rating", ascending=True)
        else:
            display_history = display_history.sort_values("title", ascending=True)

        for _, row in display_history.iterrows():
            st.markdown(
                render_movie_card(
                    row["title"],
                    row["rating"],
                    row.get("genres", "N/A"),
                ),
                unsafe_allow_html=True,
            )


elif page == "🎬 Recomendações":
    st.markdown("## 🎬 Recomendações Personalizadas")

    selected_user = st.selectbox(
        "Selecione um usuário:",
        user_ids,
        format_func=lambda x: f"Usuário {x}",
        key="rec_user",
    )

    n_recs = st.slider("Número de recomendações:", 5, 30, 10)

    if st.button("🎯 Gerar Recomendações", type="primary", use_container_width=True):
        with st.spinner("Calculando recomendações..."):
            recs = model.recommend(selected_user, n=n_recs, movies_df=movies)

            # Mostrar resumo do usuário
            history = model.get_user_history(selected_user, movies_df=movies)
            st.markdown(
                f"""
                <div class="info-box">
                    <p>📋 <strong>Usuário {selected_user}</strong> avaliou
                    <strong>{len(history)} filmes</strong> com nota média de
                    <strong>{history['rating'].mean():.2f}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f"### 🎬 Top {n_recs} Recomendações para o Usuário {selected_user}")

            for i, (_, row) in enumerate(recs.iterrows(), 1):
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div class="movie-title">#{i} · {row['title']}</div>
                                <div class="movie-genres">🎭 {row.get('genres', 'N/A')}</div>
                            </div>
                            <div style="text-align: right;">
                                <div class="movie-rating">{render_stars(row['predicted_rating'])} {row['predicted_rating']:.2f}</div>
                                <div style="color: #8892b0; font-size: 0.75rem;">Predição</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


elif page == "❄️ Cold Start":
    st.markdown("## ❄️ Cenário Cold Start")
    st.markdown(
        """
        <div class="cold-start-banner">
            <h3>🧊 Usuário Sem Histórico</h3>
            <p>Quando um novo usuário chega ao sistema sem avaliações prévias,
            utilizamos um <strong>fallback por popularidade</strong> com score bayesiano
            para gerar recomendações iniciais.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    new_user_id = st.number_input(
        "ID do novo usuário (qualquer número não existente):",
        min_value=1000,
        max_value=99999,
        value=99999,
    )

    n_cold = st.slider("Número de recomendações:", 5, 20, 10, key="cold_n")

    is_cold = new_user_id not in user_ids

    if is_cold:
        st.success(f"✅ Usuário {new_user_id} não existe no dataset — cenário cold-start ativo!")
    else:
        st.warning(
            f"⚠️ Usuário {new_user_id} já existe no dataset. "
            "As recomendações serão personalizadas, não cold-start."
        )

    if st.button("🎯 Gerar Recomendações Cold Start", type="primary", use_container_width=True):
        with st.spinner("Gerando recomendações por popularidade..."):
            recs = model.recommend(int(new_user_id), n=n_cold, movies_df=movies)

            strategy = "Popularidade (Bayesiana)" if is_cold else "Colaborativa (KNN)"
            st.markdown(
                f"""
                <div class="info-box">
                    <p>🔄 Estratégia utilizada: <strong>{strategy}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f"### 🎬 Recomendações para o Usuário {new_user_id}")
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                st.markdown(
                    render_movie_card(
                        f"#{i} · {row['title']}",
                        row["predicted_rating"],
                        row.get("genres", "N/A"),
                        is_prediction=True,
                    ),
                    unsafe_allow_html=True,
                )


elif page == "⭐ Avaliar Filmes":
    st.markdown("## ⭐ Avaliar Filmes")

    st.markdown(
        """
        <div class="info-box">
            <p>Simule a avaliação de filmes por um usuário. As avaliações são adicionadas
            temporariamente ao modelo para demonstrar como o sistema se adapta.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Selecionar modo
    mode = st.radio(
        "Modo de avaliação:",
        ["Usuário existente", "Novo usuário (simular cold-start)"],
    )

    if mode == "Usuário existente":
        eval_user = st.selectbox(
            "Selecione o usuário:",
            user_ids,
            format_func=lambda x: f"Usuário {x}",
            key="eval_user",
        )
    else:
        eval_user = 99999
        st.info(f"🆕 Avaliando como novo usuário (ID: {eval_user})")

    # Buscar filme
    search_term = st.text_input("🔍 Buscar filme por título:")

    if search_term:
        matched = movies[
            movies["title"].str.contains(search_term, case=False, na=False)
        ].head(20)

        if len(matched) == 0:
            st.warning("Nenhum filme encontrado.")
        else:
            selected_movie = st.selectbox(
                "Selecione o filme:",
                matched["movieId"].tolist(),
                format_func=lambda x: movies[movies["movieId"] == x]["title"].values[0],
            )

            new_rating = st.slider(
                "Sua avaliação:",
                0.5,
                5.0,
                3.0,
                step=0.5,
                key="new_rating",
            )

            st.markdown(
                f"**{render_stars(new_rating)}** ({new_rating:.1f})"
            )

            if st.button("💾 Salvar Avaliação", type="primary"):
                # Salvar no session state para demonstração
                if "user_ratings" not in st.session_state:
                    st.session_state.user_ratings = []

                movie_title = movies[movies["movieId"] == selected_movie]["title"].values[0]
                st.session_state.user_ratings.append(
                    {
                        "userId": eval_user,
                        "movieId": selected_movie,
                        "rating": new_rating,
                        "title": movie_title,
                    }
                )
                st.success(f"✅ Avaliação salva: {movie_title} → {new_rating:.1f}")

    # Mostrar avaliações da sessão
    if "user_ratings" in st.session_state and st.session_state.user_ratings:
        st.markdown("---")
        st.markdown("### 📋 Avaliações desta Sessão")
        for r in st.session_state.user_ratings:
            st.markdown(
                render_movie_card(
                    r["title"],
                    r["rating"],
                    "—",
                ),
                unsafe_allow_html=True,
            )

        if st.button("🗑️ Limpar avaliações da sessão"):
            st.session_state.user_ratings = []
            st.rerun()


elif page == "📈 Resultados da Avaliação":
    st.markdown("## 📈 Resultados da Avaliação do Modelo")

    st.markdown(
        """
        <div class="info-box">
            <p>Avaliação realizada com <strong>50 usuários amostrados</strong> do conjunto de teste.
            Divisão <strong>80% treino / 20% teste</strong> estratificada por usuário.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k_eval = st.selectbox("K para Precision@K e Recall@K:", [5, 10, 15, 20], index=1)

    if st.button("🔄 Executar Avaliação", type="primary", use_container_width=True):
        with st.spinner("Avaliando modelo... (pode levar alguns minutos)"):
            results = run_evaluation("model_v1", test, movies, k=k_eval)

            overall = results["overall"]

            # Métricas principais
            st.markdown("### 📊 Métricas Globais")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(
                    render_metric_card("MAE", f"{overall['MAE']:.4f}"),
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    render_metric_card("RMSE", f"{overall['RMSE']:.4f}"),
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    render_metric_card(
                        f"Precision@{k_eval}",
                        f"{overall['Precision@K']:.4f}",
                    ),
                    unsafe_allow_html=True,
                )
            with col4:
                st.markdown(
                    render_metric_card(
                        f"Recall@{k_eval}",
                        f"{overall['Recall@K']:.4f}",
                    ),
                    unsafe_allow_html=True,
                )
            with col5:
                st.markdown(
                    render_metric_card("Coverage", f"{overall['Coverage']:.4f}"),
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Interpretação
            st.markdown("### 📝 Interpretação das Métricas")
            st.markdown(
                f"""
                | Métrica | Valor | Interpretação |
                |---------|-------|---------------|
                | **MAE** | {overall['MAE']:.4f} | Erro médio absoluto da predição. Quanto menor, melhor. |
                | **RMSE** | {overall['RMSE']:.4f} | Penaliza erros grandes mais que o MAE. Quanto menor, melhor. |
                | **Precision@{k_eval}** | {overall['Precision@K']:.4f} | {overall['Precision@K']*100:.1f}% dos itens recomendados são relevantes. |
                | **Recall@{k_eval}** | {overall['Recall@K']:.4f} | {overall['Recall@K']*100:.1f}% dos itens relevantes foram recomendados. |
                | **Coverage** | {overall['Coverage']:.4f} | {overall['Coverage']*100:.1f}% do catálogo aparece nas recomendações. |
                """
            )

            st.markdown("---")

            # Resultados por usuário
            st.markdown("### 👥 Resultados por Usuário (amostra)")
            per_user = results["per_user"]
            st.dataframe(
                per_user.style.format(
                    {
                        "MAE": "{:.4f}",
                        "RMSE": "{:.4f}",
                        "Precision@K": "{:.4f}",
                        "Recall@K": "{:.4f}",
                    }
                ),
                use_container_width=True,
                height=400,
            )

            # Gráficos
            st.markdown("### 📊 Distribuição de MAE por Usuário")
            st.bar_chart(
                per_user.set_index("userId")["MAE"],
                use_container_width=True,
            )

            st.markdown(f"### 📊 Distribuição de Precision@{k_eval} por Usuário")
            st.bar_chart(
                per_user.set_index("userId")["Precision@K"],
                use_container_width=True,
            )
