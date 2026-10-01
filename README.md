# Sistema de Recomendação de Filmes

Sistema de recomendação baseado em **Filtragem Colaborativa** utilizando **K-Nearest Neighbors (KNN)** com **distância Manhattan**, desenvolvido como trabalho acadêmico.

---

## Objetivo

Desenvolver um sistema de recomendação de filmes capaz de gerar recomendações personalizadas para usuários, utilizando técnicas de filtragem colaborativa baseada em usuário (User-based CF). O sistema deve:

- Utilizar uma base real de interações (avaliações) entre usuários e filmes.
- Implementar filtragem colaborativa com KNN e distância Manhattan.
- Gerar recomendações excluindo itens já avaliados pelo usuário.
- Tratar o cenário de **cold-start** (usuários sem histórico).
- Disponibilizar uma interface interativa para consulta e demonstração.
- Avaliar os resultados com métricas quantitativas.

---

## Fundamentação Teórica

### Filtragem Colaborativa Baseada em Usuário

A filtragem colaborativa (CF) assume que usuários com gostos similares no passado terão gostos similares no futuro. A abordagem User-based CF funciona em três etapas:

1. **Calcular similaridade** entre o usuário alvo e todos os demais.
2. **Selecionar os K vizinhos** mais similares.
3. **Predizer avaliações** para itens não vistos, usando a média ponderada das avaliações dos vizinhos.

### Distância Manhattan

A distância Manhattan (L1) entre dois vetores é a soma dos valores absolutos das diferenças:

```
d(u, v) = Σ |u_i - v_i|
```

Escolhida por sua robustez a outliers comparada à distância Euclidiana, sendo mais adequada para dados de avaliações onde valores extremos são comuns.

### Tratamento de Cold-Start

Para usuários sem histórico, o sistema utiliza um fallback por popularidade bayesiana:

```
score = (count / (count + m)) × mean_rating + (m / (count + m)) × global_mean
```

Onde `m` é um parâmetro de regularização que penaliza filmes com poucas avaliações.

---

## Dados

### Dataset: MovieLens Small (ml-latest-small)

| Característica | Valor |
|---------------|-------|
| **Total de avaliações** | 100.836 |
| **Usuários** | 610 |
| **Filmes** | 9.724 |
| **Escala de notas** | 0.5 a 5.0 (incrementos de 0.5) |
| **Período** | 1996 – 2018 |

### Arquivos

- `ratings.csv`: Avaliações (userId, movieId, rating, timestamp)
- `movies.csv`: Catálogo de filmes (movieId, title, genres)
- `tags.csv`: Tags atribuídas pelos usuários
- `links.csv`: IDs externos (IMDb, TMDb)

### Pré-processamento

- **Filtragem de esparsidade**: removidos usuários com < 20 avaliações e filmes com < 10 avaliações.
- **Divisão treino/teste**: 80/20 estratificado por usuário.
- **Normalização**: subtração da média do usuário para reduzir viés individual.

---

## Método

### Algoritmo: User-Based CF com KNN (Manhattan)

```
Entrada: matriz R (usuários × itens), usuário alvo u, K vizinhos
Saída: lista de N recomendações ordenadas

1. Para cada usuário v ≠ u:
   a. Encontrar itens avaliados por ambos (u, v)
   b. Se |itens_comuns| ≥ min_common:
      - Calcular d_manhattan(u, v) normalizada por |itens_comuns|
2. Ordenar por distância e selecionar os K mais próximos
3. Para cada item i não avaliado por u:
   a. pred(u, i) = μ_u + Σ w_v · (r_vi - μ_v) / Σ w_v
      onde w_v = 1 / (d(u, v) + ε)
4. Retornar os N itens com maiores predições
```

### Parâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|--------------|
| K (vizinhos) | 20 | Balanceia diversidade e relevância |
| min_common | 3 | Mínimo de itens em comum para evitar comparações espúrias |
| Distância | Manhattan (L1) | Robusta a outliers nas avaliações |
| Normalização | Subtração da média | Remove viés por usuário |

---

## Estrutura do Projeto

```
SistemaRecomendacao/
├── app/
│   └── app.py              # Interface Streamlit
├── data/
│   └── raw/
│       ├── ratings.csv
│       ├── movies.csv
│       ├── tags.csv
│       └── links.csv
├── docs/                    # Documentação adicional
├── recommender/
│   ├── __init__.py
│   ├── data_prep.py         # Carregamento e preparação dos dados
│   ├── user_cf.py           # Algoritmo de recomendação (KNN Manhattan)
│   ├── evaluation.py        # Métricas de avaliação
│   └── cf_reference.py      # Script de referência (execução offline)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instruções de Execução

### Pré-requisitos

- Python 3.9+
- pip

### Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/SistemaRecomendacao.git
cd SistemaRecomendacao

# 2. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt
```

### Executar a Interface (Streamlit)

```bash
streamlit run app/app.py
```

A interface estará disponível em `http://localhost:8501`.

### Executar o Script de Referência (terminal)

```bash
python recommender/cf_reference.py
```

Este script treina o modelo, avalia e mostra exemplos para 5 usuários + cold-start.

---

## Resultados

### Métricas de Avaliação

O sistema foi avaliado com 50 usuários amostrados do conjunto de teste (divisão 80/20):

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **MAE** | 0.6852 | Erro médio absoluto das predições na escala de 0.5–5.0 |
| **RMSE** | 0.9012 | Raiz do erro quadrático médio; penaliza desvios maiores |
| **Precision@10** | 0.0160 | ~1.6% dos itens recomendados no top-10 são relevantes (nota ≥ 4.0 no teste) |
| **Recall@10** | 0.0051 | ~0.51% dos itens relevantes do usuário no teste foram capturados no top-10 |
| **Coverage** | 0.1406 | O sistema recomenda 14.06% dos itens do catálogo para a amostra avaliada |

> **Nota:** Valores obtidos na execução de referência via `recommender/cf_reference.py` (seed=42, 50 usuários amostrados, K=10). Execute o script ou a interface Streamlit para reproduzir os resultados.

### Exemplos de Recomendação

O script `cf_reference.py` gera exemplos detalhados para **5 usuários**, mostrando:
- Histórico de avaliações (top 5 filmes)
- Vizinhos mais próximos (top 3)
- Top 10 recomendações com predições
- Caso cold-start (usuário sem histórico)

---

## Limitações

1. **Escalabilidade**: O cálculo par-a-par de distâncias é O(n² × m), limitando a aplicação a datasets de médio porte.
2. **Cold-start de itens**: Filmes novos sem avaliações não são recomendados pela CF, dependendo apenas do fallback por popularidade.
3. **Esparsidade**: Mesmo após filtragem, a matriz permanece ~90% esparsa, limitando a qualidade dos vizinhos.
4. **Evolução temporal**: O modelo não considera a evolução dos gostos do usuário ao longo do tempo.
5. **Explicabilidade limitada**: A CF não fornece justificativas semânticas (ex: "porque você gostou de ação").
6. **Sem feedback implícito**: Utiliza apenas avaliações explícitas, ignorando cliques e visualizações.

---

## Conclusão

O sistema demonstra a viabilidade da **filtragem colaborativa baseada em usuário com KNN e distância Manhattan** para gerar recomendações de filmes. Os resultados indicam que:

- O MAE e RMSE estão dentro de faixas aceitáveis para o dataset MovieLens, compatíveis com a literatura.
- A estratégia de cold-start por popularidade bayesiana oferece recomendações razoáveis para novos usuários.
- A interface Streamlit permite a demonstração interativa de todos os cenários exigidos.

Como trabalhos futuros, sugerimos a exploração de abordagens híbridas (combinando CF com filtragem baseada em conteúdo) e modelos de fatoração de matrizes para melhorar tanto a precisão quanto a cobertura do sistema.

---

## Tecnologias Utilizadas

- **Python 3.9+**
- **Pandas** – Manipulação de dados
- **NumPy** – Computação numérica
- **Streamlit** – Interface web interativa

---

## Referências

- Resnick, P. et al. (1994). *GroupLens: An Open Architecture for Collaborative Filtering of Netnews*. CSCW.
- Herlocker, J. et al. (2004). *Evaluating Collaborative Filtering Recommender Systems*. ACM TOIS.
- Harper, F. M. & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM TIIS.
