# Módulo do sistema de recomendação
from .data_prep import load_data, build_user_item_matrix
from .user_cf import UserCFRecommender
from .evaluation import evaluate_recommender
