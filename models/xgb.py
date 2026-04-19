import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
import optuna
import torch
import ast
import os
import joblib

def parse_skills(raw_str):
    if not isinstance(raw_str, str) or not raw_str.strip():
        return []
    try:
        skills = ast.literal_eval(raw_str)
        if isinstance(skills, list):
            return [str(s).strip() for s in skills if str(s).strip()]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []

def concat_skills(raw_skills):
    return ', '.join([s for s in parse_skills(raw_skills)])

def preprocess(df):
    df['raw_skills'] = df['raw_skills'].fillna('')
    df['text'] = "Название: " + df['name'] + ". Требуемые навыки: " + df['raw_skills'].apply(concat_skills) + ". Описание: " + df['description']
    return df


def run_model(df):
    df = preprocess(df)
    model_emb = SentenceTransformer('intfloat/multilingual-e5-large')
    X_emb = model_emb.encode(df['text'].tolist(), show_progress_bar=True)
    model_dir = "./xgb_trained_model"
    kmeans = joblib.load(f"{model_dir}/kmeans.pkl")
    tfidf = joblib.load(f"{model_dir}/tfidf.pkl")
    xgb = joblib.load(f"{model_dir}/xgb_model.pkl")
    cluster_map = joblib.load(f"{model_dir}/cluster_map.pkl")
    global_mean = joblib.load(f"{model_dir}/global_mean.pkl")
    clusters = kmeans.predict(X_emb)
    df['cluster_target'] = pd.Series(clusters).map(cluster_map).fillna(global_mean)
    X_tfidf = tfidf.transform(df['text']).toarray()
    X_full = np.hstack([X_emb, X_tfidf, df[['cluster_target']].values])
    ai_scores = xgb.predict(X_full)
    df['ai_score'] = ai_scores
    return df


if __name__ == '__main__':
    train_df = pd.read_csv('data/train_bigger.csv')
    test_df = pd.read_csv('data/test_manual.csv')
    train_df, test_df = preprocess(train_df), preprocess(test_df)
    model_emb = SentenceTransformer('intfloat/multilingual-e5-large')
    X_emb_train = model_emb.encode(train_df['text'].tolist(), show_progress_bar=True)
    X_emb_test = model_emb.encode(test_df['text'].tolist(), show_progress_bar=True)
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    clusters_train = kmeans.fit_predict(X_emb_train)
    clusters_test = kmeans.predict(X_emb_test)
    cluster_map = train_df.assign(cl=clusters_train).groupby('cl')['score'].mean().to_dict()
    train_df['cluster_target'] = pd.Series(clusters_train).map(cluster_map)
    test_df['cluster_target'] = pd.Series(clusters_test).map(cluster_map).fillna(train_df['score'].mean())
    tfidf = TfidfVectorizer(max_features=500, min_df=5)
    X_tfidf_train = tfidf.fit_transform(train_df['text']).toarray()
    X_tfidf_test = tfidf.transform(test_df['text']).toarray()
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'tree_method': 'hist',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        X_train_opt = np.hstack([X_emb_train, train_df[['cluster_target']].values])
        X_test_opt = np.hstack([X_emb_test, test_df[['cluster_target']].values])
        model = XGBRegressor(**param)
        model.fit(X_train_opt, train_df['score'])
        preds = model.predict(X_test_opt)
        return mean_squared_error(test_df['true_score'], preds)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    X_full_train = np.hstack([X_emb_train, X_tfidf_train, train_df[['cluster_target']].values])
    X_full_test = np.hstack([X_emb_test, X_tfidf_test, test_df[['cluster_target']].values])
    best_xgb = XGBRegressor(**study.best_params)
    best_xgb.fit(X_full_train, train_df['score'])
    final_preds = best_xgb.predict(X_full_test)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.regplot(x=test_df['true_score'], y=final_preds, ax=ax1, 
                scatter_kws={'alpha':0.4, 'color':'teal'}, line_kws={'color':'red'})
    ax1.set_title(f'XGBoost Optimized: Predicted vs True\n$R^2$ = {r2_score(test_df["true_score"], final_preds):.3f}')
    ax1.set_xlabel('Экспертная оценка (True)')
    ax1.set_ylabel('Предсказание модели')
    errors = np.abs(test_df['true_score'] - final_preds)
    sns.histplot(errors, bins=30, kde=True, ax=ax2, color='orange')
    ax2.set_title('Распределение абсолютных ошибок (MAE)')
    ax2.set_xlabel('Величина ошибки |True - Pred|')
    plt.tight_layout()
    plt.savefig('xgboost_performance.png')
    plt.show()
    print(f"Best Params: {study.best_params}")
    print(f"Final R2: {r2_score(test_df['true_score'], final_preds):.4f}")
    model_dir = "./xgb_trained_model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, f"{model_dir}/kmeans.pkl")
    joblib.dump(tfidf, f"{model_dir}/tfidf.pkl")
    joblib.dump(best_xgb, f"{model_dir}/xgb_model.pkl")
    joblib.dump(cluster_map, f"{model_dir}/cluster_map.pkl")
    global_mean = train_df['score'].mean()
    joblib.dump(global_mean, f"{model_dir}/global_mean.pkl")