import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
import ast
import nltk
from sklearn.metrics.pairwise import cosine_distances
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

nltk.data.find('tokenizers/punkt')

anchor_phrases = [
    "машинное обучение", "искусственный интеллект", "компьютерное зрение", 
    "обработка естественного языка", "machine learning", "artificial intelligence", 
    "computer vision", "nlp", "нейронные сети", "deep learning", "llm", "pytorch"
]

def parse_skills(raw_str):
    if not isinstance(raw_str, str) or not raw_str.strip(): return []
    try:
        skills = ast.literal_eval(raw_str)
        if isinstance(skills, list): return [str(s).strip() for s in skills if str(s).strip()]
        return []
    except: return []

def extract_domain_features(df):
    features = pd.DataFrame(index=df.index)
    features['desc_len'] = df['description'].fillna('').apply(len)
    features['desc_len_log'] = np.log1p(features['desc_len'])
    features['skills_count'] = df['raw_skills'].apply(lambda x: len(parse_skills(x)))
    names_lower = df['name'].str.lower().fillna('')
    features['is_senior'] = names_lower.str.contains('senior|сеньор|ведущий|lead|principal|head').astype(int)
    features['is_junior'] = names_lower.str.contains('junior|джуниор|младший|стажер|intern').astype(int)
    def extract_experience(text):
        match = re.search(r'(опыт|от)\s+(\d+)\s*(лет|года)', str(text).lower())
        return int(match.group(2)) if match else 0
    features['experience_years'] = df['description'].apply(extract_experience)
    return features.values

train_df = pd.read_csv('data/train_bigger.csv')
test_df = pd.read_csv('data/test_manual.csv')
def concat_skills(raw_skills):
    return ', '.join([s for s in parse_skills(raw_skills)])
def prepare_text(df):
    return "Название: " + df['name'] + ". Требуемые навыки: " + df['raw_skills'].apply(concat_skills) + ". Описание: " + df['description']
train_df['text'] = prepare_text(train_df)
test_df['text'] = prepare_text(test_df)
plt.figure(figsize=(10, 6))
plt.scatter(test_df['true_score'], test_df['score'], color='blue', marker='o')
plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Идеальный прогноз (Y=X)')
plt.xlabel("Истинные значения (True AI-score)", fontsize=11)
plt.ylabel("Предсказания Claude Opus (Predicted)", fontsize=11)
plt.title(f"Распределение предсказаний Claude Opus относительно ручной разметки", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.show()
plt.savefig('true_score.png', dpi=300)   

tfidf = TfidfVectorizer(max_features=500, min_df=5)
X_tfidf_train = tfidf.fit_transform(train_df['text']).toarray()
X_tfidf_test = tfidf.transform(test_df['text']).toarray()
X_dom_train = extract_domain_features(train_df)
X_dom_test = extract_domain_features(test_df)
model_emb = SentenceTransformer('intfloat/multilingual-e5-large')
X_emb_train = model_emb.encode(train_df['text'].tolist(), show_progress_bar=True)
X_emb_test = model_emb.encode(test_df['text'].tolist(), show_progress_bar=True)
anchor_embeddings = model_emb.encode(anchor_phrases)
def extract_sentence_level_clusters(df, text_col='text'):
    features_list = []
    for idx, row in df.iterrows():
        text = row[text_col]
        sentences = nltk.tokenize.sent_tokenize(text)
        if not sentences:
            sentences = [text]
        sent_embs = model_emb.encode(sentences, show_progress_bar=False)
        distances = cosine_distances(sent_embs, anchor_embeddings)
        min_distances = distances.min(axis=0)
        features_list.append(min_distances)
    feature_names = [f"min_dist_to_cluster_{i}" for i in range(len(anchor_phrases))]
    return pd.DataFrame(features_list, columns=feature_names)


pca = PCA(n_components=100, random_state=42)
X_pca_train = pca.fit_transform(X_emb_train)
X_pca_test = pca.transform(X_emb_test)

df_sent_clusters_train = extract_sentence_level_clusters(train_df)
df_sent_clusters_test = extract_sentence_level_clusters(test_df)
X_sent_clust_train = df_sent_clusters_train.values
X_sent_clust_test = df_sent_clusters_test.values
datasets = {
    "1. Baseline (TF-IDF + Domain)": (
        np.hstack([X_tfidf_train, X_dom_train]), 
        np.hstack([X_tfidf_test, X_dom_test])
    ),
    "2. Embeddings only (E5)": (X_emb_train, X_emb_test),
    "3. E5 + PCA(100)": (X_pca_train, X_pca_test),
    "4. Full (E5 + TF-IDF + Domain)": (
        np.hstack([X_emb_train, X_tfidf_train, X_dom_train]), 
        np.hstack([X_emb_test, X_tfidf_test, X_dom_test])
    )
}
datasets["5. Sentence-Level Clusters"] = (
    np.hstack([X_sent_clust_train]),
    np.hstack([X_sent_clust_test])
)
datasets["6. Full (Sentence Clusters + E5 Raw + TF-IDF)"] = (
    np.hstack([X_sent_clust_train, X_emb_train, X_tfidf_train, X_dom_train]),
    np.hstack([X_sent_clust_test, X_emb_test, X_tfidf_test, X_dom_test])
)
best_results = {
    "Ridge": {"r2": -float('inf'), "preds": None, "mae": None, "mse": None, "dataset": ""},
    "SVR": {"r2": -float('inf'), "preds": None, "mae": None, "mse": None, "dataset": ""},
    "XGBoost": {"r2": -float('inf'), "preds": None, "mae": None, "mse": None, "dataset": ""},
    "LightGBM": {"r2": -float('inf'), "preds": None, "mae": None, "mse": None, "dataset": ""}
}
models_to_test = [
    (Ridge(), "Ridge"),
    (SVR(kernel='rbf'), "SVR"),
    (XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1), "XGBoost"),
    (LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=-1), "LightGBM")
]

for name, (X_tr, X_te) in datasets.items():
    for model, m_name in models_to_test:
        model.fit(X_tr, train_df['score'])
        preds = model.predict(X_te)
        current_r2 = r2_score(test_df['true_score'], preds)
        current_mae = mean_absolute_error(test_df['true_score'], preds)
        current_mse = mean_squared_error(test_df['true_score'], preds)
        
        print(f"{name} [{m_name}] -> R2: {current_r2:.4f}")
        if current_r2 > best_results[m_name]["r2"]:
            best_results[m_name] = {
                "r2": current_r2,
                "preds": preds,
                "mae": current_mae,
                "mse": current_mse,
                "dataset": name
            }

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (m_name, res) in enumerate(best_results.items()):
    ax = axes[idx]
    preds = res["preds"]
    true_vals = test_df['true_score']
    ax.scatter(true_vals, preds, alpha=0.6, color='b', edgecolor='k')
    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Идеальный прогноз (Y=X)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Истинные значения (True AI-score)", fontsize=11)
    ax.set_ylabel("Предсказания (Predicted)", fontsize=11)
    ax.set_title(f"Лучшая модель: {m_name}\nНабор признаков: {res['dataset']}", fontsize=12)
    textstr = '\n'.join((
        f"$R^2$ = {res['r2']:.3f}",
        f"MAE = {res['mae']:.3f}",
        f"MSE = {res['mse']:.3f}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('scatter_predictions_best_models.png', dpi=300)
