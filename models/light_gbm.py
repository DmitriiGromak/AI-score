import pandas as pd
import numpy as np
import re
import ast
import nltk
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from lightgbm import LGBMRegressor
from sentence_transformers import SentenceTransformer

nltk.data.find('tokenizers/punkt')
ANCHOR_PHRASES = [
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

def concat_skills(raw_skills):
    return ', '.join([s for s in parse_skills(raw_skills)])

def prepare_text(df):
    return "Название: " + df['name'].fillna('') + \
           ". Требуемые навыки: " + df['raw_skills'].apply(concat_skills) + \
           ". Описание: " + df['description'].fillna('')

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

def extract_sentence_level_clusters(df, text_col, model_emb, anchor_embeddings):
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
        
    return np.array(features_list)

def train_and_save_pipeline(train_csv_path='data/train_bigger.csv'):
    train_df = pd.read_csv(train_csv_path)
    train_df['text'] = prepare_text(train_df)
    print("Обучение TF-IDF...")
    tfidf = TfidfVectorizer(max_features=500, min_df=5)
    X_tfidf_train = tfidf.fit_transform(train_df['text']).toarray()
    print("Генерация E5 эмбеддингов...")
    model_emb = SentenceTransformer('intfloat/multilingual-e5-large')
    X_emb_train = model_emb.encode(train_df['text'].tolist(), show_progress_bar=True)
    anchor_embeddings = model_emb.encode(ANCHOR_PHRASES, show_progress_bar=False)
    print("Извлечение Sentence-level признаков...")
    X_sent_clust_train = extract_sentence_level_clusters(train_df, 'text', model_emb, anchor_embeddings)
    X_dom_train = extract_domain_features(train_df)
    X_train_full = np.hstack([X_sent_clust_train, X_emb_train, X_tfidf_train, X_dom_train])
    y_train = train_df['score'].values
    print("Обучение LightGBM...")
    lgbm = LGBMRegressor(
        n_estimators=300, 
        num_leaves=31, 
        learning_rate=0.05, 
        random_state=42, 
        n_jobs=-1, 
        verbosity=-1
    )
    lgbm.fit(X_train_full, y_train)
    print("Сохранение модели и артефактов в папку 'saved_models'...")
    os.makedirs('saved_models', exist_ok=True)
    lgbm.booster_.save_model('saved_models/lgbm_model.txt')
    joblib.dump(tfidf, 'saved_models/tfidf_vectorizer.joblib')
    joblib.dump(anchor_embeddings, 'saved_models/anchor_embeddings.joblib')

def run_model(df, models_dir='saved_models'):
    print(f"Запуск модели для {len(df)} вакансий...")
    import lightgbm as lgb
    tfidf = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    anchor_embeddings = joblib.load(os.path.join(models_dir, 'anchor_embeddings.joblib'))
    bst = lgb.Booster(model_file=os.path.join(models_dir, 'lgbm_model.txt'))
    model_emb = SentenceTransformer('intfloat/multilingual-e5-large')
    df = df.copy()
    df['text'] = prepare_text(df)
    X_tfidf = tfidf.transform(df['text']).toarray()
    X_emb = model_emb.encode(df['text'].tolist(), show_progress_bar=True)
    X_sent_clust = extract_sentence_level_clusters(df, 'text', model_emb, anchor_embeddings)
    X_dom = extract_domain_features(df)
    X_full = np.hstack([X_sent_clust, X_emb, X_tfidf, X_dom])
    preds = bst.predict(X_full)
    preds = np.clip(preds, 0.0, 1.0)
    
    df['ai_score'] = preds
    print("Готово. Колонка 'ai_score' успешно добавлена.")
    
    return df
if __name__ == '__main__':
    train_and_save_pipeline('data/train_bigger.csv')
