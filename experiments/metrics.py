import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import pingouin as pg
import joblib
test_df = pd.read_csv('data/test_manual.csv')
icc_data = pd.DataFrame({
    'target': np.repeat(test_df.index, 2),
    'rater': np.tile(['LLM', 'Human'], len(test_df)),
    'score': np.concatenate([test_df['score'].values, test_df['true_score'].values])
})
icc = pg.intraclass_corr(data=icc_data, targets='target', raters='rater', ratings='score')
print("=== Inter-Annotator Agreement ===")
print("ICC (Intraclass Correlation):\n", icc[['Type', 'ICC', 'CI95']])
bins = [-0.1, 0.2, 0.4, 0.6, 0.8, 1.1]
test_df['llm_class'] = pd.cut(test_df['score'], bins=bins, labels=False)
test_df['human_class'] = pd.cut(test_df['true_score'], bins=bins, labels=False)
kappa = cohen_kappa_score(test_df['llm_class'], test_df['human_class'], weights='quadratic')

print(f"Weighted Cohen's Kappa: {kappa:.4f}")
model_dir = "./xgb_trained_model"
xgb_model = joblib.load(f"{model_dir}/xgb_model.pkl")
tfidf = joblib.load(f"{model_dir}/tfidf.pkl")
feature_names = [f"E5_{i}" for i in range(1024)] + \
                list(tfidf.get_feature_names_out()) + \
                ['Cluster_Target']
X_full_test = pd.read_csv('data/test_manual.csv')

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_full_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_full_test, feature_names=feature_names, show=False)
plt.title("SHAP Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.show()
