import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

plt.style.use('seaborn-v0_8-whitegrid')
df = pd.read_csv('scripts/reg_models/finance_lab.csv')
clean_cols = ['Debt 2023', 'EBITDA 2023', 'Revenue 2024', 'Debt 2024', 'EBITDA 2024']
for col in clean_cols:
    df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df.columns = df.columns.str.replace(' ', '_')
df['ai_score_std'] = (df['ai_score'] - df['ai_score'].mean()) / df['ai_score'].std()
df['Rev_Growth_Pct'] = ((df['Revenue_2024'] - df['Revenue_2023']) / df['Revenue_2023']) * 100
df['Log_Revenue_23'] = np.log(df['Revenue_2023'].replace(0, np.nan))

mod = smf.ols("Rev_Growth_Pct ~ ai_score_std * C(Industry, Treatment(reference='other')) + Log_Revenue_23 + Age_2023", data=df).fit(cov_type='HC3')
params = mod.params.drop('Intercept')
conf_int = mod.conf_int().drop('Intercept')
errors = params - conf_int[0]

plt.figure(figsize=(10, 6))
plt.errorbar(params, params.index, xerr=errors, fmt='o', color='teal', capsize=5, capthick=2, markersize=8)
plt.axvline(0, color='red', linestyle='--', alpha=0.7)
plt.title("Влияние факторов на рост выручки (Робастные SE с 95% ДИ)", fontsize=14)
plt.xlabel("Коэффициент регрессии (Marginal Effect)")
plt.tight_layout()
plt.savefig("coef_plot.png", dpi=300)
plt.show()
ai_range = np.linspace(df['ai_score_std'].min(), df['ai_score_std'].max(), 50)
industries = df['Industry'].unique()
pred_data = pd.DataFrame(
    [(ai, ind, df['Log_Revenue_23'].mean(), df['Age_2023'].mean()) 
     for ai in ai_range for ind in industries],
    columns=['ai_score_std', 'Industry', 'Log_Revenue_23', 'Age_2023']
)

pred_data['Predicted_Growth'] = mod.predict(pred_data)
plt.figure(figsize=(10, 6))
sns.lineplot(data=pred_data, x='ai_score_std', y='Predicted_Growth', hue='Industry', linewidth=2.5)
plt.title("Предсказанный рост выручки в зависимости от AI Score (по отраслям)", fontsize=14)
plt.xlabel("AI Score (стандартизованный)")
plt.ylabel("Предсказанный рост выручки (%)")
plt.legend(title="Отрасль")
plt.tight_layout()
plt.savefig("interaction_plot.png", dpi=300)
plt.show()
cols_to_corr = ['Rev_Growth_Pct', 'ai_score', 'Log_Revenue_23', 'Age_2023']
corr_matrix = df[cols_to_corr].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", square=True)
plt.title("Корреляционная матрица (Pearson)")
plt.tight_layout()
plt.savefig("corr_heatmap.png", dpi=300)
plt.show()
