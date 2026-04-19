import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('scripts/reg_models/finance_lab_newer.csv')
clean_cols = ['Debt 2023', 'EBITDA 2023', 'Revenue 2024', 'Debt 2024', 'EBITDA 2024']
for col in clean_cols:
    df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df.columns = df.columns.str.replace(' ', '_')
df['ai_score_std'] = (df['ai_score'] - df['ai_score'].mean()) / df['ai_score'].std()
df['Rev_Growth_Pct'] = ((df['Revenue_2024'] - df['Revenue_2023']) / df['Revenue_2023']) * 100
df['Rev_per_Emp_24'] = df['Revenue_2024'] / df['Employees_2024']
df['Rev_per_Emp_23'] = df['Revenue_2023'] / df['Employees_2023']
df['Log_Rev_Emp_24'] = np.log(df['Rev_per_Emp_24'].replace(0, np.nan))
df['Log_Rev_Emp_23'] = np.log(df['Rev_per_Emp_23'].replace(0, np.nan))
df['Log_Revenue_23'] = np.log(df['Revenue_2023'].replace(0, np.nan))
mod_base = smf.ols("Rev_Growth_Pct ~ ai_score_std * C(Industry, Treatment(reference='other')) + Log_Revenue_23 + Age_2023", data=df).fit()

print("=== 1. Проверка на мультиколлинеарность ===")
X = mod_base.model.exog
features = mod_base.model.exog_names

vif_data = pd.DataFrame()
vif_data["Feature"] = features
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print(vif_data)

print("\n=== 2. Поиск влиятельных наблюдений (Cook's Distance) ===")
influence = mod_base.get_influence()
cooks_d = influence.cooks_distance[0]
df['Cooks_Distance'] = cooks_d
outliers = df.sort_values(by='Cooks_Distance', ascending=False).head(5)
print(outliers[['Company', 'Industry', 'ai_score_std', 'Rev_Growth_Pct', 'Cooks_Distance']])
plt.figure(figsize=(10, 5))
plt.stem(df.index, cooks_d, basefmt=" ")
plt.axhline(4/len(df), color='red', linestyle='--', label='Порог (4/n)')
plt.title("Расстояние Кука (влияние отдельных компаний на модель)")
plt.legend()
plt.show()