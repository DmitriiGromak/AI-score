import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

df = pd.read_csv('scripts/reg_models/finance_lab.csv')
clean_cols = ['Debt 2023', 'EBITDA 2023', 'Revenue 2024', 'Debt 2024', 'EBITDA 2024']
for col in clean_cols:
    df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df.columns = df.columns.str.replace(' ', '_')
df['ai_score_std'] = (df['ai_score'] - df['ai_score'].mean()) / df['ai_score'].std()
df['Rev_Growth_Pct'] = ((df['Revenue_2024'] - df['Revenue_2023']) / df['Revenue_2023']) * 100
df['Log_Revenue_23'] = np.log(df['Revenue_2023'].replace(0, np.nan))
mod_base = smf.ols("Rev_Growth_Pct ~ ai_score_std * C(Industry, Treatment(reference='other')) + Log_Revenue_23 + Age_2023", data=df).fit()
bp_test = het_breuschpagan(mod_base.resid, mod_base.model.exog)
print(f"Breusch-Pagan test p-value: {bp_test[1]:.4f}")
mod_robust = smf.ols("Rev_Growth_Pct ~ ai_score_std * C(Industry, Treatment(reference='other')) + Log_Revenue_23 + Age_2023", data=df).fit(cov_type='HC3')
print("\n=== Модель 1: Рост выручки (Робастные SE HC3) ===")
print(mod_robust.summary().tables[1])
q_low = df['Rev_Growth_Pct'].quantile(0.01)
q_high = df['Rev_Growth_Pct'].quantile(0.99)
df_clean = df[(df['Rev_Growth_Pct'] > q_low) & (df['Rev_Growth_Pct'] < q_high)]

mod_clean = smf.ols("Rev_Growth_Pct ~ ai_score_std * C(Industry, Treatment(reference='other')) + Log_Revenue_23 + Age_2023", data=df_clean).fit(cov_type='HC3')
print("\n=== Модель 1 (Без выбросов, HC3) ===")
print(mod_clean.summary().tables[1])
