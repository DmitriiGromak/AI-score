import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
df = pd.read_csv('finance_lab_newer.csv')
clean_cols = ['Debt 2023', 'EBITDA 2023', 'Revenue 2024', 'Debt 2024', 'EBITDA 2024']
for col in clean_cols:
    df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df.columns = df.columns.str.replace(' ', '_')
df['ai_score_std'] = (df['ai_score'] - df['ai_score'].mean()) / df['ai_score'].std()
df['Rev_Growth_Pct'] = (df['Revenue_2024'] - df['Revenue_2023'])
df['Rev_per_Emp_24'] = df['Revenue_2024'] / df['Employees_2024']
df['Rev_per_Emp_23'] = df['Revenue_2023'] / df['Employees_2023']
df['Log_Rev_Emp_24'] = np.log(df['Rev_per_Emp_24'].replace(0, np.nan))
df['Log_Rev_Emp_23'] = np.log(df['Rev_per_Emp_23'].replace(0, np.nan))
df['Delta_Prod'] = df['Log_Rev_Emp_24'] - df['Log_Rev_Emp_23']
df['Log_Revenue_23'] = np.log(df['Revenue_2023'].replace(0, np.nan))
mod_growth = smf.ols(
    "Rev_Growth_Pct ~ ai_score * C(Industry, Treatment(reference='other')) + Log_Revenue_23 + Age_2023", 
    data=df
).fit()
mod_prod = smf.ols(
    "Delta_Prod ~ ai_score * C(Industry, Treatment(reference='other')) + Log_Revenue_23", 
    data=df
).fit()

print("=== Модель 1: Рост выручки (%) ===")
print(mod_growth.summary())

print("\n=== Модель 2: Производительность труда (Логарифм) ===")
print(mod_prod.summary())
