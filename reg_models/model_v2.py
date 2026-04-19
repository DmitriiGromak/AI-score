import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


data_dict = {
    'company': [
        'МАГНИТ, Розничная сеть', 'Яндекс', 'Тинькофф', 'СБЕР', 'Ozon', 'Ростелеком', 'FIX PRICE', 
        'Почта России', 'Сеть магазинов цифровой и бытовой техники DNS', 'Газпром нефть', 'МТС', 
        'Российские железные дороги', 'Лента, федеральная розничная сеть', 'билайн', 'Альфа-Банк', 
        'Детский Мир', 'Черкизово, Группа предприятий', 'МегаФон', 'Деловые Линии', 'Совкомбанк', 
        'Спортмастер', 'Lamoda', 'ВкусВилл', 'СИБУР, Группа компаний', 'Банк ВТБ (ПАО)', 
        'М.Видео-Эльдорадо', 'Мираторг, Агропромышленный холдинг', 'Skyeng', 'ГЛОРИЯ ДЖИНС', 
        'Холдинг Селигдар', 'АК АЛРОСА', 'ОКЕЙ – Федеральная розничная сеть', 
        'ПИК-специализированный застройщик', 'Норникель', 'РОСБАНК', 'СДЭК', 
        'Уральская горно-металлургическая компания', 'Группа компаний Система', 'ВсеИнструменты.ру', 
        'КАРИ', 'ЕВРАЗ', 'Газпромбанк', 'КАМАЗ', 'Северсталь', 'Группа компаний МЕДСИ', 
        'Банк Открытие', 'ЛУКОЙЛ', 'Х5 Group', 'РОЛЬФ, группа компаний', 'ТрансМашХолдинг, Группа компаний'
    ],
    'margin_2023': [
        66.1, 55.6, 80.9, 1508.6, 59.4, 42.3, 15.2, 12.5, 25.0, 639.3, 54.6, 
        150.0, 25.0, 20.0, 120.0, 8.0, 25.0, 45.0, 5.0, 80.0, 12.0, 10.7, 15.0, 
        200.0, 450.0, 15.0, 30.0, 1.5, 8.0, 15.0, 80.0, 25.0, 20.0, 800.0, 30.0, 
        15.0, 50.0, 10.0, 5.0, 8.0, 100.0, 100.0, 5.0, 25.0, 40.0, 120.0, 850.0, 
        90.3, 10.0, 50.0
    ],
    'margin_2024': [
        49.9, 11.5, 122.2, 1580.3, 42.7, 24.1, 18.0, 15.0, 28.0, 479.5, 49.0, 
        180.0, 30.0, 25.0, 140.0, 10.0, 28.0, 50.0, 6.0, 95.0, 15.0, 12.0, 18.0, 
        220.0, 500.0, 18.0, 35.0, 2.0, 10.0, 18.0, 90.0, 28.0, 22.0, 850.0, 35.0, 
        18.0, 55.0, 12.0, 6.0, 10.0, 120.0, 110.0, 6.0, 28.0, 45.0, 130.0, 848.5, 
        110.1, 12.0, 55.0
    ],
    'delta_ai_score': [0.19002546103937284, 0.2326332601732102, 0.20478248451319006, 0.23789592257567815, 0.20929325090312376, 0.16477341833067874, 0.15546517303356758, 0.15915561893156596, 0.13422692460673197, 0.2238695505592558, 0.1713822129699919, 0.2095334678888321, 0.18537183105945587, 0.20805935975578096, 0.18296855344222143, 0.1805740550160408, 0.157150474190712, 0.19833669845353474, 0.19379612058401108, 0.19683836276332536, 0.1855878017165444, 0.17899925634264946, 0.1918234055240949, 0.20875455566295764, 0.225777118653059, 0.1671308996155858, 0.1918273463845253, 0.15912585687197622, 0.14325730688869953, 0.17598893628879028, 0.26448429748415947, 0.0, 0.1765990195175012, 0.21672858903184533, 0.2433811521955899, 0.1540067031979561, 0.21480271816253663, 0.1208297461271286, 0.1816336154937744, 0.0, 0.17513559882839522, 0.2097353402238626, 0.2396999180316925, 0.18097589827246136, 0.10924776643514633, 0.18510897126462725, 0.20333465819175428, 0.17204746284655162, 0.17458555847406387, 0.18387515030123971],
    # 'investment_to_revenue_2023': [
    #     0.029, 0.12, 0.05, 0.08, 0.15, 0.25, 0.03, 0.10, 0.04, 0.11, 0.22, 
    #     0.15, 0.03, 0.20, 0.06, 0.04, 0.05, 0.18, 0.08, 0.07, 0.04, 0.05, 0.04, 
    #     0.09, 0.07, 0.05, 0.06, 0.06, 0.04, 0.12, 0.10, 0.028, 0.09, 0.10, 0.05, 
    #     0.08, 0.07, 0.04, 0.05, 0.06, 0.10, 0.07, 0.04, 0.08, 0.06, 0.07, 0.09, 
    #     0.028, 0.05, 0.08
    # ],
    'age_2023': [
        29, 26, 19, 32, 7, 33, 15, 23, 24, 20, 32, 186, 29, 29, 34, 35, 52, 19, 
        30, 33, 28, 13, 18, 25, 35, 35, 32, 15, 35, 25, 75, 32, 20, 90, 32, 30, 
        32, 25, 28, 30, 30, 35, 20, 35, 25, 30, 32, 24, 25, 25
    ],
    'industry': [
        'Retail', 'IT', 'Banking', 'Banking', 'Ecommerce', 'Telecom', 'Retail', 'Logistics', 'Retail', 
        'OilGas', 'Telecom', 'Logistics', 'Retail', 'Telecom', 'Banking', 'Retail', 'Food', 'Telecom', 
        'Logistics', 'Banking', 'Retail', 'Ecommerce', 'Retail', 'OilGas', 'Banking', 'Retail', 'Food', 
        'IT', 'Retail', 'Mining', 'Mining', 'Retail', 'Construction', 'Mining', 'Banking', 'Logistics', 
        'Mining', 'Other', 'Retail', 'Retail', 'Steel', 'Banking', 'Auto', 'Steel', 'Health', 'Banking', 
        'OilGas', 'Retail', 'Auto', 'Construction'
    ]
}

data = pd.DataFrame(data_dict)


data['delta_ln_margin'] = np.log(data['margin_2024']) - np.log(data['margin_2023'])
data['delta_ai_score_std'] = (data['delta_ai_score'] - data['delta_ai_score'].mean()) / data['delta_ai_score'].std()

# 2. Группировка отраслей (5 групп вместо 16)
industry_map = {
    'Retail': 'Retail',
    'Banking': 'Banking',
    'Telecom': 'Tech',
    'OilGas': 'Heavy', 'Mining': 'Heavy', 'Steel': 'Heavy', 'Construction': 'Heavy',
    'IT': 'Tech', 'Ecommerce': 'Tech',
    'Food': '', 'Health': '', 'Logistics': '', 'Auto': '', '': ''
}
# Finance, IT, Other services
data['industry_group'] = data['industry'].map(industry_map).fillna('Other')
data = pd.get_dummies(data, columns=['industry_group'], drop_first=True, dtype=int)

data['ln_age'] = np.log(data['age_2023'])
# debt to equity
# ln revenue
# age
# profit/revenue, EBITDA/revenue
# data['ai_x_capex'] = data['delta_ai_score_std'] * data['investment_to_revenue_2023']

controls = [col for col in data.columns if col.startswith('industry_group_')]
X_vars = ['ai_score_std'] + controls

def calculate_vif(df, features):
    X_vif = df[features].copy()
    X_vif = sm.add_constant(X_vif)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    return vif_data

X = data[X_vars]
X = sm.add_constant(X)
y = data['delta_ln_margin']

model = sm.OLS(y, X).fit(cov_type='HC3')

print("\n=== УЛУЧШЕННЫЕ РЕЗУЛЬТАТЫ РЕГРЕССИИ (HC3) ===")
print(model.summary())

coef = model.params
pvals = model.pvalues