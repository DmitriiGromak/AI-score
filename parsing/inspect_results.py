import pandas as pd
import numpy as np

skills = pd.read_csv('data/all_vac_skills.csv')
rel_skills = skills['score']
print(f"Skills: {len(skills):,} total")
print(f"Relatedness  mean/median = {np.mean(rel_skills):.4f} / {np.median(rel_skills):.4f}")
print("Top 30 highest-relatedness skills:")
print(skills.sort_values(by='score', ascending=False).head(30))
vacs = pd.read_csv('data/all_vac_labeled.csv')
v_rel = vacs['ai_relatedness']
print(f"\nVacancies: {len(vacs):,} total")
print(f"Relatedness > 0.1 : { (v_rel > 0.1).sum():,}  ({(v_rel > 0.1).mean():.4%})")
print(f"Relatedness > 0.5 : { (v_rel > 0.5).sum():,}")
print(f"Relatedness > 0.8 : { (v_rel > 0.8).sum():,}")
print(f"Mean / median     = {v_rel.mean():.4f} / {np.median(v_rel):.4f}")
print("Percentiles:", np.percentile(v_rel, [50, 90, 95, 99, 99.5, 99.9, 100]))