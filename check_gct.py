import pandas as pd

df = pd.read_csv("datasets/GCT_phase1_100k.csv")
lab = "human_verified_label"

# 1. Same text, conflicting labels?
g = df.groupby("text_transcript")[lab].nunique()
print("Unique texts:", len(g), "| conflicting-label texts:", int((g > 1).sum()))

# 2. Numeric columns ka label se correlation
num = df.select_dtypes("number").drop(columns=[lab], errors="ignore")
print("\nTop correlations with label:")
print(num.corrwith(df[lab]).sort_values(key=abs, ascending=False).head(10))

# 3. Dataset ki apni model_prediction label se match karti hai?
print("\nmodel_prediction vs label:")
print(pd.crosstab(df["model_prediction"], df[lab]))

# 4. Label ratio attack_type ke hisaab se
print("\nLabel mean by attack_type:")
print(df.groupby("attack_type")[lab].mean())

# 5. Scores label ke hisaab se
print("\nMean scores by label:")
print(df.groupby(lab)[["CMS", "urgency_score", "fear_trigger_score"]].mean())

# 6. Asli texts dekho
print("\nLABEL 1:")
print(df[df[lab] == 1]["text_transcript"].sample(5, random_state=1).to_string())
print("\nLABEL 0:")
print(df[df[lab] == 0]["text_transcript"].sample(5, random_state=1).to_string())