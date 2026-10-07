import joblib
import numpy as np

m = joblib.load("models/scam_classifier.pkl")
names = np.array(m.named_steps["tfidf"].get_feature_names_out())
coef = m.named_steps["classifier"].coef_[0]
idx = np.argsort(coef)

print("NON-SCAM ke shabd:", list(names[idx[:25]]))
print("SCAM ke shabd:", list(names[idx[-25:]]))