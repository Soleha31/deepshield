# DeepShield - manipulation tactics model (ScamShield dataset)
#
# ScamShield mein har message ke saath 5 manipulation tactics ke flags (0/1) hain:
#     urgency, fear, authority_impersonation, reward_bait, financial_pressure
#
# Setup:
#     datasets/scamshield/train.jsonl
#     datasets/scamshield/val.jsonl
#     datasets/scamshield/test.jsonl
#
# Run:
#     python train_manipulation.py
 
import os
import re
import json
 
import joblib
import numpy as np
import pandas as pd
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
 
DATA_DIR = os.path.join("datasets", "scamshield")
MODEL_PATH = os.path.join("models", "manipulation_model.pkl")
META_PATH = os.path.join("models", "manipulation_model_metadata.json")
 
TACTICS = ["urgency", "fear", "authority_impersonation", "reward_bait", "financial_pressure"]
 
# True: sirf scam messages par train/test (benign messages mein koi manipulation nahi hota)
SCAM_ONLY = True
MIN_POSITIVES = 30
RANDOM_STATE = 42
 
# Source hold-out: in sources ko poori tarah train se bahar rakh ke test karte hain
HOLDOUT_SOURCES = ["Synthetic_Tier_C", "ysangam/Indian_Cyber_Scam_Hinglish", "Indian_Telecom_SMS"]
 
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(com|in|co|org|net|info|xyz|app)\b\S*", re.IGNORECASE)
 
 
def clean_text(t):
    t = str(t)
    t = URL_PATTERN.sub(" urltoken ", t)
    t = t.lower()
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
 
 
def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nahi mila: {path}")
    df = pd.read_json(path, lines=True)
 
    flags = pd.json_normalize(df["head1_social_engineering"])
    for t in TACTICS:
        df[t] = flags[t].astype(int) if t in flags.columns else 0
 
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len() > 2]
    return df.reset_index(drop=True)
 
 
def prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return round(float(p), 3), round(float(r), 3), round(float(f), 3)
 
 
def main():
    train, val, test = load_split("train"), load_split("val"), load_split("test")
 
    print("=" * 62)
    print("  MANIPULATION TACTICS MODEL")
    print("=" * 62)
    print(f"Rows  train={len(train)}  val={len(val)}  test={len(test)}")
 
    # --- Label quality check: kis source mein flags kitne hain? ---
    print("\nTactic ka positive rate (sirf scam rows), source ke hisaab se:")
    scam_rows = train[train["is_scam"] == 1]
    table = scam_rows.groupby("source_dataset")[TACTICS].mean().round(3)
    table["rows"] = scam_rows.groupby("source_dataset").size()
    print(table.to_string())
    print("\n(Jis source mein sab ~0.00 hain, wahan manipulation labels nahi hain - wo model ko 'no tactic' sikhayenge.)")
 
    full_scam = pd.concat([train, val, test], ignore_index=True)
    full_scam = full_scam[full_scam["is_scam"] == 1].reset_index(drop=True)
 
    if SCAM_ONLY:
        train = train[train["is_scam"] == 1]
        val = val[val["is_scam"] == 1]
        test = test[test["is_scam"] == 1]
        print(f"\nSCAM_ONLY=True -> train={len(train)} val={len(val)} test={len(test)}")
 
    # --- Vectorize ---
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, max_features=150000)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
 
    from scipy.sparse import hstack
    X_train = hstack([vec.fit_transform(train["clean"]), char_vec.fit_transform(train["clean"])]).tocsr()
    X_val = hstack([vec.transform(val["clean"]), char_vec.transform(val["clean"])]).tocsr()
    X_test = hstack([vec.transform(test["clean"]), char_vec.transform(test["clean"])]).tocsr()
 
    classifiers, results = {}, {}
 
    print("\n" + "=" * 62)
    print("  PER-TACTIC RESULT (test set)")
    print("=" * 62)
    print(f"{'tactic':26s} {'pos(train)':>10s} {'precision':>10s} {'recall':>8s} {'F1':>6s}")
 
    for t in TACTICS:
        pos = int(train[t].sum())
        if pos < MIN_POSITIVES or train[t].nunique() < 2:
            print(f"{t:26s} {pos:>10d}   skip (positive examples kam)")
            continue
 
        clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced", solver="liblinear")
        clf.fit(X_train, train[t])
 
        pred = clf.predict(X_test)
        p, r, f = prf(test[t], pred)
        classifiers[t] = clf
        results[t] = {"precision": p, "recall": r, "f1": f, "train_positives": pos}
        print(f"{t:26s} {pos:>10d} {p:>10.3f} {r:>8.3f} {f:>6.3f}")
 
    # --- Language-wise F1 ---
    if "language" in test.columns and classifiers:
        print("\nLanguage-wise F1 (test):")
        for lang in test["language"].unique():
            mask = (test["language"] == lang).values
            g = test[mask]
            if len(g) < 50:
                continue
            Xg = X_test[mask]
            row = []
            for t, clf in classifiers.items():
                if g[t].sum() == 0:
                    continue
                row.append(prf(g[t], clf.predict(Xg))[2])
            if row:
                print(f"  {lang:10s} rows={len(g):5d}  avg F1={np.mean(row):.3f}")
 
    # --- Source hold-out (asli generalization check) ---
    from scipy.sparse import hstack as _hstack
 
    print("\n" + "=" * 62)
    print("  SOURCE HOLD-OUT: ek source train se bahar, usi par test")
    print("  (same-source test se zyada sakht aur zyada imaandar)")
    print("=" * 62)
    holdout_results = {}
    for hs in HOLDOUT_SOURCES:
        te = full_scam[full_scam["source_dataset"] == hs]
        tr = full_scam[full_scam["source_dataset"] != hs]
        if len(te) < 100:
            print(f"[{hs}] skip (rows kam: {len(te)})")
            continue
 
        v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, max_features=150000)
        v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
        Xtr = _hstack([v1.fit_transform(tr["clean"]), v2.fit_transform(tr["clean"])]).tocsr()
        Xte = _hstack([v1.transform(te["clean"]), v2.transform(te["clean"])]).tocsr()
 
        row = {}
        for t in TACTICS:
            if te[t].sum() < 10 or tr[t].sum() < MIN_POSITIVES:
                row[t] = None
                continue
            c = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced", solver="liblinear")
            c.fit(Xtr, tr[t])
            row[t] = prf(te[t], c.predict(Xte))[2]
        holdout_results[hs] = row
        cells = "  ".join(f"{t[:8]}={('%.2f' % v) if v is not None else ' n/a'}" for t, v in row.items())
        print(f"[{hs}] rows={len(te)}  F1: {cells}")
 
    # --- Save ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"word_vectorizer": vec, "char_vectorizer": char_vec, "classifiers": classifiers, "tactics": list(classifiers)},
        MODEL_PATH,
    )
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"scam_only": SCAM_ONLY, "results": results, "source_holdout_f1": holdout_results}, f, indent=2)
 
    print(f"\nSaved: {MODEL_PATH}")
    print(f"Saved: {META_PATH}")
 
    # --- Demo ---
    demo = [
        "Alert: Aapka SBI account suspend ho gaya hai. Turant KYC update karein warna account permanently band ho jayega.",
        "Congratulations! You won a Rs 10 lakh lottery. Claim your prize now.",
        "Traffic Police Notice: Non-bailable warrant issued. Pay penalty immediately to Sub-Inspector.",
    ]
    print("\nDemo predictions:")
    Xd = hstack([vec.transform([clean_text(d) for d in demo]), char_vec.transform([clean_text(d) for d in demo])]).tocsr()
    for i, d in enumerate(demo):
        found = [t for t, clf in classifiers.items() if clf.predict(Xd[i])[0] == 1]
        print(f"- {d[:70]}...\n    tactics: {found if found else 'none'}")
 
 
if __name__ == "__main__":
    main()
