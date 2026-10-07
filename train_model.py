# DeepShield - scam classifier training (multi-source, hold-out tested)
#
# Usage:
#     python train_model.py --inspect     # sirf datasets ka schema + label mapping dikhao (train nahi)
#     python train_model.py               # inspect + evaluate + train + save
#
# Folder structure expected (project root se):
#     datasets/agent_conversation_all.csv
#     datasets/India_Cyber_Scam_Hinglish_Dataset.csv
#     datasets/sentinel_dataset_multiclass_clean.csv
#     datasets/icfd-31k/...                      (optional, jo csv/json/jsonl/parquet mile wo load hoga)
#     datasets/scamshield/train.jsonl, val.jsonl, test.jsonl   (optional)
#
# GCT_phase1_100k.csv JAANBUJH KE use nahi hota (labels random nikle the).
 
import os
import re
import sys
import glob
import json
 
import joblib
import numpy as np
import pandas as pd
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
 
# --------------------------------------------------
# CONFIG
# --------------------------------------------------
 
DATA_DIR = "datasets"
MODEL_PATH = "models/scam_classifier.pkl"
META_PATH = "models/scam_classifier_metadata.json"
 
# Har source: file + (optional) manual overrides.
#   text_col     : text wale column ka naam
#   label_col    : label wale column ka naam
#   benign_values: label ki wo values jo "scam NAHI" hain (strings ya numbers)
# Agar --inspect mein mapping galat dikhe, yahan override karo. Example:
#   "sentinel": {"file": "...", "label_col": "category", "benign_values": ["legitimate"]}
SOURCES = {
    "agent_conversation": {"file": "agent_conversation_all.csv"},
    "india_hinglish": {"file": "India_Cyber_Scam_Hinglish_Dataset.csv"},
    "sentinel": {"file": "sentinel_dataset_multiclass_clean.csv"},
}
 
ICFD_DIR = os.path.join(DATA_DIR, "icfd-31k")
# Agar icfd ke columns auto-detect na ho, yahan set karo, jaise:
#   ICFD_CFG = {"text_col": "text", "label_col": "label", "benign_values": ["legit"]}
ICFD_CFG = {"text_col": "cumulative_text", "label_col": "final_verdict"}
# icfd mein har conversation ke kai "chunk" rows hain (cumulative_text badhta jata hai).
# True: har conversation ka sirf aakhri (sabse lamba) chunk rakho, taaki same call ke
# kai prefix rows train/test mein na bantein (leakage).
ICFD_KEEP_LAST_CHUNK = True
 
# ScamShield mein "General Spam / Telemarketing" ko is_scam=1 mana gaya hai, jabki wo
# fraud nahi, promotion hai (aur kai rows galat label ke lagte hain). DeepShield fraud ke liye hai,
# isliye aise rows hata do. False karoge to ye wapas aa jayenge.
SCAMSHIELD_DROP_SPAM = True
SHARDS_PER_SPLIT = 3        # icfd mein har split (train/val/test) ki itni shards hi padhi jayengi
 
MAX_PER_SOURCE = 20000      # bade source ko dominate karne se roko
MIN_ROWS_FOR_HOLDOUT = 200  # isse chhote source par hold-out test skip
RANDOM_STATE = 42
 
TEXT_CANDIDATES = [
    "text", "dialogue", "message", "sms", "transcript", "text_transcript",
    "content", "body", "sentence", "utterance", "conversation", "input",
    "cumulative_text",
]
LABEL_CANDIDATES = [
    "is_scam", "label", "labels", "target", "class", "category",
    "scam_type", "type", "fraud_type", "scam_category", "output",
    "final_verdict",
]
BENIGN_PATTERN = re.compile(
    r"legit|benign|safe|normal|^ham$|not[_ \-]?scam|non[_ \-]?scam|genuine|"
    r"clean|no[_ \-]?fraud|not[_ \-]?fraud|low[_ \-]?risk|no[_ \-]?risk|^none$|^no$|^false$|^0(\.0)?$",
    re.IGNORECASE,
)
 
SANITY_TEXTS = [
    "Aapka SBI account block ho jayega. Abhi OTP share karein warna 2 ghante mein account band.",
    "Sir main bank se bol raha hoon, KYC update karna hai, UPI PIN bata dijiye turant.",
    "Congratulations! You won Rs 25 lakh lottery. Click http://bit.ly/claim-prize to claim now.",
    "Your parcel is held at customs. Pay Rs 49 fee at this link to release it.",
    "Kal shaam ko ghar aa raha hoon, mummy ko bata dena.",
    "Hi, are we still meeting for lunch tomorrow at 1?",
    "Aapka Jio recharge of Rs 299 successful. Validity 28 din. MyJio app mein check karein.",
    "Your OTP for login is 482910. Do not share it with anyone. - HDFC Bank",
]
 
# --------------------------------------------------
# LOADING HELPERS
# --------------------------------------------------
 
 
def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")
 
 
def find_text_col(df):
    lower = {c.lower(): c for c in df.columns}
    for name in TEXT_CANDIDATES:
        if name in lower:
            return lower[name]
    # fallback: sabse lamba average string wala object column
    best, best_len = None, 0
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(500)
            if len(sample) and sample.str.len().mean() > best_len:
                best, best_len = c, sample.str.len().mean()
    return best
 
 
def find_label_col(df, text_col):
    lower = {c.lower(): c for c in df.columns}
    for name in LABEL_CANDIDATES:
        if name in lower and lower[name] != text_col:
            return lower[name]
    return None
 
 
def to_binary(series, benign_values=None):
    # Label ko 0 (scam nahi) / 1 (scam) mein badlo. Mapping return karta hai verify karne ke liye.
    s = series.copy()
    mapping = {}
 
    if benign_values is not None:
        benign = {str(v).strip().lower() for v in benign_values}
        for v in s.dropna().unique():
            mapping[v] = 0 if str(v).strip().lower() in benign else 1
        return s.map(mapping), mapping
 
    values = list(s.dropna().unique())
    num_like = all(isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) for v in values)
 
    if num_like:
        for v in values:
            mapping[v] = 0 if float(v) == 0 else 1  # 0 = scam nahi (assumption)
    else:
        for v in values:
            mapping[v] = 0 if BENIGN_PATTERN.search(str(v).strip()) else 1
 
    return s.map(mapping), mapping
 
 
SPEAKER_TAGS = re.compile(r"\b(innocent|suspect|caller|receiver|scammer|victim|agent|user)\s*:", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(com|in|co|org|net|info|xyz|app)\b\S*", re.IGNORECASE)
 
 
def clean_text(t):
    t = str(t)
    t = SPEAKER_TAGS.sub(" ", t)
    t = URL_PATTERN.sub(" urltoken ", t)
    t = t.lower()
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
 
 
def frame_from_df(name, path, df, cfg=None):
    cfg = cfg or {}
    text_col = cfg.get("text_col") or find_text_col(df)
    label_col = cfg.get("label_col") or find_label_col(df, text_col)
 
    info = {"name": name, "path": path, "columns": list(df.columns), "rows_raw": len(df)}
 
    if text_col is None or label_col is None:
        info["error"] = f"text_col={text_col}, label_col={label_col} - detect nahi hua"
        try:
            info["sample"] = df.head(3).to_string(max_colwidth=60)
        except Exception:  # noqa
            info["sample"] = ""
        return None, info
 
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip() != ""]
 
    y, mapping = to_binary(df[label_col], cfg.get("benign_values"))
    out = pd.DataFrame({"text": df[text_col].values, "y": y.values, "raw_label": df[label_col].astype(str).values})
    out = out.dropna(subset=["y"])
    out["y"] = out["y"].astype(int)
    out["source"] = name
 
    info.update({
        "text_col": text_col,
        "label_col": label_col,
        "mapping": {str(k): int(v) for k, v in mapping.items()},
        "rows": len(out),
        "binary_counts": out["y"].value_counts().to_dict(),
    })
    return out, info
 
 
def load_source(name, path, cfg=None):
    return frame_from_df(name, path, read_any(path), cfg)
 
 
def collect_files():
    # (source_name, path, cfg) list banao.
    items = []
 
    for name, cfg in SOURCES.items():
        path = os.path.join(DATA_DIR, cfg["file"])
        if os.path.exists(path):
            items.append((name, path, cfg))
        else:
            print(f"[skip] {name}: file nahi mili -> {path}")
 
    return items
 
 
def load_icfd():
    # icfd-31k folder ki parquet shards: har split se thodi shards padho (poora data nahi), 2 source banao:
    #   icfd_main         = train + validation + test shards
    #   icfd_cross_domain = cross_domain shards (hold-out ke liye alag)
    # dataset_manifest.jsonl / release_summary.json metadata hain, unhe skip kiya jata hai.
    results = []
    if not os.path.isdir(ICFD_DIR):
        return results
 
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("[icfd] parquet padhne ke liye pyarrow chahiye -> pip install pyarrow   (icfd abhi skip)")
        return results
 
    files = sorted(glob.glob(os.path.join(ICFD_DIR, "**", "*.parquet"), recursive=True))
    by_split = {}
    for p in files:
        base = os.path.basename(p)
        m = re.match(r"^(.*?)-\d+-of-\d+", base)
        key = m.group(1) if m else os.path.splitext(base)[0]
        by_split.setdefault(key, []).append(p)
 
    def pick(paths):
        n = min(SHARDS_PER_SPLIT, len(paths))
        idx = np.unique(np.linspace(0, len(paths) - 1, n).astype(int))
        return [paths[i] for i in idx]
 
    groups = {"icfd_main": [], "icfd_cross_domain": []}
    for key, paths in by_split.items():
        target = "icfd_cross_domain" if key.startswith("cross_domain") else "icfd_main"
        groups[target] += pick(paths)
 
    for name, paths in groups.items():
        if not paths:
            continue
        try:
            raw = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
            tcol = ICFD_CFG.get("text_col")
            if ICFD_KEEP_LAST_CHUNK and tcol in raw.columns and "conversation_uid" in raw.columns:
                before = len(raw)
                raw = raw.assign(_len=raw[tcol].astype(str).str.len())
                raw = raw.sort_values("_len").groupby("conversation_uid", sort=False).tail(1).drop(columns="_len")
                print(f"[{name}] har conversation ka aakhri chunk rakha: {before} -> {len(raw)} rows")
            results.append(frame_from_df(name, ICFD_DIR, raw, ICFD_CFG))
        except Exception as e:  # noqa
            print(f"[skip] {name}: load error -> {e}")
    return results
 
 
def load_scamshield():
    # ScamShield ko source_dataset ke hisaab se alag-alag sources mein todta hai,
    # taaki hold-out test har sub-dataset par alag se ho sake.
    folder = os.path.join(DATA_DIR, "scamshield")
    parts = [os.path.join(folder, f) for f in ("train.jsonl", "val.jsonl", "test.jsonl")]
    parts = [p for p in parts if os.path.exists(p)]
    if not parts:
        return []
 
    df = pd.concat([pd.read_json(p, lines=True) for p in parts], ignore_index=True)
 
    # India Hinglish dataset alag se load hota hai, double count/leakage se bachne ke liye hatao
    if "source_dataset" in df.columns:
        mask = df["source_dataset"].astype(str).str.contains("ysangam|Indian_Cyber", case=False, regex=True)
        print(f"[scamshield] India Hinglish overlap ke {int(mask.sum())} rows hata diye")
        df = df[~mask]
    else:
        df["source_dataset"] = "all"
 
    if SCAMSHIELD_DROP_SPAM and "head2_scam_intent" in df.columns:
        spam = (df["is_scam"] == 1) & df["head2_scam_intent"].astype(str).str.contains("spam|telemarket", case=False, regex=True)
        print(f"[scamshield] 'General Spam / Telemarketing' wale {int(spam.sum())} scam rows hata diye (fraud nahi, promotion)")
        df = df[~spam]
 
    results = []
    for sd, g in df.groupby("source_dataset"):
        name = "ss:" + re.sub(r"[^A-Za-z0-9_]+", "_", str(sd))
        intent = g["head2_scam_intent"].astype(str).values if "head2_scam_intent" in g.columns else [""] * len(g)
        out = pd.DataFrame({
            "text": g["text"].astype(str).values,
            "y": g["is_scam"].astype(int).values,
            "raw_label": intent,
            "source": name,
        })
        info = {
            "name": name,
            "path": folder,
            "rows": len(out),
            "binary_counts": out["y"].value_counts().to_dict(),
            "text_col": "text",
            "label_col": "is_scam",
            "mapping": {"0": 0, "1": 1},
        }
        results.append((out, info))
    return results
 
 
# --------------------------------------------------
# MODEL
# --------------------------------------------------
 
 
def make_model():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, max_features=150000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)),
    ])
    clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced", solver="liblinear")
    return Pipeline([("features", features), ("clf", clf)])
 
 
def cap_per_source(df):
    parts = []
    for _, g in df.groupby("source"):
        parts.append(g.sample(min(len(g), MAX_PER_SOURCE), random_state=RANDOM_STATE))
    return pd.concat(parts, ignore_index=True)
 
 
# --------------------------------------------------
# MAIN
# --------------------------------------------------
 
 
def main():
    inspect_only = "--inspect" in sys.argv
 
    print("=" * 62)
    print("  DEEPSHIELD SCAM CLASSIFIER")
    print("=" * 62)
 
    frames, infos = [], []
 
    for name, path, cfg in collect_files():
        try:
            df, info = load_source(name, path, cfg)
        except Exception as e:  # noqa
            print(f"[skip] {name}: load error -> {e}")
            continue
        infos.append(info)
        if df is None:
            print(f"[skip] {name}: {info['error']}")
            print(f"       columns: {info['columns']}")
            continue
        frames.append(df)
 
    extra = []
    for loader in (load_icfd, load_scamshield):
        try:
            extra += loader()
        except Exception as e:  # noqa
            print(f"[skip] {loader.__name__}: error -> {e}")
 
    for df, info in extra:
        infos.append(info)
        if df is None:
            print(f"[skip] {info['name']}: {info['error']}")
            print(f"       columns: {info['columns']}")
            if info.get("sample"):
                print("       sample rows:")
                print("       " + info["sample"].replace("\n", "\n       "))
            continue
        frames.append(df)
 
    if not frames:
        print("ERROR: Koi dataset load nahi hua. DATA_DIR aur file names check karo.")
        sys.exit(1)
 
    # ---------------- INSPECT REPORT ----------------
    print("\n" + "-" * 62)
    print("DATASET REPORT (mapping verify karo: 0 = scam nahi, 1 = scam)")
    print("-" * 62)
    for info in infos:
        if "error" in info:
            continue
        print(f"\n[{info['name']}]  rows={info['rows']}")
        print(f"  text column : {info['text_col']}")
        print(f"  label column: {info['label_col']}")
        print(f"  mapping     : {info['mapping']}")
        print(f"  0/1 counts  : {info['binary_counts']}")
        if info.get("overlap_removed"):
            print(f"  (India Hinglish overlap ke {info['overlap_removed']} rows hata diye)")
 
    all_df = pd.concat(frames, ignore_index=True)
 
    for src, g in all_df.groupby("source"):
        print(f"\n--- {src}: 2 benign (0) aur 2 scam (1) examples ---")
        for lab in (0, 1):
            sub = g[g["y"] == lab]
            for t in sub["text"].head(2):
                print(f"  [{lab}] {str(t)[:140]}")
 
    if inspect_only:
        print("\n--inspect mode: yahin ruk gaya. Mapping galat ho to SOURCES mein benign_values set karo.")
        return
 
    # ---------------- CLEAN ----------------
    all_df["clean"] = all_df["text"].map(clean_text)
    all_df = all_df[all_df["clean"].str.len() > 2]
 
    before = len(all_df)
    nun = all_df.groupby("clean")["y"].nunique()
    conflict = set(nun[nun > 1].index)
    all_df = all_df[~all_df["clean"].isin(conflict)]
    removed_conflict = before - len(all_df)
 
    before = len(all_df)
    all_df = all_df.drop_duplicates(subset="clean").reset_index(drop=True)
    removed_dup = before - len(all_df)
 
    print("\n" + "-" * 62)
    print(f"Conflicting-label rows removed : {removed_conflict}")
    print(f"Duplicate rows removed         : {removed_dup}")
    print(f"Rows after cleaning            : {len(all_df)}")
    print(all_df.groupby("source")["y"].agg(["count", "mean"]).rename(columns={"mean": "scam_ratio"}))
 
    # ---------------- LEAVE-ONE-SOURCE-OUT ----------------
    print("\n" + "=" * 62)
    print("  HOLD-OUT TEST: ek source par test, baaki par train")
    print("  (ye asli accuracy hai - same-dataset accuracy se zyada bharosemand)")
    print("=" * 62)
 
    loso = {}
    for src in all_df["source"].unique():
        test = all_df[all_df["source"] == src]
        train = all_df[all_df["source"] != src]
 
        if len(test) < MIN_ROWS_FOR_HOLDOUT or test["y"].nunique() < 2 or train["y"].nunique() < 2:
            print(f"[{src}] skip (rows kam ya ek hi class)")
            continue
 
        train = cap_per_source(train)
        model = make_model()
        model.fit(train["clean"], train["y"])
        pred = model.predict(test["clean"])
 
        acc = accuracy_score(test["y"], pred)
        f1 = f1_score(test["y"], pred, average="macro")
        loso[src] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        print(f"[{src}] test rows={len(test)}  accuracy={acc * 100:.2f}%  macro-F1={f1:.3f}")
 
    # ---------------- IN-DISTRIBUTION SPLIT ----------------
    print("\n" + "=" * 62)
    print("  MIXED 80/20 SPLIT (saare sources mila ke)")
    print("=" * 62)
 
    capped = cap_per_source(all_df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        capped["clean"], capped["y"], test_size=0.2, random_state=RANDOM_STATE, stratify=capped["y"]
    )
    model = make_model()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Accuracy: {accuracy_score(y_te, pred) * 100:.2f}%")
    print(classification_report(y_te, pred, target_names=["not_scam", "scam"]))
 
    # ---------------- FINAL MODEL ----------------
    print("Final model (saara data par) train ho raha hai...")
    final = make_model()
    final.fit(capped["clean"], capped["y"])
 
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final, MODEL_PATH)
 
    meta = {
        "sources": sorted(all_df["source"].unique().tolist()),
        "rows_used": int(len(capped)),
        "holdout_results": loso,
        "preprocess": "clean_text(): speaker tags hatao, url->urltoken, lowercase, digits->0",
        "note": "Predict karne se pehle text ko train_model.clean_text() se clean karo.",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
 
    print(f"\nSaved model   : {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")
 
    # ---------------- SANITY ----------------
    print("\n" + "=" * 62)
    print("  SANITY CHECK (naye sentences)")
    print("=" * 62)
    proba = final.predict_proba([clean_text(t) for t in SANITY_TEXTS])[:, 1]
    for t, p in zip(SANITY_TEXTS, proba):
        tag = "SCAM    " if p >= 0.5 else "not scam"
        print(f"{tag} ({p:.2f})  {t[:90]}")
 
 
if __name__ == "__main__":
    main()
 