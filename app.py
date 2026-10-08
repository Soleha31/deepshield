import os
import re
import json
import sqlite3
import uuid
import html
from datetime import datetime
 
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash
)
 
import joblib
 
 
# =========================================================
# APP CONFIGURATION
# =========================================================
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
 
DB_PATH = os.path.join(BASE_DIR, "deepshield.db")
 
MODELS_DIR = os.path.join(BASE_DIR, "models")
 
 
# [CHANGE] .env file load karo (pehle MONGO_URI kabhi load hi nahi hota tha)
try:
    from dotenv import load_dotenv
 
    load_dotenv(os.path.join(BASE_DIR, ".env"))
 
except ImportError:
    pass
 
 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
 
app = Flask(__name__)
 
app.secret_key = os.environ.get("SECRET_KEY", "deepshield-secret-key")
 
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
 
 
# =========================================================
# OPTIONAL WHISPER
# =========================================================
 
try:
 
    from faster_whisper import WhisperModel
 
    WHISPER_AVAILABLE = True
 
except ImportError:
 
    WhisperModel = None
 
    WHISPER_AVAILABLE = False
 
 
WHISPER_MODEL = None
 
# [CHANGE] Whisper settings. Pehle "tiny" model aur auto language tha, jisse Hinglish
# call ka transcript toota-phoota aata tha. Ye sab environment variable se badal sakte ho.
# [RENDER-SAFE] Render apni RENDER variable khud set karta hai. Wahan free instance par RAM kam hoti hai
# (~512 MB), isliye default "tiny" model aur beam 1 rakha hai. Laptop par "small" aur beam 5.
ON_RENDER = bool(os.environ.get("RENDER"))
WHISPER_ENABLED = os.environ.get("WHISPER_ENABLED", "1") == "1"    # "0" = audio transcription band (sirf text paste)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny" if ON_RENDER else "small")   # tiny / base / small / medium
WHISPER_BEAM = int(os.environ.get("WHISPER_BEAM", "1" if ON_RENDER else "5"))
WHISPER_TASK = os.environ.get("WHISPER_TASK", "translate")         # translate = English text, transcribe = jo bola wahi
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "hi")        # "hi" Hindi/Hinglish, "en" English, "auto" khud pehchane
WHISPER_PROMPT = (
    "Namaste, main bank security team se bol raha hoon. "
    "OTP, UPI PIN, KYC, account, transaction, verification."
)
 
 
# =========================================================
# LOAD MACHINE LEARNING MODELS
# =========================================================
 
SCAM_MODEL_PATH = os.path.join(MODELS_DIR, "scam_classifier.pkl")
 
MANIPULATION_MODEL_PATH = os.path.join(MODELS_DIR, "manipulation_model.pkl")
 
 
scam_model = None
manipulation_model = None
 
 
try:
 
    if os.path.exists(SCAM_MODEL_PATH):
 
        scam_model = joblib.load(SCAM_MODEL_PATH)
 
        print("Loaded scam_classifier.pkl")
 
    else:
 
        print("WARNING: scam_classifier.pkl not found.")
 
 
except Exception as exc:
 
    print("Could not load scam classifier:", exc)
 
 
try:
 
    if os.path.exists(MANIPULATION_MODEL_PATH):
 
        manipulation_model = joblib.load(MANIPULATION_MODEL_PATH)
 
        print("Loaded manipulation_model.pkl")
 
    else:
 
        print("WARNING: manipulation_model.pkl not found.")
 
 
except Exception as exc:
 
    print("Could not load manipulation model:", exc)
 
 
# =========================================================
# MANIPULATION LABELS
# =========================================================
 
TACTIC_LABELS = {
 
    "urgency": "Urgency / Time Pressure",
 
    "fear": "Fear / Threat",
 
    "authority_impersonation": "Authority Impersonation",
 
    "reward_bait": "Reward / Prize Bait",
 
    "financial_pressure": "Financial Pressure"
 
}
 
 
# =========================================================
# DATABASE
# =========================================================
 
def get_db():
 
    connection = sqlite3.connect(DB_PATH)
 
    connection.row_factory = sqlite3.Row
 
    return connection
 
 
def init_db():
 
    connection = get_db()
 
    cursor = connection.cursor()
 
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
 
            id TEXT PRIMARY KEY,
 
            created_at TEXT,
 
            filename TEXT,
 
            transcript TEXT,
 
            risk_score INTEGER,
 
            risk_level TEXT,
 
            scam_detected INTEGER,
 
            scam_probability INTEGER,
 
            scam_type TEXT,
 
            scam_intent TEXT,
 
            social_engineering TEXT,
 
            impersonation TEXT,
 
            language TEXT,
 
            summary TEXT,
 
            evidence TEXT,
 
            evidence_phrases TEXT,
 
            manipulation_tactics TEXT,
 
            suspicious_sentences TEXT,
 
            recommendations TEXT
 
        )
        """
    )
 
    # [CHANGE] Purani deepshield.db mein ye 2 naye columns nahi hote, to unhe jod do
    cursor.execute("PRAGMA table_info(analyses)")
 
    existing_columns = {row["name"] for row in cursor.fetchall()}
 
    for column in ("suspicious_sentences", "recommendations"):
 
        if column not in existing_columns:
 
            cursor.execute(
                "ALTER TABLE analyses ADD COLUMN " + column + " TEXT"
            )
 
    connection.commit()
 
    connection.close()
 
 
init_db()
 
 
# =========================================================
# MONGODB
# =========================================================
 
mongo_client = None
mongo_collection = None
 
 
def setup_mongodb():
 
    global mongo_client
    global mongo_collection
 
    # [CHANGE] MONGO_URI ya MONGODB_URI, dono chalenge; quotes/spaces hat jayenge
    mongo_uri = (
        os.environ.get("MONGO_URI")
        or os.environ.get("MONGODB_URI")
        or ""
    ).strip().strip('"').strip("'")
 
    if not mongo_uri:
 
        print("MONGO_URI not found. Using SQLite.")
 
        return False
 
 
    try:
 
        from pymongo import MongoClient
 
        client_options = {"serverSelectionTimeoutMS": 10000}
 
        # [CHANGE] SSL certificates ke liye certifi (Windows par handshake error kam karta hai)
        try:
 
            import certifi
 
            client_options["tlsCAFile"] = certifi.where()
 
        except ImportError:
 
            pass
 
        mongo_client = MongoClient(mongo_uri, **client_options)
 
        mongo_client.admin.command("ping")
 
        database_name = os.environ.get("MONGO_DB", "deepshield")
 
        mongo_db = mongo_client[database_name]
 
        mongo_collection = mongo_db["analyses"]
 
        print("MongoDB connected successfully.")
 
        return True
 
 
    except Exception as exc:
 
        print("MongoDB connection failed:", exc)
 
        mongo_client = None
 
        mongo_collection = None
 
        print("Using SQLite fallback.")
 
        return False
 
 
setup_mongodb()
 
 
# =========================================================
# TEXT HELPERS
# =========================================================
 
def clean_text(text):
 
    text = text or ""
 
    text = re.sub(r"\s+", " ", text)
 
    return text.strip()
 
 
def split_sentences(text):
 
    text = clean_text(text)
 
    if not text:
 
        return []
 
 
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
 
 
    return [part.strip() for part in parts if part.strip()]
 
 
# [CHANGE] Models jis tarah ke text par train hue the, wahi cleaning yahan bhi chahiye
# (speaker tags hatao, URL -> urltoken, lowercase, digits -> 0). Pehle sirf spaces
# theek hote the, isliye model ko alag type ka text milta tha.
SPEAKER_TAGS = re.compile(
    r"\b(innocent|suspect|caller|receiver|scammer|victim|agent|user)\s*:",
    re.IGNORECASE
)
 
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|\b\S+\.(com|in|co|org|net|info|xyz|app)\b\S*",
    re.IGNORECASE
)
 
 
def prepare_text(text, strip_speakers=True):
 
    text = str(text or "")
 
    if strip_speakers:
 
        text = SPEAKER_TAGS.sub(" ", text)
 
    text = URL_PATTERN.sub(" urltoken ", text)
 
    text = text.lower()
 
    text = re.sub(r"\d+", "0", text)
 
    text = re.sub(r"\s+", " ", text).strip()
 
    return text
 
 
# =========================================================
# MODEL PROBABILITY
# =========================================================
 
def model_probability(model, text):
 
    if model is None:
 
        return 0.0
 
 
    text = prepare_text(text)
 
    if not text:
 
        return 0.0
 
 
    try:
 
        probabilities = model.predict_proba([text])
 
        probabilities = probabilities[0]
 
        classes = list(model.classes_)
 
 
        scam_indexes = []
 
 
        for index, label in enumerate(classes):
 
            label_text = str(label).lower().strip()
 
 
            if (
                label_text in {"scam", "fraud", "spam", "1", "true", "yes"}
                or "scam" in label_text
                or "fraud" in label_text
            ):
 
                scam_indexes.append(index)
 
 
        if scam_indexes:
 
            probability = sum(probabilities[i] for i in scam_indexes)
 
            return float(max(0.0, min(1.0, probability)))
 
 
        # If binary classifier has two classes,
        # use the second class as positive class.
 
        if len(classes) == 2:
 
            return float(probabilities[1])
 
 
        return float(max(probabilities))
 
 
    except Exception as exc:
 
        print("Model probability error:", exc)
 
        return 0.0
 
 
# =========================================================
# MANIPULATION MODEL
# =========================================================
 
def manipulation_probabilities(text):
 
    result = {key: 0.0 for key in TACTIC_LABELS}
 
 
    if manipulation_model is None:
 
        return result
 
 
    text = prepare_text(text, strip_speakers=False)
 
    if not text:
 
        return result
 
 
    # [CHANGE] manipulation_model.pkl ek sklearn model nahi, ek dict hai:
    #   {"word_vectorizer", "char_vectorizer", "classifiers": {tactic: model}, "tactics"}
    # Isliye purana manipulation_model.predict_proba(...) hamesha fail hota tha aur
    # saari tactics 0 aati thi. Ab har tactic ka apna classifier chalta hai.
    try:
 
        from scipy.sparse import hstack
 
        features = hstack([
 
            manipulation_model["word_vectorizer"].transform([text]),
 
            manipulation_model["char_vectorizer"].transform([text])
 
        ]).tocsr()
 
 
        for key, classifier in manipulation_model["classifiers"].items():
 
            if key in result:
 
                result[key] = float(classifier.predict_proba(features)[0][1])
 
 
    except Exception as exc:
 
        print("Manipulation model error:", exc)
 
 
    return result
 
 
# =========================================================
# EVIDENCE PHRASES
# =========================================================
 
EVIDENCE_PATTERNS = [
 
    r"\botp\b",
 
    r"\bupi pin\b",
 
    r"\bupi\b",
 
    r"\bpin\b",
 
    r"\bpassword\b",
 
    r"\bcvv\b",
 
    r"\batm\b",
 
    r"\bbank account\b",
 
    r"\baccount will be blocked\b",
 
    r"\baccount blocked\b",
 
    r"\bverify your account\b",
 
    r"\bverify immediately\b",
 
    r"\btransfer money\b",
 
    r"\bsend money\b",
 
    r"\bpay immediately\b",
 
    r"\bpayment\b",
 
    r"\brefund\b",
 
    r"\bcustoms\b",
 
    r"\bparcel\b",
 
    r"\blottery\b",
 
    r"\bprize\b",
 
    r"\breward\b",
 
    r"\bpolice\b",
 
    r"\barrest\b",
 
    r"\blegal action\b",
 
    r"\bcase will be filed\b",
 
    r"\bclick this link\b",
 
    r"\bclick the link\b",
 
    r"\bshare the code\b",
 
    r"\bverification code\b",
 
    r"\burgent\b",
 
    r"\bimmediately\b",
 
    r"\bright now\b",
 
    r"\bwithin \d+ minutes?\b",
 
    r"\btoday\b"
 
]
 
 
def evidence_phrases(text):
 
    text_lower = clean_text(text).lower()
 
 
    found = []
 
 
    for pattern in EVIDENCE_PATTERNS:
 
        matches = re.findall(pattern, text_lower, flags=re.IGNORECASE)
 
 
        for match in matches:
 
            phrase = match.strip()
 
 
            if phrase and phrase not in found:
 
                found.append(phrase)
 
 
    return found[:20]
 
 
# [CHANGE] "Strong" signals: ye normal baat-cheet mein kam aate hain, isliye risk score
# ko inse zyada boost milta hai. Credential maangna sabse khatarnak signal hai.
CREDENTIAL_PATTERNS = [
 
    r"\botp\b",
 
    r"\bupi pin\b",
 
    r"\bpin\b",
 
    r"\bcvv\b",
 
    r"\bpassword\b",
 
    r"\bverification code\b",
 
    r"\bshare the code\b",
 
    r"\bcard number\b"
 
]
 
STRONG_PATTERNS = CREDENTIAL_PATTERNS + [
 
    r"\barrest\b",
 
    r"\blegal action\b",
 
    r"\bcase will be filed\b",
 
    r"\baccount will be blocked\b",
 
    r"\baccount blocked\b",
 
    r"\bwarrant\b",
 
    r"\bblock ho\b",
 
    r"\bband ho jayega\b",
 
    r"\banydesk\b",
 
    r"\bteamviewer\b",
 
    r"\bquicksupport\b",
 
    r"\bremote access\b",
 
    r"\btransfer money\b",
 
    r"\bsend money\b",
 
    r"\bpay immediately\b",
 
    r"\bprocessing fee\b",
 
    r"\bregistration fee\b"
 
]
 
 
def critical_signals(text):
 
    text_lower = clean_text(text).lower()
 
    credential = any(
        re.search(pattern, text_lower) for pattern in CREDENTIAL_PATTERNS
    )
 
    strong_hits = sum(
        1 for pattern in STRONG_PATTERNS if re.search(pattern, text_lower)
    )
 
    return credential, strong_hits
 
 
# =========================================================
# HIGHLIGHT TRANSCRIPT
# =========================================================
 
def highlight_transcript(text):
 
    safe_text = html.escape(text or "")
 
 
    patterns = [
 
        r"\bOTP\b",
 
        r"\bUPI PIN\b",
 
        r"\bUPI\b",
 
        r"\bPIN\b",
 
        r"\bpassword\b",
 
        r"\bCVV\b",
 
        r"\bbank account\b",
 
        r"\baccount will be blocked\b",
 
        r"\bverify immediately\b",
 
        r"\btransfer money\b",
 
        r"\bsend money\b",
 
        r"\bpay immediately\b",
 
        r"\brefund\b",
 
        r"\bcustoms\b",
 
        r"\bparcel\b",
 
        r"\blottery\b",
 
        r"\bprize\b",
 
        r"\breward\b",
 
        r"\bpolice\b",
 
        r"\barrest\b",
 
        r"\blegal action\b",
 
        r"\bclick this link\b",
 
        r"\bclick the link\b",
 
        r"\bshare the code\b",
 
        r"\bverification code\b",
 
        r"\burgent\b",
 
        r"\bimmediately\b",
 
        r"\bright now\b"
 
    ]
 
 
    for pattern in patterns:
 
        safe_text = re.sub(
 
            pattern,
 
            lambda match: "<mark>" + match.group(0) + "</mark>",
 
            safe_text,
 
            flags=re.IGNORECASE
 
        )
 
 
    return safe_text
 
 
# =========================================================
# RISK SCORE
# =========================================================
 
# [CHANGE] Risk score ab zyada sensitive hai:
#   - har active tactic ka bonus 5 se 8 (max 30)
#   - strong signals (OTP/PIN maangna, arrest ki dhamki, AnyDesk, fee) ka alag bonus (max 24)
#   - credential maangne + dabav/tactic ho to score kam se kam 90
#   - kisi ek sentence par model bahut sure ho to score kam se kam 85
#   - tactic bonus tab poora lagta hai jab scam model ya strong signal bhi shak kare
#     (manipulation model sirf scam messages par train hua hai, normal baat par bharosa kam)
 
def build_risk_score(
    scam_probability,
    tactics,
    evidence_count,
    strong_hits=0,
    credential=False,
    max_sentence_probability=0.0
):
 
    # Base score
    score = scam_probability * 100
 
    # Strong manipulation tactics
    active_tactics = sum(
        1
        for value in tactics.values()
        if value >= 0.55
    )
 
    tactic_factor = (
        1.0
        if (scam_probability >= 0.25 or strong_hits >= 1)
        else 0.5
    )
 
    # Tactics contribution
    score += min(
        30,
        round(active_tactics * 8 * tactic_factor)
    )
 
    # Evidence contribution
    score += min(
        15,
        evidence_count * 3
    )
 
    # Strong signals contribution
    score += min(
        24,
        strong_hits * 6
    )
 
    # Strong scam detection
    if scam_probability >= 0.50:
        score = max(score, 80)
 
    if scam_probability >= 0.70:
        score = max(score, 90)
 
    if scam_probability >= 0.85:
        score = max(score, 97)
 
    # Ek sentence bahut suspicious
    if max_sentence_probability >= 0.80:
        score = max(score, 85)
 
    # Strong signal + manipulation
    if strong_hits >= 1 and active_tactics >= 1:
        score = max(score, 75)
 
    # Credential (OTP / PIN / CVV / password) maangna
    if credential and scam_probability >= 0.30:
        score = max(score, 85)
 
    if credential and active_tactics >= 1:
        score = max(score, 90)
 
    if strong_hits >= 3:
        score = max(score, 92)
 
    # Very strong scam indicators
    if active_tactics >= 3 and evidence_count >= 3:
        score = max(score, 95)
 
    if active_tactics >= 4 and evidence_count >= 4:
        score = max(score, 98)
 
    # Keep between 0 and 100
    score = max(
        0,
        min(
            100,
            round(score)
        )
    )
 
    return int(score)
 
 
# =========================================================
# RISK LEVEL
# =========================================================
 
def risk_level(score):
 
    if score >= 70:
 
        return "HIGH RISK"
 
 
    if score >= 40:
 
        return "MEDIUM RISK"
 
 
    return "LOW RISK"
 
 
# =========================================================
# LANGUAGE DETECTION
# =========================================================
 
def detect_language(text):
 
    text_lower = (text or "").lower()
 
 
    hindi_words = [
 
        "hai",
        "haan",
        "aap",
        "apka",
        "mera",
        "mujhe",
        "karo",
        "karna",
        "paisa",
        "bank",
        "account",
        "abhi",
        "turant",
        "kyu",
        "kaise",
        "police"
 
    ]
 
 
    hindi_count = sum(
 
        1
        for word in hindi_words
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower)
 
    )
 
 
    if hindi_count >= 3:
 
        return "Hindi / Hinglish"
 
 
    return "English"
 
 
# =========================================================
# TRANSCRIPTION
# =========================================================
 
def transcribe_audio(path):
 
    global WHISPER_MODEL
 
 
    if not WHISPER_ENABLED:
 
        return (
            "",
            "Audio transcription is switched off on this server. "
            "Please paste the transcript instead."
        )
 
 
    if not WHISPER_AVAILABLE:
 
        return (
            "",
            "Whisper is not installed. "
            "Run: pip install faster-whisper"
        )
 
 
    try:
 
        if WHISPER_MODEL is None:
 
            print("Loading Whisper " + WHISPER_MODEL_SIZE + " model...")
 
            WHISPER_MODEL = WhisperModel(
 
                WHISPER_MODEL_SIZE,
 
                device="cpu",
 
                compute_type="int8"
 
            )
 
 
        language = (
            None
            if WHISPER_LANGUAGE.lower() in ("", "auto")
            else WHISPER_LANGUAGE
        )
 
 
        segments, info = WHISPER_MODEL.transcribe(
 
            path,
 
            language=language,
 
            task=WHISPER_TASK,
 
            beam_size=WHISPER_BEAM,
 
            temperature=0.0,
 
            vad_filter=True,
 
            condition_on_previous_text=False,
 
            initial_prompt=WHISPER_PROMPT
 
        )
 
 
        text = " ".join(
 
            segment.text.strip()
 
            for segment in segments
 
            if segment.text.strip()
 
        ).strip()
 
 
        detected = getattr(info, "language", None)
 
 
        return (
            text,
            detected or language or "unknown"
        )
 
 
    except Exception as exc:
 
        print("Whisper transcription failed:", exc)
 
 
        return (
            "",
            "Transcription failed: " + str(exc)
        )
 
 
# =========================================================
# ANALYZE TEXT
# =========================================================
 
def analyze_text(text, spoken_language=None):
 
    text = clean_text(text)
 
 
    if not text:
 
        text = "No transcript was generated."
 
 
    # -------------------------
    # MAIN MODEL
    # -------------------------
 
    scam_probability = model_probability(scam_model, text)
 
 
    # -------------------------
    # MANIPULATION MODEL
    # -------------------------
 
    tactics = manipulation_probabilities(text)
 
 
    # -------------------------
    # EVIDENCE
    # -------------------------
 
    evidence = evidence_phrases(text)
 
    credential, strong_hits = critical_signals(text)
 
 
    # -------------------------
    # SENTENCE ANALYSIS
    # -------------------------
 
    sentence_results = []
 
    max_sentence_probability = 0.0
 
 
    for sentence in split_sentences(text):
 
        probability = model_probability(scam_model, sentence)
 
        max_sentence_probability = max(max_sentence_probability, probability)
 
 
        sentence_tactics = manipulation_probabilities(sentence)
 
 
        max_tactic = max(sentence_tactics.values(), default=0.0)
 
 
        if probability >= 0.55 or max_tactic >= 0.55:
 
            sentence_results.append({
 
                "text": sentence,
 
                "scam_probability": round(probability * 100),
 
                "tactics": [
 
                    TACTIC_LABELS[key]
 
                    for key, value in sentence_tactics.items()
 
                    if value >= 0.55
 
                ]
 
            })
 
 
    sentence_results.sort(
 
        key=lambda item: item["scam_probability"],
 
        reverse=True
 
    )
 
 
    sentence_results = sentence_results[:8]
 
 
    # -------------------------
    # RISK SCORE
    # -------------------------
 
    score = build_risk_score(
 
        scam_probability,
 
        tactics,
 
        len(evidence),
 
        strong_hits,
 
        credential,
 
        max_sentence_probability
 
    )
 
 
    detected = (
 
        scam_probability >= 0.50
        or
        score >= 50
 
    )
 
 
    level = risk_level(score)
 
 
    # -------------------------
    # ACTIVE TACTICS
    # -------------------------
 
    active_tactics = [
 
        TACTIC_LABELS[key]
 
        for key, probability in tactics.items()
 
        if probability >= 0.55
 
    ]
 
 
    # -------------------------
    # SCAM TYPE
    # -------------------------
 
    if scam_probability >= 0.75:
 
        scam_type = "AI-detected Voice Scam"
 
    elif active_tactics:
 
        scam_type = "Suspicious Social Engineering"
 
    elif detected:
 
        # [CHANGE] Score high ho par model/tactics kam hon, to "No Strong Scam Pattern"
        # likhna galat lagta tha
        scam_type = "Suspicious Voice Scam"
 
    else:
 
        scam_type = "No Strong Scam Pattern"
 
 
    # -------------------------
    # INTENT
    # -------------------------
 
    financial_words = [
 
        "money",
        "payment",
        "upi",
        "bank",
        "account",
        "pin",
        "otp",
        "transfer",
        "refund",
        "fee",
        "cash"
 
    ]
 
 
    financial_intent = any(
 
        word in text.lower()
 
        for word in financial_words
 
    )
 
 
    if financial_intent:
 
        scam_intent = "Possible financial or credential theft"
 
    elif detected:
 
        scam_intent = "Possible deceptive or fraudulent request"
 
    else:
 
        scam_intent = "No strong malicious intent detected"
 
 
    # -------------------------
    # SOCIAL ENGINEERING
    # -------------------------
 
    if active_tactics:
 
        social_engineering = "Detected"
 
    else:
 
        social_engineering = "Not strongly detected"
 
 
    # -------------------------
    # IMPERSONATION
    # -------------------------
 
    authority_words = [
 
        "bank",
        "police",
        "government",
        "customs",
        "officer",
        "customer care",
        "rbi",
        "income tax"
 
    ]
 
 
    impersonation_found = any(
 
        word in text.lower()
 
        for word in authority_words
 
    )
 
 
    if impersonation_found:
 
        impersonation = "Possible"
 
    else:
 
        impersonation = "Not detected"
 
 
    # -------------------------
    # SUMMARY
    # -------------------------
 
    if detected:
 
        summary = (
 
            "DeepShield detected suspicious "
            "signals in this conversation. "
            "The result is based on the trained "
            "scam classifier together with "
            "social-engineering indicators "
            "and supporting evidence."
 
        )
 
    else:
 
        summary = (
 
            "DeepShield did not detect a strong "
            "scam pattern in this conversation. "
            "However, users should still verify "
            "unexpected requests independently."
 
        )
 
 
    # -------------------------
    # RECOMMENDATIONS
    # -------------------------
 
    recommendations = []
 
 
    if detected:
 
        recommendations.extend([
 
            "Do not share OTP, UPI PIN, password or CVV.",
 
            "Do not transfer money because of pressure from the caller.",
 
            "Verify the caller through the organisation's official website or phone number.",
 
            "If the caller claims to be from a bank or government department, contact the organisation directly.",
 
            "End the call if the caller continues using threats or pressure."
 
        ])
 
    else:
 
        recommendations.extend([
 
            "Do not share confidential banking information with unknown callers.",
 
            "Verify unexpected requests using official contact information.",
 
            "If the conversation becomes suspicious, end the call and investigate independently."
 
        ])
 
 
    # -------------------------
    # LANGUAGE
    # -------------------------
 
    language = detect_language(text)
 
    # [CHANGE] Whisper "translate" ke baad transcript English hota hai, to language galat
    # "English" dikhti. Jo asli bhasha Whisper ne pakdi wahi use karo.
    if spoken_language in ("hi", "ur"):
 
        language = "Hindi / Hinglish"
 
 
    # -------------------------
    # RETURN RESULT
    # -------------------------
 
    return {
 
        "risk_score": score,
 
        "risk_level": level,
 
        "scam_detected": detected,
 
        "scam_probability": round(scam_probability * 100),
 
        "scam_type": scam_type,
 
        "scam_intent": scam_intent,
 
        "social_engineering": social_engineering,
 
        "impersonation": impersonation,
 
        "manipulation_tactics": active_tactics,
 
        "evidence": evidence,
 
        "evidence_phrases": evidence,
 
        "suspicious_sentences": sentence_results,
 
        "recommendations": recommendations,
 
        "language": language,
 
        "transcript": text,
 
        "highlighted_transcript": highlight_transcript(text),
 
        "summary": summary
 
    }
 
 
# =========================================================
# SAVE ANALYSIS
# =========================================================
 
def save_analysis(result, filename=""):
 
    analysis_id = str(uuid.uuid4())
 
 
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
 
    data = {
 
        "id": analysis_id,
 
        "created_at": created_at,
 
        "filename": filename or "",
 
        "transcript": result["transcript"],
 
        "risk_score": result["risk_score"],
 
        "risk_level": result["risk_level"],
 
        "scam_detected": bool(result["scam_detected"]),
 
        "scam_probability": result["scam_probability"],
 
        "scam_type": result["scam_type"],
 
        "scam_intent": result["scam_intent"],
 
        "social_engineering": result["social_engineering"],
 
        "impersonation": result["impersonation"],
 
        "language": result["language"],
 
        "summary": result["summary"],
 
        "evidence": result["evidence"],
 
        "evidence_phrases": result["evidence_phrases"],
 
        "manipulation_tactics": result["manipulation_tactics"],
 
        # [CHANGE] Pehle ye save nahi hote the, isliye result page par suspicious
        # sentences hamesha khali aur recommendations generic dikhte the
        "suspicious_sentences": result["suspicious_sentences"],
 
        "recommendations": result["recommendations"]
 
    }
 
 
    # =====================================================
    # MONGODB
    # =====================================================
 
    if mongo_collection is not None:
 
        try:
 
            mongo_collection.insert_one(dict(data))
 
            print("Analysis saved to MongoDB.")
 
            return analysis_id
 
        except Exception as exc:
 
            print("MongoDB save failed:", exc)
 
 
    # =====================================================
    # SQLITE FALLBACK
    # =====================================================
 
    connection = get_db()
 
    cursor = connection.cursor()
 
 
    cursor.execute(
 
        """
        INSERT INTO analyses (
 
            id,
            created_at,
            filename,
            transcript,
            risk_score,
            risk_level,
            scam_detected,
            scam_probability,
            scam_type,
            scam_intent,
            social_engineering,
            impersonation,
            language,
            summary,
            evidence,
            evidence_phrases,
            manipulation_tactics,
            suspicious_sentences,
            recommendations
 
        )
 
        VALUES (
 
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?
 
        )
        """,
 
        (
 
            data["id"],
 
            data["created_at"],
 
            data["filename"],
 
            data["transcript"],
 
            data["risk_score"],
 
            data["risk_level"],
 
            int(data["scam_detected"]),
 
            data["scam_probability"],
 
            data["scam_type"],
 
            data["scam_intent"],
 
            data["social_engineering"],
 
            data["impersonation"],
 
            data["language"],
 
            data["summary"],
 
            "|".join(data["evidence"]),
 
            "|".join(data["evidence_phrases"]),
 
            "|".join(data["manipulation_tactics"]),
 
            json.dumps(data["suspicious_sentences"]),
 
            json.dumps(data["recommendations"])
 
        )
 
    )
 
 
    connection.commit()
 
    connection.close()
 
 
    print("Analysis saved to SQLite.")
 
 
    return analysis_id
 
 
# =========================================================
# GET ONE ANALYSIS
# =========================================================
 
DEFAULT_RECOMMENDATIONS = [
 
    "Do not share OTP, UPI PIN, password or CVV.",
 
    "Verify suspicious requests using official channels.",
 
    "End the call if the caller pressures you."
 
]
 
 
def get_analysis(analysis_id):
 
    # -----------------------------------------------------
    # MongoDB
    # -----------------------------------------------------
 
    if mongo_collection is not None:
 
        try:
 
            document = mongo_collection.find_one({"id": analysis_id})
 
 
            if document:
 
                document.pop("_id", None)
 
                # [CHANGE] Mongo wale result mein ye keys nahi hoti thi, to
                # result page par highlighted transcript khali aata
                document["highlighted_transcript"] = highlight_transcript(
                    document.get("transcript", "")
                )
 
                document.setdefault("suspicious_sentences", [])
 
                document.setdefault("recommendations", DEFAULT_RECOMMENDATIONS)
 
                return document
 
 
        except Exception as exc:
 
            print("MongoDB read failed:", exc)
 
 
    # -----------------------------------------------------
    # SQLite
    # -----------------------------------------------------
 
    connection = get_db()
 
    cursor = connection.cursor()
 
 
    cursor.execute(
 
        """
        SELECT *
        FROM analyses
        WHERE id = ?
        """,
 
        (analysis_id,)
 
    )
 
 
    row = cursor.fetchone()
 
 
    connection.close()
 
 
    if not row:
 
        return None
 
 
    data = dict(row)
 
 
    data["scam_detected"] = bool(data["scam_detected"])
 
 
    data["evidence"] = (
 
        data["evidence"].split("|")
 
        if data["evidence"]
 
        else []
 
    )
 
 
    data["evidence_phrases"] = (
 
        data["evidence_phrases"].split("|")
 
        if data["evidence_phrases"]
 
        else []
 
    )
 
 
    data["manipulation_tactics"] = (
 
        data["manipulation_tactics"].split("|")
 
        if data["manipulation_tactics"]
 
        else []
 
    )
 
 
    data["highlighted_transcript"] = highlight_transcript(
        data.get("transcript", "")
    )
 
 
    # [CHANGE] Ab save kiye hue suspicious sentences / recommendations wapas padhte hain
    # (purani rows jinme ye khali hai, unke liye pehle jaisa default)
    try:
 
        data["suspicious_sentences"] = json.loads(
            data.get("suspicious_sentences") or "[]"
        )
 
    except ValueError:
 
        data["suspicious_sentences"] = []
 
 
    try:
 
        data["recommendations"] = (
            json.loads(data.get("recommendations") or "[]")
            or DEFAULT_RECOMMENDATIONS
        )
 
    except ValueError:
 
        data["recommendations"] = DEFAULT_RECOMMENDATIONS
 
 
    return data
 
 
# =========================================================
# GET HISTORY
# =========================================================
 
def get_history():
 
    # -----------------------------------------------------
    # MongoDB
    # -----------------------------------------------------
 
    if mongo_collection is not None:
 
        try:
 
            documents = list(
 
                mongo_collection.find(
                    {},
                    {"_id": 0}
                ).sort(
                    "created_at",
                    -1
                )
 
            )
 
 
            return documents
 
 
        except Exception as exc:
 
            print("MongoDB history failed:", exc)
 
 
    # -----------------------------------------------------
    # SQLite
    # -----------------------------------------------------
 
    connection = get_db()
 
    cursor = connection.cursor()
 
 
    cursor.execute(
 
        """
        SELECT *
        FROM analyses
        ORDER BY created_at DESC
        """
    )
 
 
    rows = cursor.fetchall()
 
 
    connection.close()
 
 
    history = []
 
 
    for row in rows:
 
        item = dict(row)
 
 
        item["scam_detected"] = bool(item["scam_detected"])
 
 
        history.append(item)
 
 
    return history
 
 
# =========================================================
# HOME
# =========================================================
 
@app.route("/")
def index():
 
    return render_template("index.html")
 
 
# =========================================================
# ANALYZE PAGE
# =========================================================
 
@app.route("/analyze", methods=["GET"])
def analyze_page():
 
    return render_template("analyze.html")
 
 
# =========================================================
# ANALYZE CALL
# =========================================================
 
@app.route("/analyze", methods=["POST"])
def analyze():
 
    transcript = clean_text(request.form.get("transcript", ""))
 
 
    audio_file = request.files.get("audio")
 
 
    filename = ""
 
    spoken_language = None
 
 
    # =====================================================
    # AUDIO
    # =====================================================
 
    if audio_file and audio_file.filename:
 
        filename = audio_file.filename
 
 
        safe_name = (
            str(uuid.uuid4())
            + "_"
            + os.path.basename(audio_file.filename)
        )
 
 
        audio_path = os.path.join(UPLOAD_FOLDER, safe_name)
 
 
        audio_file.save(audio_path)
 
 
        # If transcript was not manually provided,
        # use Whisper.
 
        if not transcript:
 
            transcript, detected_language = transcribe_audio(audio_path)
 
 
            if not transcript:
 
                flash(
                    "Audio transcription failed. "
                    + detected_language,
                    "error"
                )
 
                return redirect(url_for("analyze_page"))
 
 
            spoken_language = detected_language
 
 
    # =====================================================
    # EMPTY CHECK
    # =====================================================
 
    if not transcript:
 
        flash("Please upload audio or enter a transcript.", "error")
 
        return redirect(url_for("analyze_page"))
 
 
    # =====================================================
    # ANALYZE
    # =====================================================
 
    result = analyze_text(transcript, spoken_language)
 
 
    # =====================================================
    # SAVE
    # =====================================================
 
    analysis_id = save_analysis(result, filename)
 
 
    return redirect(url_for("result", analysis_id=analysis_id))
 
 
# =========================================================
# RESULT PAGE
# =========================================================
 
@app.route("/result/<analysis_id>")
def result(analysis_id):
 
    analysis = get_analysis(analysis_id)
 
 
    if not analysis:
 
        flash("Analysis not found.", "error")
 
        return redirect(url_for("history"))
 
 
    return render_template("result.html", result=analysis)
 
 
# =========================================================
# HISTORY PAGE
# =========================================================
 
@app.route("/history")
def history():
 
    records = get_history()
 
 
    return render_template("history.html", history=records)
 
 
# =========================================================
# REPORTS PAGE
# =========================================================
 
@app.route("/reports")
def reports():
 
    records = get_history()
 
 
    return render_template("reports.html", reports=records)
 
 
# =========================================================
# PDF REPORT
# =========================================================
 
@app.route("/report/<analysis_id>/pdf")
def download_report(analysis_id):
 
    analysis = get_analysis(analysis_id)
 
 
    if not analysis:
 
        flash("Analysis not found.", "error")
 
        return redirect(url_for("reports"))
 
 
    try:
 
        from reportlab.lib.pagesizes import A4
 
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )
 
        from reportlab.lib import colors
 
        from reportlab.lib.styles import getSampleStyleSheet
 
        from reportlab.lib.enums import TA_CENTER
 
 
    except ImportError:
 
        flash(
 
            "PDF support is not installed. "
            "Run: pip install reportlab",
 
            "error"
 
        )
 
        return redirect(url_for("result", analysis_id=analysis_id))
 
 
    pdf_filename = "DeepShield_Report_" + analysis_id[:8] + ".pdf"
 
 
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
 
 
    document = SimpleDocTemplate(
 
        pdf_path,
 
        pagesize=A4,
 
        rightMargin=40,
 
        leftMargin=40,
 
        topMargin=40,
 
        bottomMargin=40
 
    )
 
 
    styles = getSampleStyleSheet()
 
 
    title_style = styles["Title"]
 
    title_style.alignment = TA_CENTER
 
 
    story = []
 
 
    story.append(Paragraph("DeepShield - AI Scam Analysis Report", title_style))
 
    story.append(Spacer(1, 20))
 
 
    table_data = [
 
        ["Risk Score", str(analysis.get("risk_score", 0)) + "/100"],
 
        ["Risk Level", analysis.get("risk_level", "")],
 
        ["Scam Probability", str(analysis.get("scam_probability", 0)) + "%"],
 
        [
            "Result",
            (
                "SCAM PATTERN DETECTED"
                if analysis.get("scam_detected")
                else "NO STRONG SCAM PATTERN"
            )
        ],
 
        ["Scam Type", analysis.get("scam_type", "")],
 
        ["Language", analysis.get("language", "")]
 
    ]
 
 
    table = Table(table_data, colWidths=[150, 330])
 
 
    table.setStyle(
 
        TableStyle([
 
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
 
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
 
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
 
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
 
            ("FONTSIZE", (0, 0), (-1, -1), 9),
 
            ("PADDING", (0, 0), (-1, -1), 7)
 
        ])
 
    )
 
 
    story.append(table)
 
    story.append(Spacer(1, 20))
 
 
    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
 
    story.append(
        Paragraph(
            html.escape(analysis.get("summary", "")),
            styles["BodyText"]
        )
    )
 
    story.append(Spacer(1, 15))
 
 
    story.append(Paragraph("<b>Suspicious Evidence</b>", styles["Heading2"]))
 
 
    evidence = analysis.get("evidence_phrases", [])
 
 
    if evidence:
 
        for item in evidence:
 
            story.append(
                Paragraph("• " + html.escape(str(item)), styles["BodyText"])
            )
 
    else:
 
        story.append(
            Paragraph("No specific evidence extracted.", styles["BodyText"])
        )
 
 
    story.append(Spacer(1, 15))
 
 
    story.append(Paragraph("<b>Manipulation Tactics</b>", styles["Heading2"]))
 
 
    tactics = analysis.get("manipulation_tactics", [])
 
 
    if tactics:
 
        for tactic in tactics:
 
            story.append(
                Paragraph("• " + html.escape(str(tactic)), styles["BodyText"])
            )
 
    else:
 
        story.append(
            Paragraph(
                "No strong manipulation tactics detected.",
                styles["BodyText"]
            )
        )
 
 
    story.append(Spacer(1, 15))
 
 
    story.append(Paragraph("<b>Transcript</b>", styles["Heading2"]))
 
 
    transcript = html.escape(analysis.get("transcript", ""))
 
 
    story.append(Paragraph(transcript, styles["BodyText"]))
 
 
    document.build(story)
 
 
    return send_file(
 
        pdf_path,
 
        as_attachment=True,
 
        download_name=pdf_filename
 
    )
 
 
# =========================================================
# ERROR HANDLERS
# =========================================================
 
@app.errorhandler(404)
def page_not_found(error):
 
    return (
 
        """
        <h2>DeepShield - Page Not Found</h2>
        <p>The requested page does not exist.</p>
        <a href="/">Go to Dashboard</a>
        """,
 
        404
 
    )
 
 
@app.errorhandler(500)
def server_error(error):
 
    return (
 
        """
        <h2>DeepShield - Server Error</h2>
        <p>Please check the terminal for the error.</p>
        <a href="/">Go to Dashboard</a>
        """,
 
        500
 
    )
 
 
# =========================================================
# RUN SERVER
# =========================================================
 
if __name__ == "__main__":
 
    print("")
    print("=" * 55)
    print("       DEEPSHIELD AI SCAM DETECTION")
    print("=" * 55)
 
    print(
        "Scam model:",
        "LOADED"
        if scam_model is not None
        else "NOT FOUND"
    )
 
    print(
        "Manipulation model:",
        "LOADED"
        if manipulation_model is not None
        else "NOT FOUND"
    )
 
    print(
        "Whisper:",
        (
            "AVAILABLE (" + WHISPER_MODEL_SIZE + ", " + WHISPER_TASK + ")"
            if WHISPER_ENABLED
            else "SWITCHED OFF"
        )
        if WHISPER_AVAILABLE
        else "NOT INSTALLED"
    )
 
    print(
        "MongoDB:",
        "CONNECTED"
        if mongo_collection is not None
        else "SQLITE FALLBACK"
    )
 
    print("")
    print("Local URL: http://127.0.0.1:8000")
 
    print("Network URL: http://0.0.0.0:8000")
 
    print("=" * 55)
    print("")
 
 
    app.run(
 
        host="0.0.0.0",
 
        port=int(os.environ.get("PORT", 8000)),
 
        debug=not ON_RENDER
 
    )
 