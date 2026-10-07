import os
import re
import sqlite3
from datetime import datetime

import certifi
import joblib
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, send_file
from markupsafe import Markup, escape
from pymongo import MongoClient, DESCENDING
from bson.objectid import ObjectId
from bson.errors import InvalidId

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_CENTER
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    WHISPER_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "deepshield-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# -----------------------------
# DATABASE
# -----------------------------
load_dotenv(os.path.join(BASE_DIR, ".env"))
MONGO_URI = (os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI") or "").strip().strip('"').strip("'")
MONGO_CONNECTED = False
client = None
db = None
analyses_col = None
SQLITE_DB = os.path.join(BASE_DIR, "deepshield.db")

SQLITE_COLUMNS = {
    "filename": "TEXT", "transcript": "TEXT", "scam_type": "TEXT", "risk_score": "INTEGER",
    "risk_level": "TEXT", "scam_probability": "REAL", "scam_detected": "INTEGER",
    "scam_intent": "TEXT", "social_engineering": "TEXT", "impersonation": "TEXT",
    "manipulation_tactics": "TEXT", "evidence": "TEXT", "suspicious_sentences": "TEXT",
    "evidence_phrases": "TEXT", "recommendations": "TEXT", "language": "TEXT",
    "transcription_status": "TEXT", "created_at": "TEXT", "created_at_ts": "TEXT"
}


def init_sqlite():
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    for name, dtype in SQLITE_COLUMNS.items():
        try:
            cur.execute(f"ALTER TABLE analyses ADD COLUMN {name} {dtype}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def connect_mongodb():
    global client, db, analyses_col, MONGO_CONNECTED
    if not MONGO_URI:
        print("MongoDB URI not found. Using SQLite.")
        return False
    try:
        client = MongoClient(
            MONGO_URI, tls=True, tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=10000, connectTimeoutMS=10000,
            socketTimeoutMS=15000, retryWrites=True, appname="DeepShield"
        )
        client.admin.command("ping")
        db = client["deepshield"]
        analyses_col = db["analyses"]
        analyses_col.create_index([("created_at_ts", DESCENDING)])
        MONGO_CONNECTED = True
        print("MongoDB Atlas connected")
        return True
    except Exception as exc:
        MONGO_CONNECTED = False
        print("MongoDB connection failed; using SQLite:", exc)
        return False


init_sqlite()
connect_mongodb()

# -----------------------------
# AI MODELS
# -----------------------------
SCAM_MODEL_PATH = os.path.join(MODEL_FOLDER, "scam_classifier.pkl")
MANIP_MODEL_PATH = os.path.join(MODEL_FOLDER, "manipulation_model.pkl")
scam_model = None
manipulation_model = None

try:
    scam_model = joblib.load(SCAM_MODEL_PATH)
    print("Scam classifier loaded")
except Exception as exc:
    print("Scam classifier could not be loaded:", exc)

try:
    manipulation_model = joblib.load(MANIP_MODEL_PATH)
    print("Manipulation model loaded")
except Exception as exc:
    print("Manipulation model could not be loaded:", exc)

WHISPER_MODEL = None


def clean_text(text):
    text = str(text or "")
    text = re.sub(r"\b(innocent|suspect|caller|receiver|scammer|victim|agent|user)\s*:", " ", text, flags=re.I)
    text = re.sub(r"https?://\S+|www\.\S+|\b\S+\.(com|in|co|org|net|info|xyz|app)\b\S*", " urltoken ", text, flags=re.I)
    text = text.lower()
    text = re.sub(r"\d+", "0", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def model_probability(model, text):
    if model is None or not text.strip():
        return 0.0
    cleaned = clean_text(text)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned])[0]
            classes = list(getattr(model, "classes_", [0, 1]))
            if 1 in classes:
                return float(probs[classes.index(1)])
            return float(probs[-1])
        pred = model.predict([cleaned])[0]
        return 1.0 if int(pred) == 1 else 0.0
    except Exception as exc:
        print("Scam model prediction failed:", exc)
        return 0.0


def manipulation_probabilities(text):
    result = {}
    if not manipulation_model or not text.strip():
        return result
    try:
        vec = manipulation_model["word_vectorizer"]
        char_vec = manipulation_model["char_vectorizer"]
        classifiers = manipulation_model["classifiers"]
        from scipy.sparse import hstack
        cleaned = clean_text(text)
        X = hstack([vec.transform([cleaned]), char_vec.transform([cleaned])]).tocsr()
        for tactic, clf in classifiers.items():
            try:
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X)[0]
                    classes = list(getattr(clf, "classes_", [0, 1]))
                    p = float(probs[classes.index(1)]) if 1 in classes else float(probs[-1])
                else:
                    p = float(clf.predict(X)[0])
                result[tactic] = p
            except Exception:
                result[tactic] = 0.0
    except Exception as exc:
        print("Manipulation model prediction failed:", exc)
    return result


TACTIC_LABELS = {
    "urgency": "Urgency / Pressure",
    "fear": "Fear / Threat",
    "authority_impersonation": "Authority Impersonation",
    "reward_bait": "Reward / Prize Bait",
    "financial_pressure": "Financial Pressure",
}

EVIDENCE_PATTERNS = {
    "OTP / verification code": [r"\botp\b", r"one time password", r"verification code", r"verify.*code"],
    "UPI / PIN / banking": [r"\bupi\b", r"upi pin", r"\bpin\b", r"\bcvv\b", r"bank account", r"account blocked", r"kyc"],
    "Money / payment request": [r"send money", r"transfer money", r"transfer.*amount", r"make.*payment", r"pay.*fee", r"payment"],
    "Urgency": [r"urgent", r"immediately", r"right now", r"turant", r"jaldi", r"abhi", r"within \d+ minutes", r"today only"],
    "Threat / fear": [r"account.*block", r"account.*suspend", r"account.*close", r"arrest", r"police", r"warrant", r"legal action", r"penalty"],
    "Authority claim": [r"bank officer", r"calling from.*bank", r"cyber crime", r"income tax", r"government officer", r"police officer", r"customer care", r"official"],
    "Reward / prize": [r"congratulations", r"you won", r"lottery", r"prize", r"reward", r"cashback"],
    "Suspicious link": [r"https?://", r"www\.", r"click.*link", r"link.*claim"],
}


def evidence_phrases(text):
    found = []
    for category, patterns in EVIDENCE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                phrase = match.group(0).strip()
                if phrase and not any(x["phrase"].lower() == phrase.lower() for x in found):
                    found.append({"category": category, "phrase": phrase})
                break
    return found


def detect_language(text):
    if not text.strip():
        return "Unknown"
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if devanagari > latin * 0.2:
        return "Hindi / Devanagari"
    hinglish_words = ["aapka", "hai", "ho gaya", "karo", "karein", "turant", "abhi", "nahi", "kyu", "paise"]
    low = text.lower()
    if sum(1 for w in hinglish_words if w in low) >= 2:
        return "Hinglish"
    return "English"


def split_sentences(text):
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def risk_level(score):
    if score >= 70:
        return "High Risk"
    if score >= 40:
        return "Medium Risk"
    return "Low Risk"


def build_risk_score(scam_probability, tactics, evidence_count):
    # Main model carries most of the score. Manipulation adds supporting evidence,
    # while repeated keyword evidence gives a small explainability bonus.
    tactic_values = [p for p in tactics.values() if p >= 0.55]
    tactic_component = min(20.0, sum(tactic_values) * 5.0)
    evidence_component = min(10.0, evidence_count * 2.5)
    score = round(min(100.0, scam_probability * 70.0 + tactic_component + evidence_component))
    return int(score)


def analyze_text(text):
    text = (text or "").strip()
    if not text:
        text = "No transcript was generated."

    scam_probability = model_probability(scam_model, text)
    tactics = manipulation_probabilities(text)
    evidence = evidence_phrases(text)

    sentence_results = []
    for sentence in split_sentences(text):
        p = model_probability(scam_model, sentence)
        tp = manipulation_probabilities(sentence)
        max_tactic = max(tp.values(), default=0.0)
        if p >= 0.55 or max_tactic >= 0.55:
            sentence_results.append({
                "text": sentence,
                "scam_probability": round(p * 100),
                "tactics": [TACTIC_LABELS[k] for k, v in tp.items() if v >= 0.55],
            })
    sentence_results.sort(key=lambda x: x["scam_probability"], reverse=True)
    sentence_results = sentence_results[:8]

    score = build_risk_score(scam_probability, tactics, len(evidence))
    detected = scam_probability >= 0.50 or score >= 50
    level = risk_level(score)

    active_tactics = [TACTIC_LABELS[k] for k, p in tactics.items() if p >= 0.55]
    if scam_probability >= 0.75:
        scam_type = "AI-detected Voice Scam"
    elif active_tactics:
        scam_type = "Suspicious Social Engineering"
    else:
        scam_type = "No Strong Scam Pattern"

    intent = []
    if any(x["category"] == "OTP / verification code" for x in evidence): intent.append("Credential / OTP harvesting")
    if any(x["category"] == "UPI / PIN / banking" for x in evidence): intent.append("Sensitive banking information")
    if any(x["category"] == "Money / payment request" for x in evidence): intent.append("Financial request")
    if any(x["category"] == "Reward / prize" for x in evidence): intent.append("Reward / prize bait")
    scam_intent = ", ".join(intent) if intent else ("Suspicious intent detected by AI" if detected else "No strong scam intent detected")

    impersonation = "Possible authority impersonation detected" if tactics.get("authority_impersonation", 0) >= 0.55 else "No clear impersonation detected"
    social = ", ".join(active_tactics) if active_tactics else "No strong manipulation tactic detected"

    recommendations = []
    if detected:
        recommendations.append("Do not share OTP, UPI PIN, CVV, passwords or banking credentials.")
        recommendations.append("Do not transfer money because of pressure from the caller.")
        recommendations.append("Verify the caller independently using an official number or app.")
    else:
        recommendations.append("No strong scam pattern was detected, but remain cautious with unexpected calls.")

    highlighted = highlight_transcript(text, [x["phrase"] for x in evidence])

    return {
        "risk_score": score,
        "risk_level": level,
        "scam_detected": detected,
        "scam_probability": round(scam_probability * 100, 1),
        "scam_type": scam_type,
        "scam_intent": scam_intent,
        "social_engineering": social,
        "impersonation": impersonation,
        "manipulation_tactics": [{"name": TACTIC_LABELS[k], "probability": round(v * 100, 1)} for k, v in tactics.items() if v >= 0.55],
        "evidence": [f'{x["category"]}: "{x["phrase"]}"' for x in evidence],
        "evidence_phrases": evidence,
        "suspicious_sentences": sentence_results,
        "recommendations": recommendations,
        "language": detect_language(text),
        "transcript": text,
        "highlighted_transcript": highlighted,
        "summary": (
            f"DeepShield detected a possible scam with {round(scam_probability * 100, 1)}% scam probability."
            if detected else
            f"DeepShield did not detect a strong scam pattern. Scam probability: {round(scam_probability * 100, 1)}%."
        ),
    }


def highlight_transcript(text, phrases):
    safe = str(escape(text))
    for phrase in sorted(set(phrases), key=len, reverse=True):
        if not phrase:
            continue
        safe = re.sub(
            re.escape(str(escape(phrase))),
            lambda m: f'<mark class="evidence-highlight">{m.group(0)}</mark>',
            safe,
            flags=re.I,
        )
    return Markup(safe.replace("\n", "<br>"))

# -----------------------------
# WHISPER
# -----------------------------

def transcribe_audio(path):
    global WHISPER_MODEL
    if not WHISPER_AVAILABLE:
        return "", "Whisper is not installed. Run: pip install faster-whisper"
    try:
        if WHISPER_MODEL is None:
            print("Loading Whisper tiny model...")
            WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = WHISPER_MODEL.transcribe(path, beam_size=1, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        detected_language = getattr(info, "language", None)
        return text, detected_language or "unknown"
    except Exception as exc:
        print("Whisper transcription failed:", exc)
        return "", f"Transcription failed: {exc}"

# -----------------------------
# DB HELPERS
# -----------------------------

def serialize(doc):
    if doc is None:
        return None
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc["_id"])
    return doc


def to_object_id(raw_id):
    try:
        return ObjectId(raw_id)
    except (InvalidId, TypeError):
        return None


def normalize_doc(doc):
    out = dict(doc)
    out.setdefault("id", str(out.get("_id", "")))
    out.setdefault("filename", "Unknown")
    out.setdefault("transcript", "")
    out.setdefault("risk_score", 0)
    out.setdefault("risk_level", "Low Risk")
    out.setdefault("scam_probability", 0)
    out.setdefault("scam_detected", False)
    out.setdefault("scam_type", "Unknown")
    out.setdefault("scam_intent", "No strong scam intent detected")
    out.setdefault("social_engineering", "No strong manipulation tactic detected")
    out.setdefault("impersonation", "No clear impersonation detected")
    out.setdefault("manipulation_tactics", [])
    out.setdefault("evidence", [])
    out.setdefault("evidence_phrases", [])
    out.setdefault("suspicious_sentences", [])
    out.setdefault("recommendations", [])
    out.setdefault("language", "Unknown")
    out.setdefault("transcription_status", "Manual transcript")
    out.setdefault("created_at", "Unknown")
    return out


def get_all_analyses():
    if MONGO_CONNECTED:
        try:
            return [normalize_doc(serialize(d)) for d in analyses_col.find().sort("created_at_ts", DESCENDING)]
        except Exception as exc:
            print("Mongo read failed:", exc)
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    cols = ["id"] + list(SQLITE_COLUMNS.keys())
    cur.execute("SELECT " + ", ".join(cols) + " FROM analyses ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        for key in ["manipulation_tactics", "evidence", "evidence_phrases", "suspicious_sentences", "recommendations"]:
            raw = d.get(key)
            if isinstance(raw, str):
                import json
                try: d[key] = json.loads(raw)
                except Exception: d[key] = [x for x in raw.split("\n") if x]
        d["scam_detected"] = bool(d.get("scam_detected"))
        result.append(normalize_doc(d))
    return result


def get_analysis(report_id):
    if MONGO_CONNECTED:
        try:
            oid = to_object_id(report_id)
            if oid:
                doc = analyses_col.find_one({"_id": oid})
                if doc: return normalize_doc(serialize(doc))
        except Exception as exc:
            print("Mongo report read failed:", exc)
    try:
        number = int(report_id)
    except (ValueError, TypeError):
        return None
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    cols = ["id"] + list(SQLITE_COLUMNS.keys())
    cur.execute("SELECT " + ", ".join(cols) + " FROM analyses WHERE id = ?", (number,))
    row = cur.fetchone()
    conn.close()
    if not row: return None
    d = dict(zip(cols, row))
    import json
    for key in ["manipulation_tactics", "evidence", "evidence_phrases", "suspicious_sentences", "recommendations"]:
        raw = d.get(key)
        if isinstance(raw, str):
            try: d[key] = json.loads(raw)
            except Exception: d[key] = [x for x in raw.split("\n") if x]
    d["scam_detected"] = bool(d.get("scam_detected"))
    return normalize_doc(d)


def insert_analysis(document):
    if MONGO_CONNECTED:
        try:
            result = analyses_col.insert_one(document)
            return str(result.inserted_id)
        except Exception as exc:
            print("Mongo insert failed; using SQLite:", exc)
    import json
    conn = sqlite3.connect(SQLITE_DB)
    cur = conn.cursor()
    fields = list(SQLITE_COLUMNS.keys())
    values = []
    for field in fields:
        value = document.get(field)
        if field in {"manipulation_tactics", "evidence", "evidence_phrases", "suspicious_sentences", "recommendations"}:
            value = json.dumps(value, ensure_ascii=False)
        elif field == "created_at_ts" and hasattr(value, "isoformat"):
            value = value.isoformat()
        values.append(value)
    placeholders = ",".join("?" for _ in fields)
    cur.execute(f"INSERT INTO analyses ({','.join(fields)}) VALUES ({placeholders})", values)
    new_id = cur.lastrowid
    conn.commit(); conn.close()
    return str(new_id)


def delete_analysis(report_id):
    if MONGO_CONNECTED:
        try:
            oid = to_object_id(report_id)
            if oid:
                analyses_col.delete_one({"_id": oid}); return
        except Exception as exc:
            print("Mongo delete failed:", exc)
    try: number = int(report_id)
    except (ValueError, TypeError): return
    conn = sqlite3.connect(SQLITE_DB); cur = conn.cursor()
    cur.execute("DELETE FROM analyses WHERE id = ?", (number,)); conn.commit(); conn.close()


def get_counts():
    data = get_all_analyses()
    return len(data), sum(x["risk_level"] == "High Risk" for x in data), sum(x["risk_level"] == "Medium Risk" for x in data), sum(x["risk_level"] == "Low Risk" for x in data)

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    total, high, medium, low = get_counts()
    return render_template("index.html", total_calls=total, high_risk=high, medium_risk=medium, low_risk=low, reports_count=total)

@app.route("/analyze", methods=["GET"])
def analyze_page():
    return render_template("analyze.html")

@app.route("/history")
def history():
    return render_template("history.html", calls=get_all_analyses())

@app.route("/reports")
def reports():
    return render_template("reports.html", reports=get_all_analyses())

@app.route("/report/<report_id>")
def view_report(report_id):
    report = get_analysis(report_id)
    if report is None:
        return "Report not found", 404
    try:
        display_analysis = analyze_text(report.get("transcript", ""))
        report["highlighted_transcript"] = display_analysis["highlighted_transcript"]
        report["summary"] = display_analysis["summary"]
    except Exception:
        report["highlighted_transcript"] = Markup(str(escape(report.get("transcript", ""))).replace("\n", "<br>"))
        report["summary"] = "DeepShield analysis result."
    return render_template("result.html", result=report)

@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded_file = request.files.get("audio")
    manual_transcript = request.form.get("transcript", "").strip()
    filename = "Manual Text Analysis"
    transcription_status = "Manual transcript"

    if uploaded_file and uploaded_file.filename:
        filename = re.sub(r"[^a-zA-Z0-9._-]", "_", uploaded_file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(save_path)
        if not manual_transcript:
            transcript, status = transcribe_audio(save_path)
            if transcript:
                manual_transcript = transcript
                transcription_status = f"Whisper transcription ({status})"
            else:
                transcription_status = status

    if not manual_transcript:
        return "No transcript could be generated. Upload an audio file or enter a transcript.", 400

    analysis = analyze_text(manual_transcript)
    now = datetime.now()
    document = {
        **{k: v for k, v in analysis.items() if k not in {"highlighted_transcript", "summary", "transcript"}},
        "filename": filename,
        "transcript": manual_transcript,
        "transcription_status": transcription_status,
        "created_at": now.strftime("%d %b %Y, %I:%M %p"),
        "created_at_ts": now,
    }
    report_id = insert_analysis(document)
    return redirect(url_for("view_report", report_id=report_id))

@app.route("/report/<report_id>/pdf")
def download_pdf(report_id):
    if not PDF_AVAILABLE:
        return "PDF library is not installed. Run: pip install reportlab", 500
    report = get_analysis(report_id)
    if not report: return "Report not found", 404
    pdf_filename = f"DeepShield_Report_{report_id}.pdf"
    pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet(); styles["Title"].alignment = TA_CENTER
    def ml(value): return str(value or "").replace("\n", "<br/>")
    elements = [Paragraph("DEEPSHIELD", styles["Title"]), Paragraph("AI Voice Scam Analysis Report", styles["Heading2"]), Spacer(1, 20)]
    table = Table([
        ["File", str(report.get("filename"))], ["Date", str(report.get("created_at"))],
        ["Result", "SCAM DETECTED" if report.get("scam_detected") else "NO STRONG SCAM PATTERN"],
        ["Risk Level", str(report.get("risk_level"))], ["Risk Score", f'{report.get("risk_score", 0)} / 100'],
        ["Scam Probability", f'{report.get("scam_probability", 0)}%'],
    ], colWidths=[120, 350])
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (0,-1), colors.lightgrey), ("GRID", (0,0), (-1,-1), .5, colors.grey), ("VALIGN", (0,0), (-1,-1), "TOP"), ("FONTSIZE", (0,0), (-1,-1), 9)]))
    elements += [table, Spacer(1, 18), Paragraph("Transcript", styles["Heading2"]), Paragraph(ml(report.get("transcript")), styles["BodyText"]), Spacer(1, 15), Paragraph("Detected Evidence", styles["Heading2"]), Paragraph(ml("\n".join(report.get("evidence", []))), styles["BodyText"]), Spacer(1, 15), Paragraph("Manipulation Tactics", styles["Heading2"]), Paragraph(ml(", ".join(x.get("name", "") for x in report.get("manipulation_tactics", []))), styles["BodyText"]), Spacer(1, 15), Paragraph("Recommendations", styles["Heading2"]), Paragraph(ml("\n".join(report.get("recommendations", []))), styles["BodyText"])]
    doc.build(elements)
    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename, mimetype="application/pdf")

@app.route("/report/<report_id>/delete", methods=["POST"])
def delete_report(report_id):
    delete_analysis(report_id)
    return redirect(url_for("reports"))

@app.errorhandler(413)
def too_large(_):
    return "Audio file is too large. Maximum size is 50 MB.", 413

if __name__ == "__main__":
    print("DeepShield started")
    print("MongoDB:", "connected" if MONGO_CONNECTED else "SQLite fallback")
    print("Scam model:", "loaded" if scam_model is not None else "NOT loaded")
    print("Manipulation model:", "loaded" if manipulation_model is not None else "NOT loaded")
    print("Whisper:", "available" if WHISPER_AVAILABLE else "not installed")
    app.run(host="0.0.0.0", port=8000, debug=True)
