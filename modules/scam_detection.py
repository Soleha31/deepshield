import re
from difflib import SequenceMatcher


# ============================================================
# DEEPSHIELD SCAM PATTERNS
# ============================================================

SCAM_PATTERNS = {

    "OTP / Credential Request": [
        "otp",
        "one time password",
        "one-time password",
        "verification code",
        "verification number",
        "security code",
        "cvv",
        "upi pin",
        "upi pin number",
        "atm pin",
        "pin number",
        "password",
        "login password",
        "bank password"
    ],

    "KYC / Account Verification": [
        "kyc",
        "kyc pending",
        "verify your account",
        "verify account",
        "account verification",
        "account verify",
        "documents verify",
        "verification required",
        "complete your kyc",
        "kyc verification"
    ],

    "Money / Payment Request": [
        "send money",
        "transfer money",
        "make payment",
        "upi payment",
        "payment karo",
        "paise bhej",
        "paise bhejo",
        "money transfer",
        "send the money",
        "pay now",
        "payment karna",
        "account mein paise"
    ],

    "Urgency / Threat": [
        "urgent",
        "immediately",
        "right now",
        "act now",
        "account will be blocked",
        "account blocked",
        "last warning",
        "jaldi karo",
        "jaldi kijiye",
        "abhi karo",
        "abhi kijiye",
        "turant",
        "police case",
        "arrest",
        "legal action",
        "account band",
        "account bandh"
    ],

    "Remote Access / Suspicious Link": [
        "click this link",
        "click the link",
        "open this link",
        "install this app",
        "download this app",
        "screen share",
        "screen sharing",
        "remote access",
        "anydesk",
        "teamviewer",
        "send me the link",
        "link open karo"
    ]
}


# ============================================================
# SOCIAL ENGINEERING
# ============================================================

SOCIAL_ENGINEERING_PATTERNS = {

    "Urgency": [
        "urgent",
        "immediately",
        "right now",
        "act now",
        "jaldi karo",
        "abhi karo",
        "turant",
        "last warning"
    ],

    "Authority / Fear": [
        "account will be blocked",
        "account blocked",
        "police case",
        "police complaint",
        "arrest",
        "legal action",
        "government",
        "cyber crime",
        "income tax",
        "court notice"
    ],

    "Pressure / Threat": [
        "otherwise",
        "or else",
        "you will lose",
        "do it now",
        "don't tell anyone",
        "do not tell anyone",
        "warna",
        "nahi to",
        "nahin to"
    ]
}


# ============================================================
# IMPERSONATION
# ============================================================

IMPERSONATION_PATTERNS = [
    "bank officer",
    "bank employee",
    "bank security team",
    "bank manager",
    "customer care",
    "customer support",
    "rbi officer",
    "rbi",
    "police officer",
    "cyber crime officer",
    "government officer",
    "income tax officer",
    "sbi security team",
    "hdfc bank",
    "icici bank",
    "axis bank",
    "bank se bol raha",
    "bank se baat kar raha",
    "official department"
]


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.lower()

    # Common Whisper errors
    replacements = {
        "kvs": "kyc",
        "kvc": "kyc",
        "kyc pending hai": "kyc pending",
        "one time passcode": "one time password"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Keep Hindi unicode + English + numbers
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# FUZZY MATCH
# ============================================================

def fuzzy_match(text, pattern, threshold=0.88):

    if not text or not pattern:
        return False

    if pattern in text:
        return True

    words = text.split()
    pattern_words = pattern.split()

    if len(pattern_words) == 1:
        for word in words:
            if SequenceMatcher(None, word, pattern).ratio() >= threshold:
                return True

    window = len(pattern_words)

    for i in range(max(1, len(words) - window + 1)):

        chunk = " ".join(words[i:i + window])

        ratio = SequenceMatcher(
            None,
            chunk,
            pattern
        ).ratio()

        if ratio >= threshold:
            return True

    return False


# ============================================================
# SCAM INDICATOR DETECTION
# ============================================================

def detect_scam_indicators(text):

    text = normalize_text(text)

    detected = []

    for category, patterns in SCAM_PATTERNS.items():

        matches = []

        for pattern in patterns:

            if fuzzy_match(text, pattern):

                if pattern not in matches:
                    matches.append(pattern)

        if matches:

            detected.append({
                "category": category,
                "matches": matches
            })

    return detected


# ============================================================
# SOCIAL ENGINEERING DETECTION
# ============================================================

def detect_social_engineering(text):

    text = normalize_text(text)

    detected = []

    for tactic, patterns in SOCIAL_ENGINEERING_PATTERNS.items():

        matches = []

        for pattern in patterns:

            if fuzzy_match(text, pattern):

                if pattern not in matches:
                    matches.append(pattern)

        if matches:

            detected.append({
                "tactic": tactic,
                "matches": matches
            })

    return detected


# ============================================================
# IMPERSONATION DETECTION
# ============================================================

def detect_impersonation(text):

    text = normalize_text(text)

    matches = []

    for pattern in IMPERSONATION_PATTERNS:

        if fuzzy_match(text, pattern):

            if pattern not in matches:
                matches.append(pattern)

    return {
        "detected": len(matches) > 0,
        "matches": matches
    }


# ============================================================
# CONVERSATION CONTEXT
# ============================================================

def analyze_context(text):

    text = normalize_text(text)

    contexts = []

    has_credential = any(
        fuzzy_match(text, p)
        for p in SCAM_PATTERNS["OTP / Credential Request"]
    )

    has_kyc = any(
        fuzzy_match(text, p)
        for p in SCAM_PATTERNS["KYC / Account Verification"]
    )

    has_money = any(
        fuzzy_match(text, p)
        for p in SCAM_PATTERNS["Money / Payment Request"]
    )

    has_urgency = any(
        fuzzy_match(text, p)
        for p in SCAM_PATTERNS["Urgency / Threat"]
    )

    has_remote = any(
        fuzzy_match(text, p)
        for p in SCAM_PATTERNS["Remote Access / Suspicious Link"]
    )


    if has_credential:

        contexts.append(
            "Caller appears to be requesting sensitive credentials."
        )


    if has_kyc:

        contexts.append(
            "Conversation involves account verification or account status."
        )


    if has_money:

        contexts.append(
            "Conversation involves a financial or payment request."
        )


    if has_urgency:

        contexts.append(
            "Caller is creating urgency or pressure."
        )


    if has_remote:

        contexts.append(
            "Caller may be attempting to obtain remote device access."
        )


    if has_kyc and has_credential:

        contexts.append(
            "KYC/account verification is combined with a sensitive credential request."
        )


    if has_kyc and has_urgency:

        contexts.append(
            "Account verification is combined with pressure or threat."
        )


    if has_money and has_urgency:

        contexts.append(
            "Financial request is combined with urgency or pressure."
        )


    return contexts


# ============================================================
# SCAM TYPE CLASSIFICATION
# ============================================================

def classify_scam(scam_indicators):

    if not scam_indicators:

        return "No Strong Scam Indicator"


    categories = [
        item["category"]
        for item in scam_indicators
    ]


    if "OTP / Credential Request" in categories:

        if "KYC / Account Verification" in categories:

            return "KYC / Credential Scam"

        return "Credential / OTP Scam"


    if "Money / Payment Request" in categories:

        return "Payment / Money Scam"


    if "Remote Access / Suspicious Link" in categories:

        return "Remote Access / Link Scam"


    if "Urgency / Threat" in categories:

        return "Urgency / Threat Scam"


    if "KYC / Account Verification" in categories:

        return "KYC / Account Verification Scam"


    return "Potential Scam"


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk(
    scam_indicators,
    social_engineering,
    impersonation,
    context
):

    score = 0

    evidence = []


    # --------------------------------------------------------
    # Scam indicators
    # --------------------------------------------------------

    category_points = {

        "OTP / Credential Request": 35,

        "KYC / Account Verification": 15,

        "Money / Payment Request": 30,

        "Urgency / Threat": 20,

        "Remote Access / Suspicious Link": 30
    }


    for item in scam_indicators:

        points = category_points.get(
            item["category"],
            10
        )

        score += points

        evidence.append({

            "type": "Scam Indicator",

            "category": item["category"],

            "matches": item["matches"],

            "points": points
        })


    # --------------------------------------------------------
    # Social engineering
    # --------------------------------------------------------

    for item in social_engineering:

        points = 8

        score += points

        evidence.append({

            "type": "Social Engineering",

            "category": item["tactic"],

            "matches": item["matches"],

            "points": points
        })


    # --------------------------------------------------------
    # Impersonation
    # --------------------------------------------------------

    if impersonation.get("detected"):

        points = 20

        score += points

        evidence.append({

            "type": "Impersonation",

            "category": "Identity Impersonation",

            "matches": impersonation["matches"],

            "points": points
        })


    # --------------------------------------------------------
    # Context combinations
    # --------------------------------------------------------

    context_text = " ".join(context).lower()


    if (
        "sensitive credential request" in context_text
        and "account verification" in context_text
    ):

        score += 15

        evidence.append({

            "type": "Conversation Context",

            "category": "KYC + Credential Request",

            "matches": [
                "KYC verification combined with sensitive credential request"
            ],

            "points": 15
        })


    if (
        "account verification" in context_text
        and "pressure" in context_text
    ):

        score += 15

        evidence.append({

            "type": "Conversation Context",

            "category": "KYC + Urgency",

            "matches": [
                "Account verification combined with pressure or threat"
            ],

            "points": 15
        })


    if (
        "financial or payment request" in context_text
        and "pressure" in context_text
    ):

        score += 15

        evidence.append({

            "type": "Conversation Context",

            "category": "Payment + Urgency",

            "matches": [
                "Financial request combined with pressure"
            ],

            "points": 15
        })


    # --------------------------------------------------------
    # Multiple suspicious behaviours
    # --------------------------------------------------------

    if len(scam_indicators) >= 3:

        score += 10

        evidence.append({

            "type": "Conversation Context",

            "category": "Multiple Suspicious Behaviours",

            "matches": [],

            "points": 10
        })


    # Maximum 100
    score = min(score, 100)


    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if score >= 70:

        risk_level = "HIGH"

    elif score >= 40:

        risk_level = "MEDIUM"

    else:

        risk_level = "SAFE"


    return score, risk_level, evidence


# ============================================================
# RISK TIMELINE
# ============================================================

def create_timeline(text):

    text = text.strip()

    if not text:

        return []


    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )


    timeline = []

    current_score = 0


    for index, sentence in enumerate(sentences, start=1):

        indicators = detect_scam_indicators(sentence)

        social = detect_social_engineering(sentence)

        impersonation = detect_impersonation(sentence)


        points = 0

        reasons = []


        for item in indicators:

            if item["category"] == "OTP / Credential Request":

                points += 35

            elif item["category"] == "Money / Payment Request":

                points += 30

            elif item["category"] == "Remote Access / Suspicious Link":

                points += 30

            elif item["category"] == "Urgency / Threat":

                points += 20

            elif item["category"] == "KYC / Account Verification":

                points += 15


            reasons.append(item["category"])


        points += len(social) * 8


        if impersonation["detected"]:

            points += 20

            reasons.append("Impersonation")


        current_score = min(
            current_score + points,
            100
        )


        if points > 0:

            timeline.append({

                "sentence_number": index,

                "sentence": sentence,

                "points": points,

                "score": current_score,

                "reasons": reasons

            })


    return timeline


# ============================================================
# SAFETY RECOMMENDATIONS
# ============================================================

def get_recommendations(
    risk_level,
    scam_indicators
):

    recommendations = []


    categories = [
        item["category"]
        for item in scam_indicators
    ]


    if "OTP / Credential Request" in categories:

        recommendations.append(
            "Never share OTP, UPI PIN, CVV or banking passwords over a call."
        )


    if "Money / Payment Request" in categories:

        recommendations.append(
            "Do not transfer money based only on instructions received during an unexpected call."
        )


    if "KYC / Account Verification" in categories:

        recommendations.append(
            "Verify KYC or account issues through the bank's official app or website."
        )


    if "Urgency / Threat" in categories:

        recommendations.append(
            "Do not act under pressure. End the call and independently verify the claim."
        )


    if "Remote Access / Suspicious Link" in categories:

        recommendations.append(
            "Do not install remote-access applications or open suspicious links."
        )


    if risk_level == "HIGH":

        recommendations.append(
            "If money or credentials were already shared, contact your bank immediately."
        )


    if not recommendations:

        recommendations.append(
            "No strong scam indicators were detected. Still avoid sharing sensitive information."
        )


    return recommendations