"""
RoSense: Code-Mixed Telecommunications Triage & SLA Routing Console.
Copyright (c) 2026. Enterprise Support Systems Integration.
"""

from typing import Dict, Any, List, Tuple
import json
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from google import genai
    from google.genai import types
    GENAI_SUPPORTED = True
except ImportError:
    GENAI_SUPPORTED = False

# Page Configuration
st.set_page_config(
    page_title="RoSense Console | Telecom Operations",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Slate & Monochrome Styling (No Cartoons / No Emojis)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: #f0f6fc !important;
    }
    .status-badge-critical {
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.4);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .status-badge-nominal {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .panel-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.title("RoSense: Code-Mixed NLP Triage Console")
st.caption("Operations Dashboard / Dialectal Text Ingestion & SLA Escalation Router")

# Vocabulary & Morphology Constants
HINGLISH_VOCAB = {
    "mera", "meri", "mere", "mujhe", "mai", "main", "hum", "tu", "tera", "aap", "aapka",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "pe", "par", "hai", "hain", "ho", "tha",
    "thi", "the", "kar", "karo", "karna", "kiya", "raha", "rahi", "gaya", "gya", "jaunga",
    "kya", "kyu", "kyun", "kab", "kaise", "kitna", "bohot", "bahut", "jaldi", "turant",
    "nahi", "nhi", "paise", "paisa", "rupaye", "ghante", "bhai", "badh", "bada", "du", "dun"
}
INDIC_SUFFIX_REGEX = re.compile(r"(unga|ungi|enge|ega|egi|oge|kar|wala|wali|wale|ta|ti|te|raha|rahi)$")

REGULATORY_INDICATORS: List[Tuple[str, int, str]] = [
    (r"\b(consumer court|consumer forum|court)\b", 45, "Legal / Consumer Forum"),
    (r"\b(trai|government|nodal|ombudsman)\b", 40, "Regulatory Escalation (TRAI)"),
    (r"\b(legal notice|lawyer|police|fir|fraud)\b", 35, "Legal / Fraud Allegation"),
    (r"\b(port|porting|switch|chhod dunga|jio|airtel)\b", 25, "Churn Risk"),
    (r"\b(utterly disappointed|mental harassment|harass|ghatiya)\b", 20, "Severe Dissatisfaction"),
    (r"\b(wfh|office|hospital|emergency)\b", 15, "Critical Outage Impact")
]


def calculate_indic_density(text: str) -> float:
    """Calculates proportion of Indic/Hinglish morphological markers in text."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not tokens:
        return 0.0
    matched = sum(1 for t in tokens if t in HINGLISH_VOCAB or INDIC_SUFFIX_REGEX.search(t))
    return round((matched / len(tokens)) * 100, 1)


@st.cache_resource
def build_edge_classifier() -> Pipeline:
    """Trains a subword character n-gram pipeline for local edge inference."""
    corpus = [
        ("recharge ka price badh gaya itna jyada ab kyu extra paise du", "Billing & Tariffs"),
        ("balance deduct ho gaya bina kisi reason ke refund chahiye", "Billing & Tariffs"),
        ("mera 499 ka recharge fail ho gaya bank se paise kat gaye", "Billing & Tariffs"),
        ("monthly bill me hidden charges add huye hain explain karo", "Billing & Tariffs"),
        ("pehle plan 455 ka tha ab 555 le rahe ho itna tariff hike kyu", "Billing & Tariffs"),
        ("auto debit ho gaya without OTP payment refund initiate karo", "Billing & Tariffs"),
        ("red light router pe blink kar rahi hai subah se internet down", "Network Operations"),
        ("call drop ho rahi hai continuous internet slow chal raha hai", "Network Operations"),
        ("fiber connection dead hai WFH meeting miss ho rahi hai", "Network Operations"),
        ("tower me network coverage zero hai 5G bilkul nahi chal raha", "Network Operations"),
        ("packet loss high hai ping bohot kharab aa raha hai gaming me", "Network Operations"),
        ("mujhe e-sim me switch karna hai physical sim se kya process hai", "SIM Services"),
        ("sim block ho gaya puk code chahiye handset unlock karne", "SIM Services"),
        ("naya sim card order kiya tha delivery abhi tak nahi aayi", "SIM Services"),
        ("kyc verify karwana hai documents upload nahi ho rahe app pe", "SIM Services"),
        ("consumer court me jaunga complaint file karne fraud service ke khilaf", "Regulatory & Churn"),
        ("i am utterly disappointed will complain to the government consumer forum", "Regulatory & Churn"),
        ("trai me appeal karunga tumhara license cancel hona chahiye", "Regulatory & Churn"),
        ("agar 1 ghante me resolve nahi hua toh jio airtel me port karwa lunga", "Regulatory & Churn"),
        ("legal notice bhej raha hu fraud deductions ke against lawyer se", "Regulatory & Churn")
    ]
    df = pd.DataFrame(corpus, columns=["text", "label"])
    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)),
        ('classifier', LogisticRegression(C=2.5, class_weight='balanced', max_iter=300))
    ])
    pipe.fit(df["text"], df["label"])
    return pipe

classifier_model = build_edge_classifier()


def execute_edge_inference(text: str) -> Dict[str, Any]:
    """Runs local Subword Character N-Gram classification."""
    classes = classifier_model.named_steps['classifier'].classes_
    probabilities = classifier_model.predict_proba([text])[0]
    prob_map = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probabilities)}
    primary_intent = max(prob_map, key=prob_map.get)
    confidence = prob_map[primary_intent]

    threat_score = 0
    detected_triggers = []
    for pattern, weight, label in REGULATORY_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            threat_score += weight
            detected_triggers.append(label)

    if "Regulatory & Churn" in classes:
        idx = list(classes).index("Regulatory & Churn")
        threat_score += probabilities[idx] * 40

    threat_score = min(int(threat_score), 100)
    density = calculate_indic_density(text)

    if threat_score >= 65 or primary_intent == "Regulatory & Churn":
        queue = "L3 Executive Regulatory Desk"
    elif primary_intent == "Network Operations":
        queue = "NOC Automated Diagnostics"
    elif primary_intent == "Billing & Tariffs":
        queue = "Tier-2 Revenue Assurance"
    else:
        queue = "Tier-1 Digital Ingestion"

    return {
        "engine": "Tier-1 Edge Pipeline (Subword TF-IDF)",
        "indic_density": density,
        "intent": primary_intent,
        "confidence": confidence,
        "distributions": prob_map,
        "risk_score": threat_score,
        "triggers": list(set(detected_triggers)),
        "destination_queue": queue,
        "directive": "Automated routing executed based on edge heuristic evaluation."
    }


def execute_cloud_inference(text: str, api_token: str) -> Dict[str, Any]:
    """Invokes Foundation LLM for zero-shot semantic parsing with strict JSON schema."""
    if not GENAI_SUPPORTED:
        raise RuntimeError("Missing google-genai dependency.")

    client = genai.Client(api_key=api_token)
    prompt = f"""
    Analyze the following telecommunications support telemetry:
    "{text}"

    Output strict JSON adhering to this schema:
    {{
        "indic_density": <int 0-100 representing code-mixed percentage>,
        "primary_intent": <"Billing & Tariffs" | "Network Operations" | "SIM Services" | "Regulatory & Churn">,
        "confidence": <int 0-100>,
        "risk_score": <int 0-100 representing urgency or regulatory escalation danger>,
        "identified_triggers": [<string list of threat indicators>],
        "destination_queue": <"L3 Executive Regulatory Desk" | "NOC Automated Diagnostics" | "Tier-2 Revenue Assurance" | "Tier-1 Digital Ingestion">,
        "directive": <one sentence actionable instruction for operations staff>
    }}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
    )
    payload = json.loads(response.text)
    return {
        "engine": "Tier-2 Foundation LLM (Gemini 2.5 Flash)",
        "indic_density": payload.get("indic_density", 0),
        "intent": payload.get("primary_intent", "Unknown"),
        "confidence": payload.get("confidence", 95),
        "distributions": {payload.get("primary_intent", "Unknown"): payload.get("confidence", 95)},
        "risk_score": payload.get("risk_score", 0),
        "triggers": payload.get("identified_triggers", []),
        "destination_queue": payload.get("destination_queue", "Tier-1 Digital Ingestion"),
        "directive": payload.get("directive", "Review customer history prior to dispatch.")
    }


# Sidebar Controls
st.sidebar.subheader("System Configuration")
pipeline_mode = st.sidebar.selectbox(
    "Routing Pipeline",
    [
        "Tier-1 Edge Classifier (Subword TF-IDF)",
        "Tier-2 Cloud Foundation Model (Zero-Shot)",
        "Cascading Hybrid Architecture (Edge -> Cloud Escalation)"
    ]
)

default_secret_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
gemini_key_input = st.sidebar.text_input(
    "API Key (Required for Tier-2 Modes)",
    value=default_secret_key,
    type="password"
)

sla_limit = st.sidebar.slider("SLA Escalation Threshold", min_value=30, max_value=90, value=65)

# Telemetry Presets
PRESET_RECORDS = {
    "Tariff Increase & Regulatory Complaint": "recharge ka price is baar itna badh kyu gya, pehle to 455 me ho jata tha ab kyu 555 du mai. i am utterly disappointed. i will complain to the government about this. consumer court me jaunga",
    "Fiber Outage (WFH Impact)": "Red light router pe blink kar rahi hai subah 9 baje se. Internet completely dead hai, office WFH urgent meetings miss ho gayi. Solve it immediately.",
    "eSIM Migration Request": "Mujhe naya e-SIM chahiye existing physical SIM se switch karne ke liye. Kya store aana hoga verify karne ke liye?"
}

selected_preset = st.sidebar.selectbox("Test Telemetry Stream", list(PRESET_RECORDS.keys()))
input_stream = st.text_area("Live Ingested Telemetry Feed", value=PRESET_RECORDS[selected_preset], height=110)

if st.button("Ingest and Evaluate Telemetry", type="primary"):
    analysis_result = None

    if "Tier-1 Edge" in pipeline_mode:
        analysis_result = execute_edge_inference(input_stream)
    elif "Tier-2 Cloud" in pipeline_mode:
        if not gemini_key_input:
            st.error("Authentication required: Provide an API key in the configuration panel.")
        else:
            with st.spinner("Dispatching payload to Tier-2 Foundation Engine..."):
                try:
                    analysis_result = execute_cloud_inference(input_stream, gemini_key_input)
                except Exception as exc:
                    st.error(f"Inference error: {exc}")
    elif "Cascading Hybrid" in pipeline_mode:
        with st.spinner("Processing through Tier-1 Edge Engine..."):
            edge_res = execute_edge_inference(input_stream)
            if (edge_res["risk_score"] >= sla_limit or edge_res["confidence"] < 60) and gemini_key_input:
                st.info("Cascading Trigger: Telemetry exceeded confidence or threat bounds. Escalating to Tier-2 Foundation LLM...")
                try:
                    analysis_result = execute_cloud_inference(input_stream, gemini_key_input)
                    analysis_result["engine"] = "Cascading Pipeline (Escalated to Tier-2 Cloud)"
                except Exception:
                    analysis_result = edge_res
                    analysis_result["engine"] = "Cascading Pipeline (Tier-1 Fallback)"
            else:
                analysis_result = edge_res
                analysis_result["engine"] = "Cascading Pipeline (Resolved at Tier-1 Edge)"

    if analysis_result:
        st.markdown(f"**Execution Route:** `{analysis_result['engine']}`")

        # Top-level Metric Tiles
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Indic Language Ratio", f"{analysis_result['indic_density']}%")
        with c2:
            st.metric("Primary Classification", analysis_result['intent'])
        with c3:
            r_score = analysis_result['risk_score']
            st.metric(
                "SLA Risk Factor",
                f"{r_score} / 100",
                delta="BREACH ALERT" if r_score >= sla_limit else "NOMINAL",
                delta_color="inverse" if r_score >= sla_limit else "normal"
            )
        with c4:
            st.metric("Assigned Route", analysis_result['destination_queue'])

        # Detailed Layout
        col_left, col_right = st.columns([1.3, 1])

        with col_left:
            st.markdown("#### Probability Distribution")
            if len(analysis_result["distributions"]) > 1:
                df_chart = pd.DataFrame(
                    list(analysis_result["distributions"].items()),
                    columns=["Class", "Confidence"]
                ).sort_values("Confidence", ascending=True)
                fig = px.bar(
                    df_chart,
                    x="Confidence",
                    y="Class",
                    orientation='h',
                    color="Confidence",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(
                    plot_bgcolor='#161b22',
                    paper_bgcolor='#161b22',
                    font_color='#e6edf3',
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=240
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"Class Confidence: **{analysis_result['confidence']}%** on `{analysis_result['intent']}`")

            st.markdown("#### Extracted Threat Indicators")
            if analysis_result["triggers"]:
                for trigger in analysis_result["triggers"]:
                    st.markdown(f"- `{trigger}`")
            else:
                st.write("No critical regulatory or legal indicators detected.")

        with col_right:
            st.markdown("#### Operational Directive")
            st.info(analysis_result['directive'])

            st.markdown("#### Compliance & Queue Status")
            if r_score >= sla_limit:
                st.markdown('<span class="status-badge-critical">CRITICAL SLA WINDOW</span>', unsafe_allow_html=True)
                st.markdown("""
                - **Resolution Window:** Truncated to 30 minutes.
                - **Queue Override:** Direct SIP dispatch to Senior Regulatory Liaison.
                - **Audit Trail:** Ingested into compliance review log.
                """)
            else:
                st.markdown('<span class="status-badge-nominal">STANDARD SLA WINDOW</span>', unsafe_allow_html=True)
                st.markdown("""
                - **Resolution Window:** Standard 24-hour SLA.
                - **Routing:** Enqueued to digital self-care conversational system.
                """)