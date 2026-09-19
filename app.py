import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Optional GenAI import
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="RoSense | Enterprise Dual-Engine NLP", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 14px 18px;
        border-radius: 10px;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.82rem !important;
        color: #9ca3af !important;
        font-weight: 600;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #f9fafb !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ RoSense: Enterprise Code-Mixed NLP Triage Engine")
st.caption("Dual-Engine Architecture: Classical Subword N-Gram ML + Cloud Foundation LLM Fallback")

# ==========================================
# 1. CLASSICAL ML ENGINE (EDGE / SUBWORD TF-IDF)
# ==========================================
HINGLISH_VOCAB = {
    "mera", "meri", "mere", "mujhe", "mai", "main", "hum", "tu", "tera", "aap", "aapka", 
    "ka", "ki", "ke", "ko", "se", "me", "mein", "pe", "par", "hai", "hain", "ho", "tha", 
    "thi", "the", "kar", "karo", "karna", "kiya", "raha", "rahi", "gaya", "gya", "jaunga", 
    "kya", "kyu", "kyun", "kab", "kaise", "kitna", "bohot", "bahut", "jaldi", "turant", 
    "nahi", "nhi", "paise", "paisa", "rupaye", "ghante", "bhai", "badh", "bada", "du", "dun"
}

INDIC_SUFFIX_PATTERN = r"(unga|ungi|enge|ega|egi|oge|kar|wala|wali|wale|ta|ti|te|raha|rahi)$"

def compute_indic_density(text):
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not tokens:
        return 0.0
    indic_count = sum(1 for tok in tokens if tok in HINGLISH_VOCAB or re.search(INDIC_SUFFIX_PATTERN, tok))
    return round((indic_count / len(tokens)) * 100, 1)

@st.cache_resource
def get_classical_pipeline():
    training_data = [
        ("recharge ka price badh gaya itna jyada ab kyu extra paise du", "Billing & Tariffs"),
        ("balance deduct ho gaya bina kisi reason ke refund chahiye", "Billing & Tariffs"),
        ("mera 499 ka recharge fail ho gaya bank se paise kat gaye", "Billing & Tariffs"),
        ("monthly bill me hidden charges add huye hain explain karo", "Billing & Tariffs"),
        ("pehle plan 455 ka tha ab 555 le rahe ho itna tariff hike kyu", "Billing & Tariffs"),
        ("auto debit ho gaya without OTP payment refund initiate karo", "Billing & Tariffs"),
        
        ("red light router pe blink kar rahi hai subah se internet down", "Network Ops"),
        ("call drop ho rahi hai continuous internet slow chal raha hai", "Network Ops"),
        ("fiber connection dead hai WFH meeting miss ho rahi hai", "Network Ops"),
        ("tower me network coverage zero hai 5G bilkul nahi chal raha", "Network Ops"),
        ("packet loss high hai ping bohot kharab aa raha hai gaming me", "Network Ops"),
        
        ("mujhe e-sim me switch karna hai physical sim se kya process hai", "SIM & Account"),
        ("sim block ho gaya puk code chahiye handset unlock karne", "SIM & Account"),
        ("naya sim card order kiya tha delivery abhi tak nahi aayi", "SIM & Account"),
        ("kyc verify karwana hai documents upload nahi ho rahe app pe", "SIM & Account"),
        
        ("consumer court me jaunga complaint file karne fraud service ke khilaf", "Legal & Churn Escalation"),
        ("i am utterly disappointed will complain to the government consumer forum", "Legal & Churn Escalation"),
        ("trai me appeal karunga tumhara license cancel hona chahiye", "Legal & Churn Escalation"),
        ("agar 1 ghante me resolve nahi hua toh jio airtel me port karwa lunga", "Legal & Churn Escalation"),
        ("legal notice bhej raha hu fraud deductions ke against lawyer se", "Legal & Churn Escalation")
    ]
    df = pd.DataFrame(training_data, columns=["text", "label"])
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)),
        ('clf', LogisticRegression(C=2.5, class_weight='balanced', max_iter=300))
    ])
    pipe.fit(df["text"], df["label"])
    return pipe

classical_pipe = get_classical_pipeline()

REGULATORY_SIGNALS = [
    (r"\b(consumer court|consumer forum|court)\b", 45, "Legal / Consumer Forum"),
    (r"\b(trai|government|nodal|ombudsman)\b", 40, "Regulatory Escalation"),
    (r"\b(legal notice|lawyer|police|fir|fraud)\b", 35, "Legal / Fraud Allegation"),
    (r"\b(port|porting|switch|chhod dunga|jio|airtel)\b", 25, "Customer Churn Risk"),
    (r"\b(utterly disappointed|mental harassment|harass|ghatiya)\b", 20, "Severe Dissatisfaction"),
    (r"\b(wfh|office|hospital|emergency)\b", 15, "Mission Critical Outage")
]

def run_classical_inference(text):
    classes = classical_pipe.named_steps['clf'].classes_
    probs = classical_pipe.predict_proba([text])[0]
    prob_dict = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probs)}
    top_intent = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[top_intent]
    
    risk_score = 0
    detected_signals = []
    for pattern, weight, label in REGULATORY_SIGNALS:
        if re.search(pattern, text, re.IGNORECASE):
            risk_score += weight
            detected_signals.append(label)
    
    if "Legal & Churn Escalation" in classes:
        idx = list(classes).index("Legal & Churn Escalation")
        risk_score += probs[idx] * 40
        
    risk_score = min(int(risk_score), 100)
    density = compute_indic_density(text)
    
    if risk_score >= 65 or top_intent == "Legal & Churn Escalation":
        dept = "L3 Executive Regulatory Desk"
    elif top_intent == "Network Ops":
        dept = "NOC Automated Diagnostics"
    elif top_intent == "Billing & Tariffs":
        dept = "Tier-2 Revenue Management"
    else:
        dept = "Digital Self-Care Bot"

    return {
        "engine": "Classical Subword ML (Edge Inference)",
        "indic_density": density,
        "intent": top_intent,
        "confidence": confidence,
        "prob_distribution": prob_dict,
        "risk_score": risk_score,
        "signals": list(set(detected_signals)),
        "routed_department": dept,
        "action": "Automated priority routing executed via Edge heuristic scoring."
    }

# ==========================================
# 2. FOUNDATION LLM ENGINE (GEMINI API)
# ==========================================
def run_llm_inference(text, key):
    if not GENAI_AVAILABLE:
        raise Exception("google-genai library not installed. Run: pip install google-genai")
    client = genai.Client(api_key=key)
    prompt = f"""
    You are RoSense, an enterprise telecom customer triage engine for Tech Mahindra.
    Analyze this message containing conversational Hinglish or English:
    "{text}"

    Return ONLY a JSON object with this schema:
    {{
        "indic_density_pct": <int 0-100>,
        "dominant_intent": <"Billing & Tariffs" | "Network Ops" | "SIM & Account" | "Legal & Churn Escalation">,
        "confidence": <int 80-100>,
        "escalation_risk_score": <int 0-100>,
        "risk_signals": [<strings of detected threats, e.g. "Consumer Court Mention">],
        "routed_department": <"L3 Executive Regulatory Desk" | "NOC Automated Diagnostics" | "Tier-2 Revenue Management" | "Digital Self-Care Bot">,
        "recommended_action": <one sentence enterprise action plan>
    }}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
    )
    res = json.loads(response.text)
    return {
        "engine": "Foundation LLM (Gemini 2.5 Flash)",
        "indic_density": res["indic_density_pct"],
        "intent": res["dominant_intent"],
        "confidence": res.get("confidence", 95),
        "prob_distribution": {res["dominant_intent"]: res.get("confidence", 95)},
        "risk_score": res["escalation_risk_score"],
        "signals": res["risk_signals"],
        "routed_department": res["routed_department"],
        "action": res["recommended_action"]
    }

# ==========================================
# 3. STREAMLIT FRONTEND & PIPELINE SELECTION
# ==========================================
st.sidebar.header("Pipeline Architecture")
engine_choice = st.sidebar.selectbox(
    "Active Inference Engine",
    [
        "⚡ Classical ML (Subword Character N-Grams / Edge)",
        "🧠 Foundation LLM (Gemini 2.5 Flash / Zero-Shot)",
        "🔄 Hybrid Cascading Pipeline (Edge First -> LLM Fallback)"
    ]
)

gemini_key = st.sidebar.text_input(
    "Gemini API Key (Required for LLM modes)", 
    type="password", 
    help="Get a free key from aistudio.google.com"
)
sla_threshold = st.sidebar.slider("SLA Escalation Alert Threshold", 30, 90, 60)

# Preset telemetry
PRESET_OPTIONS = {
    "Price Hike & Consumer Court Threat": "recharge ka price is baar itna badh kyu gya, pehle to 455 me ho jata tha ab kyu 555 du mai. i am utterly disappointed. i will complain to the government about this. consumer court me jaunga",
    "Broadband WFH Outage": "Red light router pe blink kar rahi hai subah 9 baje se. Internet completely dead hai, office WFH urgent meetings miss ho gayi. Solve it immediately.",
    "Routine eSIM Migration": "Mujhe naya e-SIM chahiye existing physical SIM se switch karne ke liye. Kya store aana hoga verify karne ke liye?"
}

preset_choice = st.sidebar.selectbox("Load Sample Support Interaction", list(PRESET_OPTIONS.keys()))
user_text = st.text_area("Live Hinglish Customer Telemetry Stream", value=PRESET_OPTIONS[preset_choice], height=110)

if st.button("Execute RoSense Triage Pipeline", type="primary"):
    result = None
    
    # Engine Dispatch Logic
    if "Classical ML" in engine_choice:
        result = run_classical_inference(user_text)
        
    elif "Foundation LLM" in engine_choice:
        if not gemini_key:
            st.error("⚠️ Please enter a Gemini API Key in the left sidebar to use the LLM engine.")
        else:
            with st.spinner("Invoking Gemini 2.5 Flash for deep vernacular reasoning..."):
                try:
                    result = run_llm_inference(user_text, gemini_key)
                except Exception as e:
                    st.error(f"LLM Error: {e}")
                    
    elif "Hybrid Cascading" in engine_choice:
        with st.spinner("Running Tier-1 Edge ML inspection..."):
            local_res = run_classical_inference(user_text)
            
            # Cascading logic: If high risk or low confidence, escalate to LLM
            if (local_res["risk_score"] >= sla_threshold or local_res["confidence"] < 60) and gemini_key:
                st.info("🔄 Hybrid Cascade: High regulatory risk detected by Tier-1. Escalating to Foundation LLM for deep contextual verification...")
                try:
                    result = run_llm_inference(user_text, gemini_key)
                    result["engine"] = "Hybrid Cascade (Escalated to LLM)"
                except Exception:
                    result = local_res
                    result["engine"] = "Hybrid Cascade (Tier-1 Fallback)"
            else:
                result = local_res
                result["engine"] = "Hybrid Cascade (Resolved at Tier-1 Edge)"

    # Render Results
    if result:
        st.caption(f"**Executing Pipeline:** `{result['engine']}`")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Indic Token Density", f"{result['indic_density']}%")
        with c2:
            st.metric("Dominant Intent", result['intent'])
        with c3:
            risk = result['risk_score']
            st.metric("Escalation Risk", f"{risk} / 100",
                      delta="CRITICAL BREACH" if risk >= sla_threshold else "NORMAL SLA",
                      delta_color="inverse" if risk >= sla_threshold else "normal")
        with c4:
            st.metric("Routed Queue", result['routed_department'])

        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            st.subheader("Intent Confidence Breakdown")
            if len(result["prob_distribution"]) > 1:
                df_chart = pd.DataFrame(list(result["prob_distribution"].items()), columns=["Category", "Score"]).sort_values("Score", ascending=True)
                fig = px.bar(df_chart, x="Score", y="Category", orientation='h', color="Score", color_continuous_scale="Viridis")
                fig.update_layout(plot_bgcolor='#111827', paper_bgcolor='#111827', font_color='#e5e7eb', height=280)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Model Confidence: **{result['confidence']}%** on `{result['intent']}`")
                
            st.write("**Extracted Threat Signals:**")
            if result["signals"]:
                for s in result["signals"]:
                    st.markdown(f"- `{s}`")
            else:
                st.write("_No high-risk regulatory or legal triggers detected._")

        with col_r:
            st.subheader("Automated Operational Action")
            st.info(f"**Action Directive:**\n\n{result['action']}")
            if risk >= sla_threshold:
                st.error(f"🚨 Priority Queue Override Triggered ({risk}/100)")
                st.markdown("""
                * **SLA Window:** Shortened from 48h to **30 minutes**.
                * **Routing:** Direct SIP escalation to Senior Customer Relations Liaison.
                * **Compliance:** Pre-emptive flag logged in TRAI compliance audit log.
                """)
            else:
                st.success("✅ Standard Operating Envelope")
                st.markdown("""
                * **SLA Window:** 24-hour turnaround.
                * **Routing:** Retained in standard automated self-care tier.
                """)