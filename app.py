import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(page_title="RoSense | Code-Mixed Indic NLP Engine", page_icon="⚡", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
    .stTextArea textarea { font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ RoSense: Code-Mixed Indic NLP & Support Triage Engine")
st.caption("Subword Vernacular Tokenizer, Intent Classifier & SLA Escalation Predictor for Enterprise Support")

# Sample Hinglish Presets for Live Testing
PRESETS = {
    "Payment Failure & Churn Threat": "Bhai mera 499 ka recharge fail ho gaya, bank account se paise cut gaye par plan active nahi hua. Call drop ho rahi hai baar baar. Agar 1 ghante me fix nahi kiya toh Jio me port karwa lunga!",
    "Broadband Fiber Blackout": "Fiber router me red light blink kar rahi hai subah 9 baje se. Internet completely down hai, WFH chal raha hai mera meeting miss ho gayi. Please engineer bhejo urgently.",
    "SIM Porting & KYC Query": "Mujhe naya e-SIM chahiye existing physical SIM card se convert karne ke liye. Kya documents submit karne honge store pe?",
    "Routine Billing Clarification": "Pichle mahine ka invoice breakdown samajh nahi aa raha, roaming charges extra kyun add huye hain bill me?"
}

st.sidebar.header("Engine Settings")
selected_demo = st.sidebar.selectbox("Load Sample Support Interaction", list(PRESETS.keys()))
urgency_threshold = st.sidebar.slider("SLA Breach Alert Threshold", 30, 90, 65)

user_input = st.text_area("Live Customer Chat / Voice-to-Text Stream", value=PRESETS[selected_demo], height=110)

# Lexicon Databases for Code-Mixed Processing
INDIC_TOKENS = {
    "bhai", "mera", "meri", "mere", "ka", "ki", "ke", "paise", "cut", "gaye", "gaya", "nahi", 
    "hua", "ho", "rahi", "raha", "hai", "hain", "baar", "ghante", "me", "karwa", "lunga", 
    "subah", "baje", "se", "chal", "bhejo", "mujhe", "naya", "chahiye", "kya", "honge", 
    "pe", "pichle", "mahine", "samajh", "kyun", "huye", "turant", "jaldi", "pareshan"
}

INTENT_LEXICON = {
    "Billing & Payments": ["recharge", "paise", "cut", "refund", "invoice", "charges", "bill", "bank", "account", "deducted"],
    "Network & Connectivity": ["call drop", "fiber", "router", "red light", "internet", "down", "signal", "tower", "slow"],
    "SIM & Account Services": ["sim", "e-sim", "port", "kyc", "documents", "convert", "activate", "deactivate"],
    "SLA Escalation / Churn": ["port", "jio", "airtel", "consumer forum", "fraud", "complaint", "engineer"]
}

URGENCY_TRIGGERS = [
    (r"\b(urgently|urgent|turant|jaldi|immediately)\b", 25),
    (r"\b(port|porting|switch|chhod dunga)\b", 30),
    (r"\b(paise cut|deducted|fail|refund)\b", 20),
    (r"\b(wfh|meeting|office|loss)\b", 15),
    (r"\b(\d+\s*(ghante|hours|mins?|minutes))\b", 15)
]

def process_indic_text(text):
    words = re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", text.lower())
    total_tokens = len(words)
    if total_tokens == 0:
        return 0, {}, 0, [], 0
    
    # Vernacular Code-Mixing Density
    indic_matches = [w for w in words if w in INDIC_TOKENS]
    mixing_ratio = round((len(indic_matches) / total_tokens) * 100, 1)
    
    # Intent Distribution Scoring
    intent_scores = {}
    for intent, kws in INTENT_LEXICON.items():
        score = sum(1 for kw in kws if kw in text.lower())
        intent_scores[intent] = score
    
    # Normalize Intent Probabilities
    total_intent_hits = sum(intent_scores.values()) or 1
    intent_distribution = {k: round((v / total_intent_hits) * 100, 1) for k, v in intent_scores.items()}
    dominant_intent = max(intent_distribution, key=intent_distribution.get)
    
    # Urgency & SLA Escalation Score
    urgency_score = 0
    detected_signals = []
    for pattern, weight in URGENCY_TRIGGERS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            urgency_score += weight
            detected_signals.append(match.group(0))
    
    urgency_score = min(urgency_score, 100)
    
    return mixing_ratio, intent_distribution, dominant_intent, detected_signals, urgency_score

if st.button("Analyze Code-Mixed Stream", type="primary"):
    mix_ratio, intents, top_intent, signals, urgency = process_indic_text(user_input)
    
    # KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Indic Token Density", f"{mix_ratio}%", help="Proportion of vernacular tokens vs standard English")
    with c2:
        st.metric("Dominant Intent", top_intent)
    with c3:
        st.metric("Escalation Risk", f"{urgency}/100", delta="High Priority" if urgency >= urgency_threshold else "Normal Priority", delta_color="inverse" if urgency >= urgency_threshold else "normal")
    with c4:
        routed_dept = "L3 Retention Desk" if urgency >= urgency_threshold else "Automated Self-Care"
        st.metric("Routed Queue", routed_dept)

    col_chart, col_decision = st.columns([1.2, 1])

    with col_chart:
        df_intents = pd.DataFrame(list(intents.items()), columns=["Intent", "Confidence"])
        fig = px.bar(df_intents, x="Confidence", y="Intent", orientation='h', color="Confidence",
                     color_continuous_scale="Viridis", title="Multi-Label Intent Distribution")
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='#161b22', paper_bgcolor='#161b22', font_color='#c9d1d9')
        st.plotly_chart(fig, use_container_width=True)

    with col_decision:
        st.subheader("Autonomous Pipeline Actions")
        if urgency >= urgency_threshold:
            st.error("🚨 Critical SLA / Churn Breach Flagged")
            st.markdown(f"""
            - **Detected Risk Triggers:** {', '.join([f'`{s}`' for s in signals])}
            - **Automated Workflow:** Bypass L1 IVR; direct SIP dispatch to **Retention Queue**.
            - **Pre-emptive Action:** Auto-query Billing Ledger for reversal on transaction failures.
            - **SLA Deadline:** Reduced resolution window from 24h to **45 Minutes**.
            """)
        else:
            st.success("✅ Standard Request Envelope")
            st.markdown("""
            - **Workflow:** Dispatched to conversational self-care chatbot.
            - **SLA Category:** Standard Tier-2 (24-hour turnaround).
            - **No churn triggers detected.**
            """)