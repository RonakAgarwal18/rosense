<div align="center">

# ROSense

**Code-Mixed Indic NLP & Support Triage Engine**

*Reads Hinglish complaints the way customers actually write them.*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)

### [▶ Live Demo](https://rosense.streamlit.app)

</div>

---

## What it does

Customer support in India runs on Hinglish — complaints that mix Hindi and English in the same sentence. Normal English classifiers misread them. ROSense reads a code-mixed telecom complaint, works out what it is about, scores how risky it is (churn threats, legal and regulator mentions, SLA danger), and routes it to the right support queue — most of it locally, in milliseconds, with a cloud LLM only when the case needs one.

Paste a complaint like:

> *"Kal mai call pe baat kar rha tha and achanak se call cut ho gya... isse accha to airtel hai"*

and ROSense returns:

- **Code-mix density** — how much of the message is Indic vs English
- **Intent** — one of 4 queues: Billing & Tariffs · Network Operations · SIM Services · Regulatory & Churn
- **Risk score (0–100)** — built from weighted warning signs in the text
- **Destination queue** and a short action note for the support team

## How it works

Three routing modes, selectable in the sidebar:

### Tier-1 · Edge classifier (default, no API key needed)

- Character-level TF-IDF (3–5 gram subwords) + Logistic Regression, trained on a built-in set of example complaints — subword features are what make it work on phonetic Hinglish spellings (*"gya"*, *"rha"*, *"bohot"*)
- Runs fully locally and answers in under ~15 ms
- A rule engine scans for risk triggers and adds weighted points: consumer court (+45), TRAI / regulator (+40), legal notice or fraud (+35), port-out / churn intent (+30), competitor mention (+25), abusive language (+25), repeated call drops (+25), strong dissatisfaction (+20), WFH / emergency impact (+15)

### Tier-2 · Cloud model

- Gemini 2.5 Flash reads the complaint zero-shot and returns the same fields as strict JSON
- Needs a Gemini API key — paste it in the sidebar, or set `GEMINI_API_KEY` in Streamlit secrets

### Cascading hybrid

- Edge first. If the risk score crosses the SLA threshold (slider, default 65) **or** the edge confidence drops below 60%, the case escalates to the cloud model automatically
- If the cloud call fails, the edge result is used — the queue never stalls

### Routing

| Case | Goes to |
|---|---|
| Risk ≥ threshold | L3 Executive Regulatory & Retention Desk (30-minute window) |
| Network Operations | NOC Automated Diagnostics |
| Billing & Tariffs | Tier-2 Revenue Assurance |
| Everything else | Tier-1 Digital Ingestion (standard 24-hour SLA) |

## Dashboard

Every run shows four headline metrics (code-mix %, intent, SLA risk, assigned queue), a confidence chart across all 4 intents, the exact risk triggers found in the text, and the action note for the team. Four ready-made test complaints are built into the sidebar — including a call-drop + competitor threat, a tariff-hike regulator complaint, a WFH fiber outage, and an eSIM request.

## Quick start

```bash
git clone https://github.com/RonakAgarwal18/rosense.git
cd rosense
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 and hit **Ingest and Evaluate Telemetry**. Tier-1 works immediately; add a Gemini API key in the sidebar to try the Tier-2 and hybrid modes.

## Project structure

```
app.py              The whole app: edge classifier, risk rules,
                    Gemini Tier-2, hybrid cascade, dashboard
requirements.txt    streamlit, pandas, plotly, scikit-learn, google-genai
```

## Tech stack

Python · Streamlit · Scikit-learn · Pandas · Plotly · Google Gemini (google-genai)

## Honest notes

- The edge model trains on a small built-in example set at startup — it shows the approach clearly, but a production version should train on real, labelled ticket data.
- Risk weights are hand-set from how Indian telecom escalations usually work; tune them per operator before real use.

---

<div align="center">

Built by [Ronak Agarwal](https://github.com/RonakAgarwal18)

</div>
