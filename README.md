\# RoSense: Code-Mixed Indic NLP \& Enterprise Support Triage Engine



\[!\[Streamlit App](https://static.streamlit.io/badges/streamlit\_badge\_black\_white.svg)](#)

\[!\[Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



RoSense is a domain-specialized Natural Language Processing engine built to solve classification bottlenecks in code-mixed (Hinglish/Indic-English) enterprise support queries. Standard monolingual tokenizers frequently misclassify sentiment and intent on hybrid Indian conversational structures; RoSense tokenizes vernacular terms, extracts multi-label customer intent, and calculates real-time SLA escalation risks.



\## Key Architecture

1\. \*\*Subword \& Vernacular Tokenizer:\*\* Isolates dialectal keywords, phonetic variations, and grammatical markers to quantify code-mixing density.

2\. \*\*Multi-Label Intent Resolver:\*\* Dispatches unstructured text across 4 operational enterprise queues (Billing, Network Ops, SIM Services, Retention).

3\. \*\*Behavioral Urgency Matrix:\*\* Scans for operational pain points (financial deductions, WFH outage impact, port-out threats) to autonomously escalate ticket tiers.



\## Local Quickstart

```bash

git clone \[https://github.com/](https://github.com/)<your-username>/rosense.git

cd rosense

pip install -r requirements.txt

streamlit run app.py

