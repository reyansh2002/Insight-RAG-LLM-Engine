# 🚀 KPI Intelligence & Action Agent

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.0-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An AI-powered system that analyzes daily KPI data, detects anomalies, generates natural-language insights, and triggers automated actions such as alerts, reports, and recommendations.

This project combines **machine learning**, **LLMs**, and **agentic reasoning** to help teams monitor performance and take proactive decisions.

---

## 📌 Features

- **Automated KPI Analysis**: Reads daily KPI data (CSV) and computes trends, changes, and anomalies.
- **LLM-Powered Insights**: Generates human-like explanations for KPI changes using OpenAI-compatible models.
- **Rule-Based & ML-Driven Actions**: Suggests or executes actions such as alerts, notifications, or corrective steps.
- **Local Fallback Mode**: Works without API keys using deterministic, rule-based logic.
- **End-to-End Pipeline**: From data ingestion → analysis → explanation → recommended actions.
- **Interactive Dashboard**: Explore trends and anomalies visually using Streamlit & Plotly.

---

## 📂 Project Structure
main.py # Main KPI agent execution script
daily_kpi_data.csv # Input KPI dataset
requirements.txt # Dependencies
README.md # Project documentation

---

## 🛠️ Tech Stack

- Python 3.10+
- Pandas & Numpy for KPI calculations
- Streamlit for interactive dashboards
- Plotly for visual analytics
- OpenAI / Gemini LLM API (optional)
- Scikit-learn (optional for advanced anomaly detection)

Local fallback mode ensures the tool works **offline without API keys**.

---

## ⚙️ How It Works

1. **Load KPI data** from CSV
2. **Compute daily changes**, percent differences, and detect anomalies
3. **Generate insights** using either:
   - LLM model (if API key is available), or
   - Local rule-based logic
4. **Recommend actions** depending on KPI behavior
5. **Output a structured, human-readable report**

---

## ▶️ Usage

### Run the agent
```bash
python main.py

