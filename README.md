# 🛡️ Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)]()
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

An end-to-end machine learning system for detecting fraudulent financial transactions — built as a diploma thesis, designed as a deployable, explainable, production-grade pipeline rather than a notebook exercise.

## Key Contributions

- **Recency-weighted LightGBM** model with 467 engineered features
- **Chronological train/val/test split** to eliminate temporal data leakage
- **PR-AUC** as primary metric, chosen for severe class imbalance
- **Isotonic calibration** for reliable, decision-ready probabilities
- **SHAP explainability** for global and per-transaction interpretability
- **PSI drift monitoring** for production reliability
- **FastAPI + Streamlit** for real-time inference and monitoring

## Dataset

| Property | Value |
|---|---|
| Source | Fraud Detection Dataset |
| Transactions | ~590,000 |
| Fraud rate | ~3.5% |
| Features | 467 (engineered) |
| Split | Chronological |

## Results

| Metric | Score |
|---|---|
| PR-AUC | 0.606 |
| ROC-AUC | ~0.929 |
| ECE (calibration) | 0.007 |

All metrics evaluated on a strictly held-out, chronologically later test set.

## Architecture

```
Raw Data → Feature Engineering (467 features)
         → LightGBM (chronological split)
         → Isotonic Calibration + SHAP + PSI Monitoring
         → FastAPI (serving) → Streamlit (dashboard)
```

## Project Structure

```
fraud_detection_diploma/
├── configs/          # configuration files
├── data/             # raw, processed, external
├── notebooks/        # EDA & experiments
├── src/              # data, features, models, evaluation, utils
├── scripts/          # training pipeline scripts
├── app/
│   ├── api/          # FastAPI service
│   └── dashboard/     # Streamlit dashboard
├── figures/          # SHAP, PR curves, PSI plots
├── outputs/          # trained models
├── requirements.txt
└── README.md
```

## Installation & Usage

```bash
git clone https://github.com/yourusername/fraud_detection_diploma.git
cd fraud_detection_diploma
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train
python scripts/run_final_ieee_lightgbm.py

# Serve
uvicorn app.api.main:app --reload

# Dashboard
streamlit run app/dashboard/main.py
```

## Future Work

- Graph-based features for transaction network relationships
- Online/incremental learning for faster drift response
- Benchmarking against deep learning tabular models (e.g., TabNet)
- Automated PSI-triggered retraining pipeline

## License

MIT

---

*Diploma thesis: "Development of an ML Model for Detecting Fraudulent Operations."*