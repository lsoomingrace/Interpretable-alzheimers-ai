# Interpretable Machine Learning for Alzheimer's Disease Prediction

An interpretable machine learning project for Alzheimer's disease classification using a synthetic patient dataset. The project focuses on three aspects: **predictive performance, model interpretability, and prediction uncertainty**.

## Models & Methods

- Logistic Regression — baseline model
- Random Forest — primary interpretable model
- XGBoost — performance benchmark
- SHAP — global and patient-level interpretation
- PCA — patient data structure
- MAPIE — conformal prediction and uncertainty quantification

## Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 81.6% |
| Random Forest | 93.7% |
| XGBoost | **94.9%** |

Although XGBoost achieved the highest accuracy, Random Forest was used for further interpretation and uncertainty analysis due to its strong performance and interpretability.

## Interactive Demo

A simplified Random Forest model using five selected features was developed for an interactive Streamlit application:

**MMSE · Functional Assessment · ADL · Memory Complaints · Behavioral Problems**

The demo provides:
- Predicted class and probability
- 90% conformal prediction set
- Patient-level SHAP explanation

The simplified model achieved **94.4% test accuracy**.

## Repository

- `alzheimer_analysis.ipynb` — main analysis
- `demo_alz.ipynb` — simplified demo model
- `app.py` — Streamlit application

## Disclaimer

This project uses synthetic data and is intended for educational and portfolio purposes only. It is not a clinical diagnostic tool.
