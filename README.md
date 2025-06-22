# AAI-540 Final Project — Predicting SPY Short-Term Moves with AWS MLOps

This repository contains all materials for Group #1's final project for AAI-540: **Machine Learning Operations**. We built a fully reproducible, production-ready machine learning pipeline using AWS services, focused on predicting the next-day directional movement (up/down) for SPY, the S&P 500 ETF.

## Project Overview

- **Goal:** Predict when SPY will close higher the next day using technical and volume-based features, with an emphasis on realistic, risk-aware trade filtering.
- **Approach:** Leverage AWS S3 for all data storage, Athena for data cataloging and EDA, and SageMaker for the full ML workflow — from data prep to CI/CD and model registry.
- **Scope:** We focused on batch inference (not real-time endpoints) and implemented simulated monitoring for system resources and model quality.

## Repository Structure

```
.
├── code/
│   ├── preprocess_spy_features.py  # Feature engineering script
│   └── evaluate.py                 # Model evaluation logic
├── data/
│   ├── spy_daily.csv               # Raw SPY OHLCV data
│   ├── qqq_daily.csv               # Raw QQQ data for relative strength
│   ├── features.csv                # Engineered features
│   ├── spy_daily_train.csv         # Train split
│   ├── spy_daily_val.csv           # Validation split
│   ├── spy_daily_test.csv          # Test split
│   ├── spy_daily_production.csv    # Production (holdout) split
│   └── batch_predictions_local.csv # Model predictions (batch job)
├── rf_model.joblib                 # Trained random forest model
├── Final_Project_Team_1_Deliverable_3.ipynb  # Main project notebook
└── README.md
```

## Quick Start

1. **Clone the repo**  
   ```
   git clone https://github.com/<your-org-or-username>/AAI540-Team-1.git
   cd AAI540-Team-1
   ```

2. **Review the notebook**  
   - All steps, from data ingest to monitoring and model registry, are reproducible in `Final_Project_Team_1_Deliverable_3.ipynb`.

3. **Data & Feature Scripts**  
   - Raw and processed data files are in the `data/` folder.
   - Feature engineering and evaluation scripts are in `code/`.

## System Design Highlights

- **Data Storage:**  
  - All data versioned and stored in S3.  
  - Athena catalog enables easy, SQL-based EDA and schema enforcement.

- **Feature Engineering:**  
  - Performed in SageMaker Notebook (and Python scripts), including technical indicators and volume-based logic.

- **Modeling:**  
  - Random Forest classifier trained and evaluated using accuracy, F1, Sharpe, and filtered for high-quality trade signals.
  - Batch inference is used for all predictions.

- **CI/CD & Model Registry:**  
  - Model training, evaluation, and registration fully automated using SageMaker and boto3.
  - Passing models are versioned in the SageMaker Model Registry.

- **Monitoring:**  
  - Infrastructure and model monitoring is simulated and plotted locally (no live endpoint in this phase).

## Key Files

- **Final_Project_Team_1_Deliverable_3.ipynb:**  
  Main project notebook, contains EDA, feature engineering, model training, CI/CD pipeline, batch inference, monitoring, and model registration.

- **code/preprocess_spy_features.py:**  
  Script for reproducible feature engineering.

- **code/evaluate.py:**  
  Script for model evaluation and metric computation.

## Contributors

- Aryaz Zomorodi
- Jay Patel
- Brian Johnson

---

**Questions or feedback?**  
Open an issue or reach out via the project discussion board.