# Olive Oil Quality Prediction & Telemetry Monitoring

An industrial Machine Learning pipeline for real-time process monitoring, quality prediction, and sensor anomaly detection in olive oil extraction mills.

This repository implements a **nonlinear multi-target regression pipeline** to predict continuous oil quality indices and process yield based on raw agricultural parameters and extraction process variables. It also includes an **in-memory cached anomaly detection engine** to flag erratic sensor readings before they reach the prediction model.

---

## 🚀 Key Features

*   **Nonlinear Multi-Target Regression**: Predicts 10 continuous quality and yield outputs from 9 heterogeneous (categorical & numerical) input features.
*   **Production-Grade Model Architecture**: Employs **CatBoost** as the primary regressor, taking advantage of native categorical feature handling, ordered boosting, and small memory footprint.
*   **Stochastic Extraction Simulator**: Synthesizes custom olive batch data based on mathematical models from olive-extraction scientific literature.
*   **Telemetry Anomaly Validation**: Incorporates **Isolation Forest (iForest)** with class-level caching for high-speed sensor stream checking.
*   **Comparative Benchmarking**: Multi-model comparison suite evaluating CatBoost, Random Forest, and XGBoost performances.

---

## 📂 Repository Structure

The project has been organized into a professional, modular structure:

```
OliveOil-QualityPrediction/
├── datasets/                 # Pre-generated synthetic datasets (1K and 3K samples)
│   ├── dataset-1000.csv
│   └── dataset-3000.csv
├── docs/                     # Comprehensive documentation and reports
│   ├── machine_learning.md   # ML pipeline details, IOC thresholds, and feature space
│   ├── model_comparison.md   # head-to-head benchmarking reports and charts
│   ├── anomaly_detection.md  # Isolation Forest architecture and cache mechanism
│   ├── data_generation.md    # Process equations and stochastic variables math
│   ├── setup_and_usage.md    # Installation and operational command guides
│   └── ml_assets/            # Visualization charts and R2 performance plots
├── src/                      # Source code directory
│   ├── anomaly_detection.py  # Anomaly detector module
│   ├── generate_synthetic_batch.py # Data generation script
│   ├── train_catboost.py     # Primary model training pipeline
│   ├── train_rf.py           # Random Forest training script
│   ├── train_xgb.py          # XGBoost training script
│   └── train_all_and_compare.py # Benchmarking pipeline
├── .gitignore                # Git ignore configuration
├── requirements-ml.txt       # Python environment dependencies
└── README.md                 # Project landing page (this file)
```

---

## 📖 Detailed Documentation

Click the links below to view the dedicated in-depth documentation:

1.  **[Machine Learning Pipeline](docs/machine_learning.md)**: Deep dive into the regression targets, feature space, CatBoost reasoning, and deterministic post-processing for International Olive Council (IOC) EVOO compliance.
2.  **[Model Benchmarking & Comparison](docs/model_comparison.md)**: Side-by-side performance results ($R^2$ scores per target, EVOO accuracy, categorical handling comparisons) between CatBoost, Random Forest, and XGBoost.
3.  **[Sensor Anomaly Detection](docs/anomaly_detection.md)**: Architectural analysis of the Isolation Forest validation sub-system, in-memory model caching, and convenience inference helper functions.
4.  **[Synthetic Data Generation](docs/data_generation.md)**: Detailed chemical and physical process formulas (yield, phenols partition, UV indices $K_{232}$/$K_{270}$/$\Delta K$, and sensory markers) used for stochastic data synthesis.
5.  **[Setup & Usage Guide](docs/setup_and_usage.md)**: Complete step-by-step installation instructions, environment activation, and operational command reference.

---

## ⚡ Quick Start

For detailed instructions, refer to the **[Setup & Usage Guide](docs/setup_and_usage.md)**.

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements-ml.txt
```

### 2. Generate Data & Train Model
```bash
# Generate data
python src/generate_synthetic_batch.py --count 1000 --csv datasets/dataset-1000.csv

# Train CatBoost (saves model to models/catboost_pipeline.pkl and plots to docs/ml_assets/)
python src/train_catboost.py
```

### 3. Run Benchmark Suite
```bash
# Train all models and output docs/model_comparison.md head-to-head report
python src/train_all_and_compare.py
```
