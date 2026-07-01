# Setup & Usage Guide

This document describes how to set up the Python development environment and run the scripts for synthetic data generation, machine learning model training, algorithm comparison, and anomaly detection.

---

## 1. Prerequisites & Installation

The pipeline requires **Python 3.10+** (recommended).

1.  **Clone the repository** (or navigate to the workspace root):
    ```bash
    cd OliveOil-QualityPrediction
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment**:
    *   **Windows (PowerShell)**:
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    *   **macOS / Linux**:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements-ml.txt
    ```

---

## 2. Command Reference

All scripts are executed from the **repository root directory**.

### A. Dataset Synthesis
Generate custom tabular datasets for training or simulation:
```bash
# Generate a 1,000-batch dataset for model training
python src/generate_synthetic_batch.py --count 1000 --csv datasets/dataset-1000.csv

# Print a single simulated batch structure to stdout (dry-run)
python src/generate_synthetic_batch.py --count 1 --dry-run
```

---

### B. Machine Learning Pipeline Training
To train the primary multi-output regression model (CatBoost) or other candidates:

#### 1. Train primary CatBoost Model
```bash
# Trains CatBoost, saves the binary pipeline to models/catboost_pipeline.pkl and plots to docs/ml_assets/
python src/train_catboost.py
```

#### 2. Train Random Forest Model
```bash
# Trains Random Forest, saves model to models/rf_pipeline.pkl
python src/train_rf.py
```

#### 3. Train XGBoost Model
```bash
# Trains XGBoost, saves model to models/xgb_pipeline.pkl
python src/train_xgb.py
```

---

### C. Automated Benchmarking & Comparison
To train all three models, evaluate them head-to-head under identical conditions, and generate comparison reports:
```bash
# Trains all models, writes comparison report to docs/model_comparison.md and charts to docs/ml_assets/
python src/train_all_and_compare.py
```

---

### D. Anomaly Detection
To run anomaly detection check, import the validation helper into your python code:
```python
from src.anomaly_detection import detect_anomaly

# Telemetry list
telemetry = [26.0, 26.5, 27.0, 26.8, 48.0] # 48.0 is anomalous

# Run check
if detect_anomaly(telemetry, "temperature"):
    print("Warning: Sensor Anomaly Detected!")
```
For more information, see the [Sensor Anomaly Detection](anomaly_detection.md) documentation.
