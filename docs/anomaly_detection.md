# Sensor Anomaly Detection

This document details the anomaly detection sub-system integrated into the **Olive Oil Quality Prediction & Monitoring** pipeline. 

To ensure the integrity of the predictive models, input sensor data must be validated in real-time. Unrealistic sensor readings (due to hardware failure, calibration drift, or process spikes) are flagged as anomalies.

---

## 1. Methodology & Algorithm

The system uses the **Isolation Forest (iForest)** algorithm, which is highly effective for multidimensional tabular sensor streams:
*   **Concept**: Instead of profiling "normal" data points, Isolation Forest isolates anomalies by randomly partitioning features. Anomalies require fewer splits to isolate in a tree structure, resulting in shorter path lengths.
*   **Parameters**:
    *   `contamination=0.1`: The expected proportion of outliers in the data stream (set to 10% by default).
    *   `random_state=42`: Establishes deterministic tree partitioning.

---

## 2. In-Memory Model Caching

For real-time operational efficiency, the detector implements a **class-level model cache** (`_model_cache`). 
*   **Purpose**: Fitting an Isolation Forest model on every incoming sensor telemetry packet is computationally prohibitive.
*   **Workflow**:
    1.  When sensor readings for a feature (e.g., `malaxationTemperature`) are checked, the system looks for an already trained model in the cache.
    2.  If found, the model is reused for inference immediately.
    3.  If not found, it trains a new model on the historical baseline stream, caches it, and then scores the latest point.
*   **Cache Management**: The system provides `clear_cache()` to invalidate models if baseline process configurations change.

---

## 3. Class Reference & Implementation

The module is implemented in [src/anomaly_detection.py](../src/anomaly_detection.py).

### AnomalyDetector Class
```python
class AnomalyDetector:
    def __init__(self, contamination=0.1, random_state=42):
        """Initializes IsolationForest parameter configuration."""
        ...
        
    def fit(self, data, feature):
        """Trains the Isolation Forest on a sensor data stream and caches the model."""
        ...
        
    def predict(self, data):
        """Predicts outlier status. Returns -1 for anomalies, 1 for normal."""
        ...
        
    def predict_proba(self, data):
        """Calculates raw anomaly scores (lower scores indicate high anomaly likelihood)."""
        ...
```

### Quick Inference Helper
A convenience function `detect_anomaly` is provided for quick checks:
```python
def detect_anomaly(data, feature):
    """
    Check if sensor data contains anomalies using cached models.
    
    Args:
        data: List of historical/current sensor readings
        feature: Sensor type (e.g., 'temperature', 'moisture')
    
    Returns:
        Boolean: True if the latest data point is anomalous
    """
    ...
```

---

## 4. Usage Example

Here is a basic code example demonstrating telemetry validation:

```python
from src.anomaly_detection import detect_anomaly

# Historic sensor stream of malaxation temperatures (in °C)
temperatures = [26.5, 27.1, 26.8, 27.2, 27.0, 26.9, 27.1, 26.7, 45.2] # 45.2 is an anomaly

# Validate the telemetry stream
is_anomaly = detect_anomaly(temperatures, "malaxation_temperature")

if is_anomaly:
    print("WARNING: Anomaly detected in malaxation temperature sensor stream!")
else:
    print("Sensor telemetry is normal.")
```
