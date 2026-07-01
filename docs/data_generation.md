# Synthetic Data Generation

This document details the stochastic process simulation used in `generate_synthetic_batch.py` to synthesize realistic olive oil extraction datasets. The formulas are designed to capture the physical, chemical, and sensory dynamics described in olive oil extraction literature.

---

## 1. Process Simulation Formulas

The simulation propagates raw agricultural inputs and malaxation process settings through a sequence of stochastic equations to compute quality metrics.

### A. Process Yield ($Y$)
Oil droplet coalescence is modeled as a function of temperature ($T$), malaxation time ($t$), olive defect index ($D$), and moisture content ($M$).
*   **Temperature factor ($f_T$)**: A parabola peaking at $27^\circ\text{C}$:
    $$f_T = \max\left(0.5, 1.0 - 0.005 \times (T - 27)^2\right)$$
*   **Time factor ($f_t$)**: Plateaus at 45 minutes, with decreased efficiency if malaxation is too short or too long:
    $$f_t = \begin{cases} 
    \max\left(0.7, 1.0 - 0.005 \times (30 - t)^2\right) & \text{if } t < 30 \\
    1.0 & \text{if } 30 \le t \le 45 \\
    \max\left(0.7, 1.0 - 0.002 \times (t - 45)^2\right) & \text{if } t > 45
    \end{cases}$$
*   **Defect factor ($f_D$)**: High fruit defects impair extraction yield:
    $$f_D = \max(0.6, 1.0 - 0.4 \times D)$$
*   **Moisture factor ($f_M$)**: Excessive moisture ($>50\%$) forms emulsions that hinder oil extraction:
    $$f_M = \begin{cases} 
    1.0 & \text{if } M \le 50 \\
    \max\left(0.8, 1.0 - 0.01 \times (M - 50)\right) & \text{if } M > 50
    \end{cases}$$

The final yield is bounded by the raw oil content ($O_C$) of the batch and incorporates Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.2)$:
$$Y = \max\left(0.0, \min\left(O_C, O_C \times f_T \times f_t \times f_D \times f_M + \epsilon\right)\right)$$

---

### B. Total Phenols ($P$)
Phenolic content determines olive oil's antioxidant properties and is reduced by ripeness (Maturation Index $MI$), malaxation time, and washing with water (water-to-paste ratio $W_r$).
*   **Baseline Phenols**: Decreases linearly with ripeness:
    $$P_{\text{base}} = \max(100, 600 - 50 \times MI)$$
*   **Temperature effect ($g_T$)**: Promotes partition of phenols into the oil phase up to $30^\circ\text{C}$, after which thermal degradation occurs:
    $$g_T = \begin{cases} 
    1.0 + 0.02 \times (T - 20) & \text{if } T < 30 \\
    1.2 - 0.05 \times (T - 30) & \text{if } T \ge 30
    \end{cases}$$
*   **Time effect ($g_t$)**: Extended malaxation exposes phenols to oxidases, decreasing concentration:
    $$g_t = \begin{cases} 
    1.0 & \text{if } t \le 30 \\
    \max\left(0.5, 1.0 - 0.015 \times (t - 30)\right) & \text{if } t > 30
    \end{cases}$$
*   **Water ratio effect ($g_W$)**: Phenols are highly water-soluble; adding water washes them out:
    $$g_W = \max(0.4, 1.0 - 0.02 \times W_r)$$

With Gaussian noise $\epsilon \sim \mathcal{N}(0, 10.0)$:
$$P = \max\left(0.0, P_{\text{base}} \times g_T \times g_t \times g_W + \epsilon\right)$$

---

### C. Free Acidity ($A$)
Acidity (measured in % oleic acid) is driven primarily by hydrolytic enzyme activity from damaged fruit (defect index $D$) and prolonged malaxation:
$$A = \max\left(0.1, 0.2 + 1.5 \times D + h_t + \epsilon\right)$$
Where $h_t = 0.005 \times (t - 30)$ if $t > 30$ else $0$, and noise $\epsilon \sim \mathcal{N}(0, 0.02)$.

---

### D. Peroxide Value ($PV$)
Peroxides represent primary oxidation products, which increase with temperature and malaxation time (exposure to oxygen). Phenols act as antioxidants to mitigate this rise:
$$PV = \max\left(2.0, 5.0 + 0.5 \times (T - 25) + 0.2 \times (t - 30) - 0.005 \times P + \epsilon\right)$$
Where noise $\epsilon \sim \mathcal{N}(0, 0.5)$.

---

### E. UV Absorbance Spectrophotometry
UV indices ($K_{232}$, $K_{270}$, and $\Delta K$) indicate secondary oxidation:
*   **$K_{232}$**: Bounded by peroxide concentration:
    $$K_{232} = \max\left(1.0, 1.5 + 0.05 \times (PV - 5.0) + \mathcal{N}(0, 0.05)\right)$$
*   **$K_{270}$**: Bounded by acidity/hydrolytic alterations:
    $$K_{270} = \max\left(0.05, 0.12 + 0.1 \times (A - 0.2) + \mathcal{N}(0, 0.01)\right)$$
*   **$\Delta K$**: Bounded by acidity:
    $$\Delta K = \max\left(-0.01, 0.005 + 0.005 \times (A - 0.2) + \mathcal{N}(0, 0.001)\right)$$

---

### F. Sensory Profiles
*   **Fruity**: Optimal under cold extraction ($T \le 27^\circ\text{C}$):
    $$\text{Fruity} = \max\left(0.0, 5.0 - d_T + \mathcal{N}(0, 0.2)\right)$$
    Where $d_T = 0.5 \times (T - 27)$ if $T > 27$, else $0.1 \times (27 - T)$.
*   **Bitter & Pungent**: Highly dependent on the polyphenol profile ($P$):
    $$\text{Bitter} = \max(0.0, 0.01 \times P + \mathcal{N}(0, 0.2))$$
    $$\text{Pungent} = \max(0.0, 0.012 \times P + \mathcal{N}(0, 0.2))$$

---

## 2. Output Formats

The generation script supporting [src/generate_synthetic_batch.py](../src/generate_synthetic_batch.py) can export data in two ways:

1.  **Flat CSV format**: Ideal for training machine learning models. Every row represents a single batch.
2.  **Eclipse Ditto JSON representation**: Formatted as an industrial Digital Twin "Thing" object, which separates agricultural properties, process parameters, and oil quality measurements.

Example of Ditto JSON output:
```json
{
  "thingId": "olive.batch:batch001",
  "policyId": "olive.default:policy",
  "features": {
    "oliveParameters": { "properties": { "maturationIndex": 2.5, "moistureContent": 50.2, ... } },
    "processParameters": { "properties": { "malaxationTemperature": 26.5, ... } },
    "oliveOilQuality": { "properties": { "yieldPercentage": 18.2, "isEvooCompliant": true, ... } }
  }
}
```

---

## 3. Command Usage

To generate a flat CSV dataset of 1,000 batches for model training:
```bash
python src/generate_synthetic_batch.py --count 1000 --csv datasets/dataset-1000.csv
```

To dry-run print a single batch JSON:
```bash
python src/generate_synthetic_batch.py --count 1 --dry-run
```
