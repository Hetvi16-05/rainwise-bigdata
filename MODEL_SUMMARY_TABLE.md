# 📊 Model Comparison: ML vs. Deep Learning

This document provides a clear breakdown of the models used in RAINWISE for both Rainfall and Flood prediction across the Machine Learning (Production) and Deep Learning (Phase 3) versions.

---

## 🏆 Model Architecture Summary

| Aspect | Task | Model Used | Implementation | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Machine Learning (ML)** | **Rainfall** | **XGBoost Regressor** | `models/rainfall_model.pkl` | Superior at modeling non-linear atmospheric relationships (Temp, Humidity, Pressure) compared to standard regression. |
| **Machine Learning (ML)** | **Flood** | **XGBoost Classifier** | `models/flood_model.pkl` | Native handling of tabular GIS features (Elevation, River Distance) and robust performance on regional datasets. |
| **Deep Learning (DL)** | **Rainfall** | **DNN Regressor** | *(Phase 4 Goal)* | Proposed for future scaling to massive time-series satellite datasets (ERA5/GPM) where feature hierarchies are complex. |
| **Deep Learning (DL)** | **Flood** | **FloodDNN (PyTorch)** | `DLmodels/flood_dnn.pth` | Multi-layer architecture (Linear + BatchNorm + ReLU) trained to capture latent correlations between topography and intensity. |

---

## 💡 Detailed Technical Justification

### 1. Machine Learning (XGBoost) - *Current Production*
*   **Why XGBoost?** It is the "gold standard" for tabular Big Data. 
*   **Efficiency:** It runs extremely fast on the CPU (Hadoop Cluster) without requiring high-end GPUs.
*   **Performance:** In our Gujarat dataset, XGBoost achieved high accuracy (R²=0.917 for Rainfall) because it captures "tipping points" in weather data (e.g., when humidity hits a specific threshold).

### 2. Deep Learning (PyTorch) - *Experimental / Phase 3*
*   **Why DNN?** Deep learning is used to explore high-dimensional feature interactions that decision trees might miss in much larger datasets.
*   **Architecture:** The `FloodDNN` uses **Batch Normalization** to stabilize training and **Dropout** to prevent overfitting on the regional rainfall patterns.
*   **Future Scope:** An **LSTM (Long Short-Term Memory)** model is planned for Rainfall to handle "Temporal Dependency" (how yesterday's rain affects today's soil saturation).

---

## 🏁 Final Verdict for Viva/Defense
For our current dataset (captured via Hadoop/Spark), **XGBoost (ML)** is the most robust and accurate. The **Deep Learning (DL)** models serve as a highly advanced feasibility study for scaling the project to a global level with millions of satellite data points.
