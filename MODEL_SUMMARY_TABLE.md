# 📊 Model Comparison: ML vs. Deep Learning

This document provides a clear breakdown of the models used in RAINWISE for both Rainfall and Flood prediction across the Machine Learning (Production) and Deep Learning (Phase 3) versions.

---

## 🏆 Model Architecture Summary

| Aspect | Task | Model Used | Implementation | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Machine Learning (ML)** | **Rainfall** | **XGBoost Regressor** | `models/rainfall_model.pkl` | **6 Features**: (Month, Temp, Humid, Press, Wind, Clouds). Superior at modeling seasonal non-linear atmospheric relationships. |
| **Machine Learning (ML)** | **Flood** | **Logistic Regression** | `models/flood_model.pkl` | **Balanced Weights**: High-recall approach (91% recall) optimized for life-safety. Best for catching rare flood events. |
| **Deep Learning (DL)** | **Rainfall** | **DNN Regressor** | *(Phase 4 Goal)* | Proposed for future scaling to massive time-series satellite datasets (ERA5/GPM) where feature hierarchies are complex. |
| **Deep Learning (DL)** | **Flood** | **FloodDNN (PyTorch)** | `DLmodels/flood_dnn.pth` | Multi-layer architecture (Linear + BatchNorm + ReLU) trained to capture latent correlations between topography and intensity. |

---

## 💡 Comparative Performance Analysis

| **Logistic (Prod)** | N/A | **0.952** | **0.910** |
| XGBoost | **0.929** | 0.903 | 0.600 |
| Naive Bayes | N/A | 0.920 | 0.613 |
| Random Forest | 0.930 | 0.917 | 0.133 |
| SVM | N/A | 0.862 | 0.815 |

**Analysis**: We have switched the production flood model to **Logistic Regression** because it achieves the highest **Recall (91%)** when using balanced class weights. In a disaster warning system, missing a flood (False Negative) is much more dangerous than a False Alarm (False Positive). Logistic Regression now provides the strongest safety net.

---

## 💡 Technical Implementation Details

### 1. Machine Learning (XGBoost) - *Current Production*
*   **Why XGBoost?** It is the "gold standard" for tabular Big Data and handles class imbalance via `scale_pos_weight`.
*   **Feature Engineering**: The flood model utilizes sophisticated spatial features including **Drainage Factor** (derived from slope/elevation) and **River Risk** (inverse distance).
*   **Performance**: In our Gujarat dataset, XGBoost achieved high accuracy (R²=0.929 for Rainfall) because it captures seasonal "tipping points" by incorporating the **Month** feature.

### 2. Deep Learning (PyTorch) - *Experimental / Phase 3*
*   **Why DNN?** Deep learning is used to explore high-dimensional feature interactions.
*   **Architecture**: The `FloodDNN` uses **Batch Normalization** to stabilize training and **Dropout** to prevent overfitting.
*   **Future Scope**: Transitioning to **LSTMs** to handle temporal memory of rainfall patterns.

---

## 🏁 Final Verdict for Viva/Defense
For the RAINWISE production pipeline, we use a **Hybrid Strategy**:
1. **Rainfall**: **XGBoost Regressor** (Best R² score).
2. **Flood**: **Logistic Regression (Balanced)** (Best Recall/Life-Safety).
This combination ensures high predictive accuracy for weather while maintaining a maximum-sensitivity safety net for citizens.
