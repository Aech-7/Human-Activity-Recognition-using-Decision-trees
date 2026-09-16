# Human Activity Recognition using Decision Trees

**End-to-end machine learning pipeline for classifying human activities from smartphone accelerometer data.**

[![Python](https://img.shields.io/badge/Python-3.x-blue)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)]()
[![TSFEL](https://img.shields.io/badge/TSFEL-Feature%20Engineering-green)]()

## Project Overview

This project develops a **Human Activity Recognition (HAR)** system using smartphone accelerometer data to classify six activities:

**Walking · Walking Upstairs · Walking Downstairs · Sitting · Standing · Laying**

The project focuses on understanding how **feature representation affects model performance**, comparing raw sensor signals, TSFEL-extracted features, and features provided with the UCI-HAR dataset.

The complete pipeline includes:

```text
Sensor Data
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
PCA Visualization
    ↓
Decision Tree Classification
    ↓
Model Evaluation
    ↓
Error Analysis
    ↓
Real-World Validation
```

---

## Technical Highlights

* Exploratory analysis of multivariate accelerometer time-series data
* Feature engineering using **TSFEL**
* Dimensionality reduction using **PCA**
* Decision Tree classification using **Scikit-learn**
* Model comparison across multiple feature representations
* Hyperparameter analysis through varying tree depth
* Accuracy, precision, recall and confusion-matrix based evaluation
* Error analysis of misclassified activities
* Evaluation on **self-collected smartphone sensor data**

---

## Dataset

The primary dataset is the **UCI Human Activity Recognition Using Smartphones Dataset**.

The dataset contains smartphone sensor measurements collected from participants performing six activities. Accelerometer signals are sampled at **50 Hz**.

| Activity           | Type    |
| ------------------ | ------- |
| Walking            | Dynamic |
| Walking Upstairs   | Dynamic |
| Walking Downstairs | Dynamic |
| Sitting            | Static  |
| Standing           | Static  |
| Laying             | Static  |

---

# 1. Exploratory Data Analysis

The first stage investigates the structure and characteristics of the accelerometer signals.

### Activity Waveforms

Representative signals from all six activities are visualized to examine differences in their temporal patterns.

![Activity Waveforms](results/eda/activity_waveforms.png)

### Static vs Dynamic Activities

Acceleration magnitude is analyzed using:

```text
accx² + accy² + accz²
```

to investigate whether static and dynamic activities exhibit distinguishable sensor patterns.

![Acceleration Magnitude](results/eda/acceleration_magnitude.png)

### PCA Visualization

PCA is used to project three feature representations into two dimensions:

* Total acceleration
* TSFEL features
* Dataset-provided features

![PCA Comparison](results/pca/pca_comparison.png)

This provides a visual comparison of class separability before classification.

---

# 2. Decision Tree Classification

Decision Trees are trained using three different representations of the sensor data:

| Model   | Input Representation     |
| ------- | ------------------------ |
| Model 1 | Raw accelerometer data   |
| Model 2 | TSFEL features           |
| Model 3 | UCI-HAR dataset features |

### Model Performance

| Feature Representation |   Accuracy | Precision | Recall |
| ---------------------- | ---------: | --------: | -----: |
| Raw Accelerometer      | **XX.XX%** |    XX.XX% | XX.XX% |
| TSFEL Features         | **XX.XX%** |    XX.XX% | XX.XX% |
| Dataset Features       | **XX.XX%** |    XX.XX% | XX.XX% |

![Model Comparison](results/decision_tree/model_comparison.png)

### Confusion Matrix

![Confusion Matrices](results/decision_tree/confusion_matrices.png)

The confusion matrices are used to identify activities that are difficult for the classifier to distinguish.

### Effect of Tree Depth

Tree depth is varied from **2 to 8** to study the relationship between model complexity and test performance.

![Accuracy vs Tree Depth](results/decision_tree/accuracy_vs_depth.png)

**Best-performing configuration:**

| Feature Representation | Best Depth | Accuracy |
| ---------------------- | ---------: | -------: |
| Raw Accelerometer      |          X |   XX.XX% |
| TSFEL Features         |          X |   XX.XX% |
| Dataset Features       |          X |   XX.XX% |

---

# 3. Real-World Validation

To evaluate model generalization beyond the benchmark dataset, smartphone accelerometer data was independently collected for the same six activities.

The collected data was processed using the same preprocessing and feature-extraction pipeline before being passed to the trained model.

### Real-World Results

| Metric    | UCI-HAR Test Set | Self-Collected Data |
| --------- | ---------------: | ------------------: |
| Accuracy  |           XX.XX% |              XX.XX% |
| Precision |           XX.XX% |              XX.XX% |
| Recall    |           XX.XX% |              XX.XX% |

![Real World Confusion Matrix](results/real_world/real_world_confusion_matrix.png)

### Generalization

![Benchmark vs Real World](results/real_world/benchmark_vs_real_world.png)

The difference between benchmark and self-collected performance highlights the effect of real-world sensor variation and provides a practical evaluation of model generalization.

---

## Key Results

* **Best feature representation:** XX
* **Best Decision Tree depth:** XX
* **Best UCI-HAR accuracy:** XX.XX%
* **Real-world accuracy:** XX.XX%
* **Most frequently confused activities:** XX and XX

---

## Repository Structure

```text
├── 01_eda.ipynb
├── 02_decision_tree.ipynb
├── 03_real_world_evaluation.ipynb
├── data/
└── results/
```

Each notebook corresponds to one stage of the project and contains the analysis, experiments and results.

---

## Technologies

**Python · NumPy · Pandas · Matplotlib · Scikit-learn · TSFEL · Jupyter**

---

## Conclusion

This project demonstrates an end-to-end approach to **sensor-based activity recognition**, from raw accelerometer data and feature engineering to model selection, error analysis and real-world validation.

The experiments show how the choice of feature representation and model complexity influences classification performance, while the real-world evaluation provides insight into how a model trained on a benchmark dataset performs on independently collected sensor data.
