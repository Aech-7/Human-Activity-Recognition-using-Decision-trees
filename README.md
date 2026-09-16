# Human Activity Recognition using Decision Trees

A machine learning project for **human activity recognition from smartphone accelerometer data**, covering feature analysis, Decision Tree classification, model evaluation, and real-world validation.

[![Python](https://img.shields.io/badge/Python-3.x-blue)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)]()
[![TSFEL](https://img.shields.io/badge/TSFEL-Feature%20Engineering-green)]()
    
## Overview

The goal is to classify six human activities from smartphone sensor data:

**Walking · Walking Upstairs · Walking Downstairs · Sitting · Standing · Laying**

The project follows an end-to-end ML workflow:

```text
Accelerometer Data
        ↓
Data Processing
        ↓
Feature Extraction
        ↓
PCA Analysis
        ↓
Decision Tree Classification
        ↓
Model Evaluation
        ↓
Real-World Validation
```

The project also includes a **from-scratch Decision Tree implementation** for classification and regression.

---

## 1. Exploratory Data Analysis

The accelerometer signals are analyzed to understand how different activities appear in the sensor data.

The analysis includes:

* Visualization of representative activity signals
* Comparison of static and dynamic activities
* Analysis of acceleration magnitude
* PCA on different feature representations
* Comparison of raw, TSFEL and dataset-provided features

![Activity Waveforms](HAR/results/activity_waveforms.png)

![PCA Comparison](HAR/results/pca_comparison.png)

---

## 2. Decision Tree Classification

Decision Tree models are trained using three feature representations:

| Feature Representation | Description                                  |
| ---------------------- | -------------------------------------------- |
| Raw Accelerometer      | Direct sensor measurements                   |
| TSFEL Features         | Automatically extracted time-series features |
| UCI-HAR Features       | Features provided with the dataset           |

The models are compared using:

* Accuracy
* Precision
* Recall
* Confusion matrices

### Model Performance

| Feature Representation |   Accuracy | Precision | Recall |
| ---------------------- | ---------: | --------: | -----: |
| Raw Accelerometer      | **XX.XX%** |    XX.XX% | XX.XX% |
| TSFEL Features         | **XX.XX%** |    XX.XX% | XX.XX% |
| UCI-HAR Features       | **XX.XX%** |    XX.XX% | XX.XX% |

![Model Comparison](HAR/results/model_comparison.png)

### Model Complexity

Decision Tree depth is varied from **2 to 8** to study the effect of model complexity on classification performance.

![Accuracy vs Tree Depth](HAR/results/accuracy_vs_depth.png)

Confusion matrices are also analyzed to identify activities that are frequently misclassified.

![Confusion Matrices](HAR/results/confusion_matrices.png)

---

## 3. Real-World Evaluation

To evaluate generalization beyond the benchmark dataset, smartphone accelerometer data was independently collected for the same six activities.

The collected data is processed using the same pipeline and evaluated using the trained model.

| Dataset             |   Accuracy | Precision | Recall |
| ------------------- | ---------: | --------: | -----: |
| UCI-HAR Test Set    | **XX.XX%** |    XX.XX% | XX.XX% |
| Self-Collected Data | **XX.XX%** |    XX.XX% | XX.XX% |

![Real-World Confusion Matrix](HAR/results/real_world_confusion_matrix.png)

This experiment evaluates how well a model trained on a standard benchmark dataset generalizes to independently collected sensor data.

---

## From-Scratch Decision Tree

A Decision Tree was also implemented from scratch in Python to understand the underlying classification and regression process.

The implementation is contained in:

```text
tree/
├── base.py
├── utils.py
└── __init__.py
```

The implementation is evaluated through additional experiments in the repository.

---

## Key Files

The main project workflow is contained in three notebooks:

* **`HAR/01_eda.ipynb`** — explores the accelerometer data, feature representations and PCA.
* **`HAR/02_har_decision_tree.ipynb`** — trains, compares and evaluates the Decision Tree models.
* **`HAR/03_real_world_evaluation.ipynb`** — tests the trained model on independently collected smartphone data.

The **`tree/`** directory contains the from-scratch Decision Tree implementation used for additional experiments.

---

## Key Results

| Experiment                  | Result      |
| --------------------------- | ----------- |
| Best feature representation | **XX**      |
| Best tree depth             | **XX**      |
| UCI-HAR accuracy            | **XX.XX%**  |
| Real-world accuracy         | **XX.XX%**  |
| Most confused activities    | **XX / XX** |

---

## Technologies

**Python · NumPy · Pandas · Scikit-learn · Matplotlib · TSFEL · Jupyter**

---

## Key Takeaways

* Different feature representations have a measurable impact on activity classification.
* PCA provides a useful view of activity separability.
* Tree depth affects the balance between model complexity and generalization.
* Confusion matrices help identify difficult activity pairs.
* Evaluation on independently collected data provides a practical test of model generalization.
* The from-scratch implementation provides a deeper understanding of Decision Tree algorithms.
The experiments show how the choice of feature representation and model complexity influences classification performance, while the real-world evaluation provides insight into how a model trained on a benchmark dataset performs on independently collected sensor data.
