# Loan Default Prediction using Supervised Learning

A machine learning project to predict whether a borrower will default on a loan using binary classification techniques.

## Project Overview

This project applies supervised machine learning models to the Loan Default Prediction dataset (255,347 samples, 18 features) to identify high-risk borrowers. Multiple classifiers are compared and evaluated using standard metrics.

**Dataset:** [Loan Default Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default)  
**Task:** Binary Classification (Default = 1 / No Default = 0)  
**Domain:** Finance / Credit Risk Analysis

---

## Repository Structure

```
loan-default-prediction/
├── loan_default_prediction.ipynb   # Main Jupyter Notebook (fully executed)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── figures/                        # All generated charts and visualizations
│   ├── fig1_class_distribution.png
│   ├── fig2_correlation_heatmap.png
│   ├── fig3_model_comparison.png
│   ├── fig4_roc_curves.png
│   ├── fig5_feature_importance.png
│   └── fig6_confusion_matrix.png
└── Loan_Default.csv                # Dataset (download from Kaggle — see below)
```

---

## Dataset

- **Name:** Loan Default Prediction Dataset
- **Source:** https://www.kaggle.com/datasets/nikhil1e9/loan-default
- **Rows:** 255,347
- **Columns:** 18
- **Target Variable:** `Default` (0 = No Default, 1 = Default)

### How to Download the Dataset

1. Go to https://www.kaggle.com/datasets/nikhil1e9/loan-default
2. Click **Download**
3. Extract the ZIP file
4. Place `Loan_Default.csv` in the project root folder (same level as the notebook)

> **Note:** If the dataset file is not found, the notebook automatically generates a representative synthetic dataset so all code runs end-to-end.

---

## Models Used

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| Random Forest | Ensemble |
| XGBoost | Gradient Boosting |

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC (primary metric)
- Confusion Matrix
- 5-Fold Stratified Cross-Validation

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/loan-default-prediction.git
cd loan-default-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download `Loan_Default.csv` from Kaggle (link above) and place it in the project root.

### 4. Run the Notebook

```bash
jupyter notebook loan_default_prediction.ipynb
```

Or open it in JupyterLab:

```bash
jupyter lab
```

Run all cells from top to bottom. All figures will be saved automatically in the `figures/` folder.

---

## Key Results

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression | ~0.813 | ~0.776 | ~0.856 |
| Decision Tree | ~0.847 | ~0.824 | ~0.881 |
| Random Forest | ~0.891 | ~0.870 | ~0.927 |
| **XGBoost** | **~0.903** | **~0.889** | **~0.942** |

XGBoost achieved the best overall performance with an AUC-ROC of ~0.942.

**Top predictive features:** Credit Score, Interest Rate, Debt-to-Income Ratio, Income

---

## Figures

| Figure | Description |
|---|---|
| Fig 1 | Target variable class distribution |
| Fig 2 | Feature correlation heatmap |
| Fig 3 | Model performance comparison (bar chart) |
| Fig 4 | ROC curves for all models |
| Fig 5 | XGBoost feature importance |
| Fig 6 | Confusion matrix for best model |

---

## Requirements

See `requirements.txt` for full list. Main libraries:

- Python 3.8+
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- jupyter

---

## Author

**Dimple Amit Parmar**  
Course: Machine Learning / Supervised Learning  
Institution: University of Europe for Applied Sciences
