# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting **fraudulent credit card transactions** using machine learning techniques on a **highly imbalanced dataset**.  
Since fraudulent transactions are extremely rare, traditional metrics like accuracy are misleading.  
The project emphasizes **recall-driven evaluation**, **precision–recall analysis**, and **threshold tuning** aligned with real-world fraud detection systems.

---

## 🎯 Objectives

- Understand and handle **extreme class imbalance**
- Perform meaningful **EDA** without misleading visualizations
- Build a **baseline fraud detection model**
- Optimize decision thresholds using **Precision–Recall curves**
- Compare **Logistic Regression** and **XGBoost**
- Select a final model based on **operational feasibility**, not accuracy

---

## 📊 Dataset Information

- **Source:** Kaggle – Credit Card Fraud Detection  
- **Link:** https://www.kaggle.com/mlg-ulb/creditcardfraud  

### Dataset Characteristics
- Total transactions: **284,807**
- Fraudulent transactions: **492 (~0.17%)**
- Legitimate transactions: **99.83%**

### Features

| Feature | Description |
|-------|------------|
| `Time` | Seconds elapsed since first transaction |
| `Amount` | Transaction amount |
| `V1`–`V28` | PCA-transformed numerical features |
| `Class` | Target variable (0 = Legitimate, 1 = Fraud) |

⚠️ **Note:**  
The dataset is **not included** in this repository due to licensing and size constraints.

---

## 📁 Data Setup

1. Download `creditcard.csv` from Kaggle  
2. Place it inside the `data/` directory:

data/creditcard.csv

3. Run the notebook normally

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights from EDA:
- Severe **class imbalance** requires non-standard evaluation
- `Amount` feature is right-skewed
- `Time` represents elapsed seconds, not clock time
- PCA features are already scaled
- Duplicate rows were removed to avoid bias

Unnecessary visualizations (e.g., heatmaps) were intentionally avoided.

---

## ⚙️ Preprocessing Steps

- Removed duplicate rows
- Handled minimal missing values
- Converted target `Class` to integer
- Scaled only `Time` and `Amount`
- Preserved PCA features as provided

Train-test split performed using **stratification** to preserve class ratio.

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression (Baseline)

- Used `class_weight='balanced'`
- Served as a strong, interpretable baseline
- Threshold optimized using **Precision–Recall curve**

### 2️⃣ XGBoost

- Used `scale_pos_weight` to handle imbalance
- Achieved higher recall but at the cost of excessive false positives
- Final selection based on business feasibility

---

## 📈 Evaluation Strategy

Due to extreme imbalance:

### ❌ Accuracy is NOT used as a primary metric  
### ✅ Primary metric: **Recall (Fraud Class)**

Other metrics considered:
- Precision
- F1-score
- Precision–Recall AUC
- ROC-AUC (secondary, for ranking quality)

---

## 🎚️ Threshold Optimization

Default probability threshold (0.5) is **arbitrary** for imbalanced data.

Thresholds were selected using:
- **Precision–Recall curve**
- Minimum recall constraints for fraud
- Operational feasibility (false positive control)

This ensures decisions are **data-driven and business-aligned**.

---

## 📌 Final Model Performance (Logistic Regression)

Confusion Matrix:
[[24158 275]
[ 9 40]]

| Metric | Fraud Class |
|------|------------|
| Recall | **0.82** |
| Precision | 0.13 |
| Accuracy | 0.99 (not meaningful) |

### Interpretation

- ~82% of frauds detected
- Some false positives accepted to reduce financial loss
- Suitable for real-world fraud detection systems

---

## 🏆 Final Model Selection

Although XGBoost achieved slightly higher recall, it produced **significantly more false positives**.  
Logistic Regression offered a **better precision–recall trade-off** and was selected as the **final model** due to its stability, interpretability, and operational suitability.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## 📂 Repository Structure

credit-card-fraud-detection/
│
├── credit_card_fraud_detection.ipynb
├── README.md
├── .gitignore
│
└── data/
└── README.md

---

## 💡 Key Learnings

- Accuracy is misleading for imbalanced datasets
- Precision–Recall curves are critical for fraud detection
- Threshold tuning is as important as model selection
- Simpler models can outperform complex ones operationally
- Business context must guide ML decisions
