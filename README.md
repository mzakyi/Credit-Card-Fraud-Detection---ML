# Credit Card Fraud Detection

This project develops a machine learning model to detect fraudulent credit card transactions. By analyzing transaction patterns from a real dataset, the model distinguishes between normal and fraudulent activity, helping financial institutions flag suspicious behavior early and reduce risk.

---

## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Project Workflow](#project-workflow)  
- [Model & Evaluation](#model--evaluation)  
- [Results](#results)  
- [How to Run](#how-to-run)  
- [Future Improvements](#future-improvements)

---

## Overview

The goal is to train a classification model that balances:  
- **Precision** → minimizing false alarms (legitimate transactions flagged as fraud).  
- **Recall** → catching as many fraudulent transactions as possible.  

Key challenge: **severe class imbalance** (fraudulent transactions are only a small fraction of total).  

---

## Dataset

- **File**: [`creditcardFraudDetection.csv`  ](https://media.geeksforgeeks.org/wp-content/uploads/20240904104950/creditcard.csv)
- **Size**: ~284,807 transactions  
- **Features**:  
  - `Time`: seconds since the first transaction  
  - `V1`–`V28`: anonymized (PCA-transformed) features  
  - `Amount`: transaction amount  
  - `Class`: target variable (0 = legitimate, 1 = fraud)  

---

## Installation

Run the following commands to set up dependencies:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Project Workflow

1. **Import libraries** (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn).  
2. **Load dataset** [`creditcardFraudDetection.csv`  ](https://media.geeksforgeeks.org/wp-content/uploads/20240904104950/creditcard.csv).  
3. **Exploratory Data Analysis (EDA)**  
   - Class imbalance check  
   - Distribution of transaction amounts  
   - Correlation heatmap  
4. **Data preparation**  
   - Define features (X) and labels (y)  
   - Train/test split  
5. **Model Training**  
   - Random Forest Classifier  
6. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix  

---

## Model & Evaluation

- **Model**: Random Forest Classifier  
- **Metrics**:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

---

## Results

The model achieves:  
- **High precision** → very few false positives  
- **Moderate recall** → some fraudulent cases are missed  
- **Near-perfect accuracy** → less informative due to imbalance  

---

## How to Run

1. Download [`creditcardFraudDetection.csv`  ](https://media.geeksforgeeks.org/wp-content/uploads/20240904104950/creditcard.csv) (dataset).  
2. Open the Jupyter Notebook:  

```bash
jupyter notebook f573533d-f668-4d1f-98ae-08edb6b3770a.ipynb
```

3. Run all cells sequentially to reproduce results.  

---

## Future Improvements

- Use **sampling techniques** (SMOTE, undersampling) to balance classes.  
- Try other models (XGBoost, LightGBM, Neural Networks).  
- Apply cross-validation for robust performance.  
- Experiment with anomaly detection approaches.  
- Optimize threshold tuning to trade off precision vs recall.  
