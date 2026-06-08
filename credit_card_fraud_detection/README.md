# 💳 Credit Card Fraud Detection using Logistic Regression

Detects **fraudulent credit card transactions** using **Logistic Regression** with undersampling to handle extreme class imbalance.

---

## 📁 Files
```
credit_fraud_detection.ipynb
creditcard.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 📊 Dataset
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Records:** 284,807 transactions
- **Features:** 30 (V1–V28 PCA components + `Time` + `Amount`)
- **Target:** `Class` (0 = Legitimate, 1 = Fraudulent)
- **Highly Imbalanced:** ~99.8% legitimate vs ~0.2% fraudulent

## ⚙️ Handling Class Imbalance — Undersampling

```python
legit = df[df.Class == 0]
fraud = df[df.Class == 1]   # 188 fraud cases

# Sample equal number of legitimate transactions
legit_sample = legit.sample(n=188)

# Combine to create balanced dataset
new_df = pd.concat([legit_sample, fraud], axis=0)
# Result: 188 legit + 188 fraud = 376 balanced samples
```

> ⚠️ **Undersampling discards a large portion of legitimate data.** This can reduce the model's ability to generalize. Consider SMOTE as an alternative.

## 🤖 Model
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```
- Stratified 80/20 train-test split on the balanced dataset

## 📈 Evaluation
- Metric: **Accuracy Score** on test set

## ⚠️ Known Limitations
- **Undersampling** reduces dataset to only 376 samples — model may not generalize to real-world scale
- **Accuracy alone is misleading** for fraud detection; Precision, Recall, F1, and AUC-ROC are more meaningful
- No prediction system built for new transactions

## 🔮 Future Work
- Replace undersampling with **SMOTE** (Synthetic Minority Oversampling Technique)
- Use **AUC-ROC, Precision-Recall** as primary metrics
- Try Random Forest or XGBoost which handle imbalance better
- Add a real-time transaction prediction system
- Train on the full dataset with `class_weight='balanced'`
