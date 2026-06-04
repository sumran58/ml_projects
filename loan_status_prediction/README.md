# 💰 Loan Approval Prediction

A machine learning project that predicts whether a loan application will be **approved** or **rejected** based on applicant details, using a Support Vector Machine (SVM) classifier.

---

## 📋 Overview

This project uses applicant demographics, income, credit history, and property details to predict the outcome of a loan application. Categorical features are encoded, missing values are dropped, and a linear SVM is trained on the cleaned dataset.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | `train_u6lujuX_CVtuZ9i.csv` (Analytics Vidhya Loan Prediction dataset) |
| **Target** | `Loan_Status` — `Y` (approved) / `N` (rejected) |

**Features:** `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`

---

## ⚙️ Requirements

```bash
pip install numpy pandas seaborn scikit-learn
```

---

## 🔁 Workflow

1. **Load Data** — read the CSV into a pandas DataFrame.
2. **Explore** — inspect shape, summary statistics, and missing values.
3. **Clean** — drop rows with missing values.
4. **Encode Labels:**
   - `Loan_Status`: Y → 1, N → 0
   - `Dependents`: '3+' → 4
   - `Gender`, `Married`, `Education`, `Self_Employed`, `Property_Area` → numeric
5. **Visualize** — count plots of `Education` and `Married` vs. `Loan_Status`.
6. **Split Features** — drop `Loan_ID` and `Loan_Status` from features.
7. **Train/Test Split** — 90% train / 10% test (stratified, `random_state=2`).
8. **Train Model** — fit an `SVC` with a linear kernel.
9. **Evaluate** — accuracy on both train and test sets.
10. **Predict** — classify a single applicant from the test set.

---

## 🤖 Model

**`SVC(kernel='linear')`** — a Support Vector Machine with a linear kernel, well-suited for binary classification on small tabular datasets.

**Evaluation Metric:** Accuracy Score.

---

## 🚀 Usage

```bash
jupyter notebook loan_prediction.ipynb
```

Place the dataset CSV in the same directory (or update the path) and run cells top to bottom.

---

## 📁 Project Structure

```
.
├── loan_prediction.ipynb              # Main notebook
├── train_u6lujuX_CVtuZ9i.csv          # Dataset
└── README.md
```

---

## 📝 Notes

- The dataset path `/content/...` is Colab-specific — update for local runs.
- `dropna()` removes ~134 rows; imputation (mean/mode) would preserve more data.
- Features aren't scaled — SVM with linear kernel still works, but RBF kernel would need `StandardScaler` first.
- 10% test size is small (~48 rows); test accuracy may vary noticeably between runs.

---

## 📄 License

Free to use for learning and personal projects.
