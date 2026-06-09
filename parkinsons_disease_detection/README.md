
# 🧠 Parkinson's Disease Detection using SVM

Detects **Parkinson's disease** from biomedical voice measurements using a **Support Vector Machine (SVM)**.

---

## 📁 Files
```
parkinsons_disease_detection.ipynb
parkinsons.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn seaborn
```

## 📊 Dataset
- **Source:** [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Records:** 195 voice recordings
- **Target:** `status` (0 = Healthy, 1 = Parkinson's)
- **No missing values** ✅
- 80/20 train-test split

Features are **22 biomedical voice measures** including:

| Feature Group | Examples |
|---|---|
| Frequency measures | `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` |
| Jitter (frequency variation) | `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP` |
| Shimmer (amplitude variation) | `MDVP:Shimmer`, `Shimmer:APQ3`, `Shimmer:APQ5` |
| Noise ratios | `NHR`, `HNR` |
| Nonlinear dynamics | `RPDE`, `DFA`, `D2` |
| Signal fractals | `spread1`, `spread2`, `PPE` |

> `name` column is dropped before training.

## ⚙️ Preprocessing
```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)   # ✅ uses transform only — correct usage
```

## 🤖 Model
```python
model = svm.SVC()   # Default RBF kernel
model.fit(X_train, y_train)
```

## 📈 Evaluation
- Metric: **Accuracy Score** on test set

## 🔍 Prediction System
```python
input_data = (116.676, 137.871, 111.366, ...)  # 22 voice features
input_data = np.asarray(input_data).reshape(1, -1)
input_data = scaler.transform(input_data)      # scale before predict
prediction = model.predict(input_data)
# 0 → No Parkinson's | 1 → Has Parkinson's
```

## 🔮 Future Work
- Tune SVM kernel and `C`/`gamma` via GridSearchCV
- Address class imbalance (more Parkinson's samples than healthy in dataset)
- Add cross-validation for more robust accuracy estimate
- Try ensemble models (Random Forest, XGBoost) for comparison
