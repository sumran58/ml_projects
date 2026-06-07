# ❤️ Heart Disease Prediction using Logistic Regression

Predicts whether a patient has heart disease using **Logistic Regression** on clinical health data.

---

## 📁 Files
```
heart_disease_prediction.ipynb
heart_disease_data.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn seaborn
```

## 📊 Dataset
- **Features:** 13 clinical attributes
- **Target:** `1` = Heart Disease, `0` = No Heart Disease
- **No missing values** ✅
- Stratified 80/20 train-test split

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Gender (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia type |

## 🤖 Model
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```
- Simple, interpretable binary classifier
- No feature scaling applied (consider adding `StandardScaler` for better performance)

## 📈 Evaluation
- Metric: **Accuracy Score** on test set

## 🔍 Prediction System
```python
input_data = (44, 1, 1, 130, 219, 0, 0, 188, 0, 0, 2, 0, 2)
input_data = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data)
# 1 → Has Heart Disease | 0 → No Heart Disease
```

## 🔮 Future Work
- Add `StandardScaler` for improved Logistic Regression performance
- Try SVM, Random Forest, or XGBoost for comparison
- Add confusion matrix and classification report
- Deploy with Streamlit
