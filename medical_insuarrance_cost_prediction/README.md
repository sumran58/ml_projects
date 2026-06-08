# 🏥 Medical Insurance Cost Prediction using Linear Regression

Predicts **individual medical insurance charges** based on personal health and demographic data using **Linear Regression**.

---

## 📁 Files
```
medical_insurance_cost_prediction.ipynb
insurance.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 📊 Dataset
- **Source:** Medical Cost Personal Dataset (Kaggle)
- **Records:** 1,338 individuals
- **Target:** `charges` (USD)
- **No missing values** ✅
- 80/20 train-test split

| Feature | Description |
|---|---|
| `age` | Age of the insured |
| `sex` | Gender (male=0, female=1) |
| `bmi` | Body Mass Index |
| `children` | Number of dependents covered |
| `smoker` | Smoking status (yes=0, no=1) |
| `region` | US region (southeast=0, southwest=1, northeast=2, northwest=3) |

## 📉 EDA
Distribution plots for `age`, `bmi`, `charges` and count plots for `sex`, `children`, `smoker`, `region`.

## ⚙️ Preprocessing — Label Encoding
```python
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
```

## 🤖 Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

## 📈 Evaluation
- Metric: **R² Score** on test set

## 🔍 Prediction System
```python
input_data = (33, 0, 22.705, 0, 1, 3)
# (age, sex, bmi, children, smoker, region)
prediction = model.predict(np.asarray(input_data).reshape(1, -1))
print(prediction)  # Predicted insurance charge in USD
```

## ⚠️ Notes
- Typo in notebook: variable named `imput_data` instead of `input_data` — fix before reuse
- `smoker` encoding (yes=0, no=1) is reversed from convention — won't break the model but may confuse interpretation

## 🔮 Future Work
- Try Ridge/Lasso to handle multicollinearity
- Add `StandardScaler` for better Linear Regression performance
- Use Random Forest or XGBoost for non-linear charge patterns
- Deploy as an insurance cost estimator with Streamlit
