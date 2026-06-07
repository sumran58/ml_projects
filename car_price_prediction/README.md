# 🚗 Car Price Prediction using Linear Regression & Lasso

Predicts the **selling price of used cars** using **Linear Regression** and **Lasso Regression** on data from CarDekho.

---

## 📁 Files
```
car_price_prediction.ipynb
CAR DETAILS FROM CAR DEKHO.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 📊 Dataset
- **Source:** CarDekho used car listings
- **Target:** `selling_price` (in INR)
- **No missing values** ✅
- 90/10 train-test split

| Feature | Description |
|---|---|
| `name` | Car model name (**dropped** before training) |
| `year` | Year of manufacture |
| `km_driven` | Total kilometers driven |
| `fuel` | Fuel type (encoded) |
| `seller_type` | Individual / Dealer / Trustmark (encoded) |
| `transmission` | Manual / Automatic (encoded) |
| `owner` | Ownership history (encoded) |

## ⚙️ Preprocessing — Label Encoding

Categorical columns are encoded numerically:

| Column | Encoding |
|---|---|
| `fuel` | Petrol=0, Diesel=1, CNG=2, LPG=3, Electric=4 |
| `seller_type` | Individual=0, Dealer=1, Trustmark Dealer=2 |
| `transmission` | Manual=0, Automatic=1 |
| `owner` | First=0, Second=1, Third=2, Fourth+=3, Test Drive=4 |

> `name` column is dropped as it has too many unique string values.

## 🤖 Models

### Model 1 — Linear Regression
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Model 2 — Lasso Regression
```python
model1 = Lasso()
model1.fit(X_train, y_train)
```
Lasso adds L1 regularization — helps reduce overfitting and performs implicit feature selection.

## 📈 Evaluation
- Metric: **R² Score** on both train and test sets
```python
metrics.r2_score(y_train, training_data_prediction)   # Train R²
metrics.r2_score(y_test, testing_data_prediction)     # Test R²
metrics.r2_score(y_test, test_data_pred)              # Lasso Test R²
```

## 🔍 Prediction System
```python
input_data = (2007, 50000, 0, 0, 0, 0)
# (year, km_driven, fuel, seller_type, transmission, owner)
input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)
print(prediction)  # Predicted selling price in INR
```

## ⚠️ Known Limitations
- No feature scaling applied — Linear Regression benefits from `StandardScaler`
- Simple label encoding may not capture ordinal meaning correctly for all features
- `name` column discarded entirely — brand/model info could improve predictions

## 🔮 Future Work
- Add `StandardScaler` for Linear Regression
- Try Random Forest or XGBoost Regressor for better accuracy
- Use `OneHotEncoding` for `fuel` and `seller_type`
- Add cross-validation and hyperparameter tuning
- Deploy as a car price estimator web app with Streamlit
