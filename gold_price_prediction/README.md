# 🥇 Gold Price Prediction using Random Forest Regressor

Predicts the **price of gold (GLD ETF)** based on other financial market indicators using a **Random Forest Regressor**.

---

## 📁 Files
```
gold_price_prediction.ipynb
gld_price_data.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 📊 Dataset
- **Source:** Historical gold ETF and financial market data
- **Target:** `GLD` (Gold ETF price)
- **No missing values** ✅
- `Date` column parsed and dropped before training

| Feature | Description |
|---|---|
| `SPX` | S&P 500 Index price |
| `GLD` | Gold ETF price (**target**) |
| `USO` | US Oil Fund ETF price |
| `SLV` | Silver ETF price |
| `EUR/USD` | Euro to US Dollar exchange rate |

## 📉 EDA
- **Correlation heatmap** to identify feature relationships with GLD
- **Distribution plot** of GLD prices
- Gold price correlations printed directly: `df.corr()['GLD']`

## 🤖 Model
```python
model = RandomForestRegressor()
model.fit(X_train, y_train)
```
- Ensemble of decision trees for regression
- No feature scaling required
- 80/20 train-test split (no stratify — regression task)

## 📈 Evaluation
- Metric: **R² Score** (coefficient of determination)
```python
metrics.r2_score(y_test, test_pred)
```
> R² close to 1.0 indicates strong predictive performance.

## ⚠️ Notes
- `Date` column is parsed to datetime but dropped before training
- No prediction system built in this notebook — only model evaluation

## 🔮 Future Work
- Add a prediction system for new input values
- Try feature engineering from the `Date` column (month, year, day-of-week)
- Compare with XGBoost or LSTM for time-series forecasting
- Tune `n_estimators` and `max_depth` via GridSearchCV
