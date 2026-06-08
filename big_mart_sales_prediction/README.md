
# 🛒 BigMart Sales Prediction using XGBoost

Predicts **outlet sales of products** across BigMart stores using an **XGBoost Regressor**.

---

## 📁 Files
```
big_mart_sales_prediction.ipynb
Train.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```

## 📊 Dataset
- **Source:** BigMart Sales dataset (Kaggle / Analytics Vidhya)
- **Target:** `Item_Outlet_Sales`
- **Missing values** in `Item_Weight` and `Outlet_Size` — handled ✅
- 80/20 train-test split

| Feature | Type | Description |
|---|---|---|
| `Item_Identifier` | Categorical | Unique product ID |
| `Item_Weight` | Numerical | Weight of product |
| `Item_Fat_Content` | Categorical | Low Fat / Regular |
| `Item_Visibility` | Numerical | Display area % in store |
| `Item_Type` | Categorical | Product category |
| `Item_MRP` | Numerical | Maximum retail price |
| `Outlet_Identifier` | Categorical | Store ID |
| `Outlet_Establishment_Year` | Numerical | Year store was opened |
| `Outlet_Size` | Categorical | Small / Medium / High |
| `Outlet_Location_Type` | Categorical | Tier 1/2/3 city |
| `Outlet_Type` | Categorical | Grocery / Supermarket type |

## ⚙️ Preprocessing

**Missing Value Handling:**
```python
# Item_Weight → filled with mean
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

# Outlet_Size → filled with mode per Outlet_Type
mode_of_outlet = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
```

**Fat Content Standardization:**
```python
df.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)
```

**Label Encoding:** All 7 categorical columns encoded with `LabelEncoder`.

## 🤖 Model
```python
model = XGBRegressor()
model.fit(X_train, y_train)
```

## 📈 Evaluation
- Metric: **R² Score** on test set

## 🔮 Future Work
- Tune XGBoost hyperparameters (`max_depth`, `learning_rate`, `n_estimators`)
- Try feature engineering (store age from establishment year)
- Use `OrdinalEncoder` instead of `LabelEncoder` for ordinal features
- Add SHAP values for feature importance explanation
