
# 🔥 Calories Burnt Prediction using XGBoost

Predicts **calories burnt during exercise** based on biometric and workout data using **XGBoost Regressor**.

---

## 📁 Files
```
Calories_burnt_prediction.ipynb
calories.csv
exercise.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```

## 📊 Dataset
- Two CSVs merged: `exercise.csv` (features) + `calories.csv` (target)
- **Target:** `Calories` (burnt during exercise)
- **No missing values** ✅
- 80/20 train-test split

| Feature | Description |
|---|---|
| `Gender` | male=0, female=1 |
| `Age` | Age in years |
| `Height` | Height in cm |
| `Weight` | Weight in kg |
| `Duration` | Exercise duration (mins) |
| `Heart_Rate` | Heart rate during exercise |
| `Body_Temp` | Body temperature during exercise |

> `User_ID` is dropped before training.

## ⚙️ Preprocessing
```python
df = pd.concat([ex, cal['Calories']], axis=1)
df.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)
```
Correlation heatmap plotted to understand feature relationships.

## 🤖 Model
```python
model = XGBRegressor()
model.fit(X_train, y_train)
```

## 📈 Evaluation
- Metric: **Mean Absolute Error (MAE)**

## 🔮 Future Work
- Add prediction system for new workout inputs
- Tune XGBoost hyperparameters
- Try feature scaling + Linear/Ridge Regression for comparison
- Deploy as a fitness tracker widget
