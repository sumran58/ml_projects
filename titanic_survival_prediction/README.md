# 🚢 Titanic Survival Prediction using Logistic Regression

Predicts whether a **Titanic passenger survived** based on demographic and travel data using **Logistic Regression**.

---

## 📁 Files
```
titanic_survival_prediction.ipynb
train.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 📊 Dataset
- **Source:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- **Target:** `Survived` (0 = No, 1 = Yes)
- 80/20 train-test split

| Feature | Handling |
|---|---|
| `Cabin` | **Dropped** (too many missing values) |
| `Age` | Missing → filled with **mean** |
| `Embarked` | Missing → filled with **mode** |
| `Name`, `PassengerId`, `Ticket` | **Dropped** (not useful) |

## 📉 EDA Insights
- More males than females on board, but **females had higher survival rate**
- **1st class passengers** had significantly better survival odds than 2nd/3rd class

## ⚙️ Encoding
```python
df.replace({'Sex': {'male': 0, 'female': 1},
            'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
```

## 🤖 Model
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

## 📈 Evaluation
- Metric: **Accuracy Score** on test set

## 🔮 Future Work
- Add confusion matrix and classification report
- Try Random Forest or Gradient Boosting for better accuracy
- Engineer features: title from `Name`, family size from `SibSp`+`Parch`
- Add prediction system for a new passenger input
