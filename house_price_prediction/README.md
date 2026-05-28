# 🏠 House Price Prediction

A machine learning project that predicts California housing prices using **XGBoost** regression on the classic California Housing dataset.

---

## 📋 Overview

This project trains an XGBoost regressor to predict the median value of houses in California districts based on features such as median income, house age, average number of rooms, population, and geographic location.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | `sklearn.datasets.fetch_california_housing` |
| **Samples** | 20,640 |
| **Features** | 8 |
| **Target** | `price` — median house value (in $100,000s) |

**Features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## 🔁 Workflow

1. **Load Data** — fetch the dataset into a pandas DataFrame and append the `price` target column.
2. **Explore** — inspect shape, check for missing values, and review summary statistics.
3. **Correlation Analysis** — visualize feature relationships with a Seaborn heatmap.
4. **Train/Test Split** — 80% training / 20% testing (`random_state=2`).
5. **Train Model** — fit an `XGBRegressor` on the training set.
6. **Evaluate** — measure R² and Mean Absolute Error on both train and test sets.
7. **Visualize** — scatter plot of actual vs. predicted prices.

---

## 🤖 Model

**`XGBRegressor`** — a gradient-boosting ensemble that builds multiple decision trees sequentially, each correcting the errors of the previous one.

**Evaluation Metrics:**
- **R² Score** — proportion of variance in price explained by the model.
- **Mean Absolute Error (MAE)** — average absolute difference between predicted and actual prices.

---

## 🚀 Usage

```bash
jupyter notebook house_price_prediction.ipynb
```

Run the cells top to bottom.

---

## 📁 Project Structure

```
.
├── house_price_prediction.ipynb   # Main notebook
└── README.md
```

---

## 📝 Notes

- The `accuracy_score` import is unused — it's a classification metric and doesn't apply to regression.
- One cell references `training_data_pred` before it's defined. Run the prediction cell **before** the print cell to avoid a `NameError`.

---

## 📄 License

Free to use for learning and personal projects.
