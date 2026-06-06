# 🍷 Wine Quality Prediction using Random Forest Classifier

A machine learning project that predicts whether a red wine is of **good or bad quality** based on its physicochemical properties. The model uses a **Random Forest Classifier** trained on the **Red Wine Quality Dataset**, with quality binarized into two classes for simplified classification.

---

## 📌 Project Overview

This project demonstrates a complete end-to-end ML pipeline for binary classification:

- Loading and exploring the red wine dataset
- Checking for missing values and generating statistical summaries
- Visualizing feature relationships and correlations with quality
- Binarizing the target variable (quality score → good/bad)
- Training a Random Forest Classifier
- Evaluating model accuracy on test data
- Building a prediction system for new wine samples

---

## 📁 Project Structure

```
wine_quality_prediction.ipynb    # Main Jupyter Notebook (all code)
winequality-red.csv              # Dataset file
README.md                        # Project documentation
```

---

## 🛠️ Requirements

### Python Version
- Python 3.7+

### Libraries

Install all dependencies with:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Loading and exploring the dataset |
| `numpy` | Array operations and data reshaping |
| `seaborn` | Data visualization (bar plots, heatmap, count plots) |
| `matplotlib` | Figure rendering and plot configuration |
| `scikit-learn` | Model training, splitting, and evaluation |

---

## 📊 Dataset

- **File:** `winequality-red.csv`
- **Source:** [UCI Machine Learning Repository — Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Type:** Red Wine only
- **Total Records:** 1,599 wines
- **Missing Values:** None ✅
- **Original Target:** Quality score from 0–10 (integer)
- **Binarized Target:** 0 = Bad Quality, 1 = Good Quality

### Feature Description

| Feature | Description |
|---|---|
| `fixed acidity` | Tartaric acid content (non-volatile acids) |
| `volatile acidity` | Acetic acid content — high values lead to vinegar taste |
| `citric acid` | Adds freshness and flavor; positively correlated with quality |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt content in wine |
| `free sulfur dioxide` | Free form of SO₂ — prevents microbial growth |
| `total sulfur dioxide` | Total SO₂ (free + bound) |
| `density` | Density of wine (related to sugar and alcohol content) |
| `pH` | Acidity level (scale of 0–14) |
| `sulphates` | Additive contributing to SO₂ levels |
| `alcohol` | Percentage of alcohol content |

---

## 📉 Exploratory Data Analysis (EDA)

### 1. Quality Distribution
A count plot is used to visualize how many wines fall into each quality score, showing the most common ratings in the dataset.

### 2. Volatile Acidity vs Quality
```
Higher volatile acidity → Lower wine quality
Lower volatile acidity  → Higher wine quality
```
> Volatile acidity and quality are **inversely proportional**.

### 3. Citric Acid vs Quality
Citric acid is **positively correlated** with quality — wines with more citric acid tend to be rated higher.

### 4. Correlation Heatmap
A full correlation matrix is plotted as an annotated heatmap (10×10) using the `Blues` colormap to reveal relationships between all features.

---

## 🏷️ Label Binarization

The original quality scores (ranging from 3 to 8 in this dataset) are converted into a binary label:

```python
y = df['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
```

| Condition | Label | Meaning |
|---|---|---|
| Quality ≥ 7 | `1` | Good Quality 🍷 |
| Quality < 7 | `0` | Bad Quality ❌ |

---

## ⚙️ Data Preprocessing

### Feature / Label Split
```python
X = df.drop('quality', axis=1)   # 11 physicochemical features
y = df['quality'].apply(...)      # Binarized labels
```

### Train-Test Split
- **Test size:** 20%
- **Random state:** 2
- **Stratified split** to preserve class ratio in both sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2, stratify=y
)
```

> No feature scaling is applied — Random Forests are **scale-invariant** and do not require normalization.

---

## 🤖 Model

### Algorithm: Random Forest Classifier

An ensemble of decision trees trained on random subsets of the data and features, combining predictions via majority voting.

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

**Why Random Forest?**
- Handles non-linear relationships well
- Robust to outliers and noisy data
- No need for feature scaling
- Naturally handles feature importance ranking
- Reduces overfitting compared to a single decision tree (bagging)

---

## 📈 Model Evaluation

```python
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, y_test)
print(test_accuracy)
```

Accuracy is measured on the held-out 20% test set.

---

## 🔍 Prediction System

The notebook includes a prediction system for classifying a new wine sample.

### Example Input

```python
input_data = (4.3, 0.65, 4.00, 1.2, 0.065, 16.0, 21.0, 0.9946, 4.39, 0.47, 10.0)
# (fixed acidity, volatile acidity, citric acid, residual sugar,
#  chlorides, free sulfur dioxide, total sulfur dioxide,
#  density, pH, sulphates, alcohol)
```

### Prediction Pipeline

```python
# Step 1: Convert tuple to NumPy array
input_data = np.asarray(input_data)

# Step 2: Reshape for single instance prediction
input_data = input_data.reshape(1, -1)

# Step 3: Predict
prediction = model.predict(input_data)

if prediction == 0:
    print("Bad Quality")
else:
    print("Good Quality")
```

> **Output → 0:** Bad Quality ❌
> **Output → 1:** Good Quality 🍷

---

## ▶️ How to Run

1. Clone or download this repository
2. Place `winequality-red.csv` in the working directory (or upload to `/content/` in Google Colab)
3. Open `wine_quality_prediction.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells from top to bottom

---

## ⚠️ Known Limitations

- **Binary Classification Only:** The model predicts good vs. bad but loses granularity of individual quality scores (e.g., 5 vs. 6 treated identically as "bad")
- **Red Wine Only:** The model is trained exclusively on red wine data and should not be used to predict white wine quality
- **Class Imbalance:** Most wines fall in the mid-range quality (5–6), making "good quality" (≥7) wines a minority class — this can affect the model's recall for the good class
- **No Hyperparameter Tuning:** The Random Forest uses default parameters; tuning `n_estimators`, `max_depth`, etc. could improve performance

---

## 🔮 Future Work

- Tune hyperparameters using `GridSearchCV` or `RandomizedSearchCV`
- Address class imbalance using SMOTE or `class_weight='balanced'`
- Try multi-class classification to preserve individual quality scores
- Add feature importance visualization to understand top predictors
- Extend to white wine dataset and build a combined model
- Deploy as an interactive web app using Streamlit or Flask
- Save the trained model with `joblib` for production use

---

## 📄 License

This project is for educational purposes. Dataset is publicly available via the UCI Machine Learning Repository.
