# 🪨 Rock vs Mine Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn)
![Colab](https://img.shields.io/badge/Colab-Notebook-orange?style=for-the-badge&logo=colab)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 Overview

This project implements a **binary classification machine learning model** to predict whether a sonar signal is bouncing off a **rock** or an **underwater mine (metal cylinder)**. The model is trained on the famous **SONAR dataset**, which contains sonar readings captured at various angles and intensities.

This is a classic problem in the domain of **signal processing and defense applications**, where accurately distinguishing between rocks and mines can be a matter of safety. The project leverages **Logistic Regression** to build an efficient and interpretable classifier.

---

## 🎯 Problem Statement

Underwater mines pose a serious threat to naval operations. Sonar systems emit sound pulses and analyze their echoes to detect objects. However, distinguishing between a mine (metal cylinder) and a natural rock using sonar signals is challenging.

**Goal:** Given 60 sonar frequency-band features, classify the object as:
- `R` → **Rock**
- `M` → **Mine**

---

## 📁 Project Structure

```
Rock_Mine_prediction/
│
├── Copy of sonar data.csv          # Dataset with 208 sonar readings (60 features + label)
├── Rock_vs_Mines_Predictions.ipynb # Main Jupyter Notebook with complete ML pipeline
└── README.md                       # Project documentation (this file)
```

---

## 📊 Dataset Description

The dataset used is the **UCI SONAR Dataset**, originally introduced by R. Paul Gorman and Terrence J. Sejnowski.

| Property         | Details                          |
|------------------|----------------------------------|
| **File**         | `Copy of sonar data.csv`         |
| **Rows**         | 208 samples                      |
| **Columns**      | 61 (60 features + 1 label)       |
| **Feature Type** | Continuous (float values 0–1)    |
| **Target Labels**| `R` (Rock), `M` (Mine)           |
| **Class Balance**| ~111 Mines, ~97 Rocks            |

### Feature Description

Each of the 60 features represents the **energy in a particular frequency band** of the sonar return signal, integrated over a certain period of time. The values range from 0.0 to 1.0, capturing the intensity of different sonar echo patterns.

| Column         | Description                                          |
|----------------|------------------------------------------------------|
| `Feature_1` to `Feature_60` | Normalized sonar frequency-band energy readings |
| `Label`        | `R` = Rock, `M` = Mine                              |

---

## 🧪 Machine Learning Pipeline

The project follows a structured end-to-end ML pipeline:

### 1. 📥 Data Collection & Loading
- The sonar CSV dataset is loaded using **Pandas**.
- Initial exploration is done to understand the shape, data types, and class distribution.

### 2. 🔍 Exploratory Data Analysis (EDA)
- Checked dataset shape: `(208, 61)`
- Reviewed statistical summary using `.describe()`
- Examined label distribution with `.value_counts()`
- Verified there are **no missing values**

### 3. 🧹 Data Preprocessing
- Features (X) and labels (Y) are separated.
- The label column is dropped from the feature matrix.
- Labels are kept as categorical strings (`R` / `M`).

### 4. ✂️ Train-Test Split
- Data split using **`train_test_split`** from Scikit-Learn.
- **Test size:** 10% of the data
- **Stratified splitting** ensures balanced class distribution in both sets.
- **Random state** fixed for reproducibility.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)
```

### 5. 🤖 Model Training — Logistic Regression
- **Logistic Regression** is chosen as the classifier due to its effectiveness on binary classification tasks with continuous features.
- The model is trained on the training set.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)
```

### 6. 📈 Model Evaluation
- Accuracy is computed on both **training** and **test** data.
- Uses **`accuracy_score`** from Scikit-Learn.

| Dataset     | Accuracy     |
|-------------|--------------|
| Training    | ~86.4%       |
| Test        | ~71.2%       |

> These numbers are approximate and depend on the train-test split. The model performs reasonably well on unseen data without overfitting.

### 7. 🔮 Predictive System
- A real-world-style prediction system is implemented.
- A single input (60 sonar readings) is reshaped and fed into the trained model.
- The model outputs either `"The object is a Rock"` or `"The object is a Mine"`.

```python
input_data = (0.0200, 0.0371, 0.0428, ...)  # 60 values

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")
```

---

## 🛠️ Technologies Used

| Technology       | Purpose                                    |
|------------------|--------------------------------------------|
| **Python 3.8+**  | Core programming language                  |
| **NumPy**        | Numerical computations and array handling  |
| **Pandas**       | Data loading, manipulation, and EDA        |
| **Scikit-Learn** | ML model, train-test split, evaluation     |
| **Colab Notebook** | Interactive code development and visualization |

---

## ⚙️ Installation & Setup

### Prerequisites
Make sure you have Python 3.8 or higher installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/sumran58/Rock_Mine_prediction.git
cd Rock_Mine_prediction
```



## 🚀 How to Run

1. Open the Colab Notebook: `Rock_vs_Mines_Predictions.ipynb`
2. Run each cell sequentially (Cell → Run All)
3. Observe the model training, evaluation, and prediction output
4. To test with custom sonar data, replace the `input_data` tuple in the **Predictive System** section with your own 60-feature values

---

## 📉 Model Performance

| Metric              | Value         |
|---------------------|---------------|
| Algorithm           | Logistic Regression |
| Training Accuracy   | ~83.4%        |
| Testing Accuracy    | ~71.2%        |
| Features Used       | 60            |
| Dataset Size        | 208 samples   |

The model achieves solid accuracy considering the small dataset size and the inherent difficulty of sonar-based classification. Logistic Regression provides a good balance of simplicity and performance for this problem.

---

## 🔬 Why Logistic Regression?

- **Interpretability:** Simple and easy to understand model decisions.
- **Efficiency:** Fast to train and predict, even on CPU.
- **Binary Classification:** Naturally designed for two-class problems.
- **Baseline Model:** Serves as an excellent starting point before trying more complex models.

### Potential Improvements

Future work could include:
- Trying **SVM, Random Forest, or XGBoost** for potentially higher accuracy
- Applying **feature scaling** (StandardScaler) for better Logistic Regression performance
- Using **cross-validation** for more reliable evaluation
- Implementing **confusion matrix and classification report** for deeper insights
- Deploying the model via **Flask/Streamlit** as a web application

---

## 📚 References

- [SONAR Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))
- Gorman, R.P., and Sejnowski, T.J. (1988). *Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets.* Neural Networks, 1, 75–89.
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)

---

## 👨‍💻 Author

**Sumran Harchirkar**

- GitHub: [@sumran58](https://github.com/sumran58)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

## ⭐ Show Your Support

If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub. It motivates further development and sharing of open-source ML projects!

---

*Built with ❤️ using Python and Scikit-Learn*
