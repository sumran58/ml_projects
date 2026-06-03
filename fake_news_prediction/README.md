# 📰 Fake News Prediction

A machine learning project that classifies news articles as **real** or **fake** using Natural Language Processing and Logistic Regression.

---

## 📋 Overview

This project uses NLP techniques (stemming, stopword removal, TF-IDF vectorization) to convert news article metadata into numerical features, then trains a Logistic Regression model to predict whether a given article is authentic or fabricated.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | `train.csv` (Kaggle Fake News dataset) |
| **Features used** | `author`, `title` (merged into `content`) |
| **Target** | `label` — `0` = real, `1` = fake |

The `author` and `title` columns are combined into a single `content` column, which serves as the input text for the model.

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

Also download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

---

## 🔁 Workflow

1. **Load Data** — read `train.csv` into a pandas DataFrame.
2. **Handle Missing Values** — fill nulls with empty strings.
3. **Feature Engineering** — merge `author` + `title` into a `content` column.
4. **Text Preprocessing** (`stemming` function):
   - Remove non-alphabetic characters
   - Convert to lowercase
   - Remove English stopwords
   - Apply Porter stemming
5. **Vectorization** — convert text into numerical features using **TF-IDF**.
6. **Train/Test Split** — 80% training / 20% testing (stratified, `random_state=2`).
7. **Train Model** — fit a `LogisticRegression` classifier.
8. **Evaluate** — measure accuracy on both train and test sets.
9. **Predict** — classify a single sample as real or fake.

---

## 🤖 Model

**`LogisticRegression`** — a linear classifier well-suited for high-dimensional sparse text features produced by TF-IDF.

**Evaluation Metric:** Accuracy Score (proportion of correctly classified articles).

---

## 🚀 Usage

```bash
jupyter notebook fake_real_news_prediction.ipynb
```

Place `train.csv` in the same directory (or update the file path) and run cells top to bottom.

---

## 📁 Project Structure

```
.
├── fake_real_news_prediction.ipynb   # Main notebook
├── train.csv                         # Dataset
└── README.md
```

---

## 📝 Notes

- The dataset path `/content/train.csv` is Colab-specific — update it to your local path if running elsewhere.
- Stemming over the full dataset can be slow; consider caching results for repeated runs.
- TF-IDF is fit on the **entire dataset before splitting**, which leaks vocabulary info from test to train. For stricter evaluation, fit only on `X_train`.

---

## 📄 License

Free to use for learning and personal projects.
