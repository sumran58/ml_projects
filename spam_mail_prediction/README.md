# 📧 Spam Mail Detection

A machine learning project that classifies emails as **spam** or **ham** (legitimate) using TF-IDF vectorization and Logistic Regression.

---

## 📋 Overview

This project trains a Logistic Regression classifier on a labeled email dataset to detect spam. Text messages are converted into numerical features using TF-IDF, then used to train a binary classifier.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | `mail_data.csv` (SMS Spam Collection / Mail Spam dataset) |
| **Columns** | `Category` (spam/ham), `Message` (email text) |
| **Encoding** | `spam` → 0, `ham` → 1 |

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🔁 Workflow

1. **Load Data** — read `mail_data.csv` into a pandas DataFrame.
2. **Explore** — inspect shape and column info.
3. **Handle Missing Values** — replace nulls with empty strings.
4. **Encode Labels** — `spam` → 0, `ham` → 1.
5. **Split Features** — `X` = message text, `Y` = category.
6. **Train/Test Split** — 80% training / 20% testing (`random_state=3`).
7. **Vectorize** — apply **TF-IDF** with English stopwords removed, lowercase, `min_df=1`.
8. **Train Model** — fit a `LogisticRegression` classifier.
9. **Evaluate** — accuracy on the test set.
10. **Predict** — classify a custom input message as spam or ham.

---

## 🤖 Model

**`LogisticRegression`** — a linear classifier well-suited for sparse, high-dimensional TF-IDF features.

**Evaluation Metric:** Accuracy Score.

---

## 🚀 Usage

```bash
jupyter notebook spam_mail_detection.ipynb
```

Place `mail_data.csv` in the same directory (or update the path) and run cells top to bottom.

**Example:**
```python
input_data = ["Congratulations! You've won a free iPhone. Click here to claim."]
# Output: spam mail
```

---

## 📁 Project Structure

```
.
├── spam_mail_detection.ipynb   # Main notebook
├── mail_data.csv               # Dataset
└── README.md
```

---

## 📝 Notes

- The dataset path `/content/mail_data.csv` is Colab-specific — update for local runs.
- Label encoding is inverted from intuition: `spam = 0`, `ham = 1`. Keep this in mind when interpreting predictions.
- The TF-IDF vectorizer is fit only on `X_train` (good practice — no data leakage).
- Accuracy alone can be misleading for imbalanced datasets — consider precision, recall, and F1-score for spam detection.

---

## 📄 License

Free to use for learning and personal projects.
