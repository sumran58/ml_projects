# 🩺 Diabetes Prediction Web App

A complete end-to-end machine learning project that predicts whether a person is **diabetic** or **non-diabetic** based on medical parameters — from model training in Jupyter to a deployed Streamlit web interface.

---

## 📋 Overview

This project trains a Support Vector Machine (SVM) classifier on the **PIMA Indians Diabetes Dataset**, saves the trained model as a `.sav` file using `pickle`, and serves predictions through two interfaces:

- **`app.py`** — a simple command-line predictor
- **`web.py`** — an interactive Streamlit web app

---

## 📊 Dataset

| | |
|---|---|
| **Source** | `diabetes.csv` (PIMA Indians Diabetes Dataset) |
| **Samples** | 768 |
| **Target** | `Outcome` — `0` = non-diabetic, `1` = diabetic |

**Features (8):** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

---

## ⚙️ Requirements

```bash
pip install numpy pandas scikit-learn streamlit
```

---

## 🔁 Workflow

### Training (`Diabetes_prediction__1_.ipynb`)

1. **Load Data** — read `diabetes.csv` into a pandas DataFrame.
2. **Explore** — inspect shape, summary stats, and class balance.
3. **Standardize** — apply `StandardScaler` to normalize features.
4. **Train/Test Split** — 80% train / 20% test (stratified, `random_state=2`).
5. **Train Model** — fit `SVC(kernel='linear')`.
6. **Evaluate** — accuracy on train and test sets.
7. **Save Model** — pickle the trained classifier to `trained_model.sav`.

### Inference

- **`app.py`** — loads the saved model and predicts on a hardcoded sample.
- **`web.py`** — Streamlit form that accepts 8 inputs and displays the diagnosis.

---

## 🤖 Model

**`SVC(kernel='linear')`** — a linear Support Vector Machine, effective for binary classification on small tabular datasets.

**Evaluation Metric:** Accuracy Score.

---

## 🚀 Usage

### 1. Train the model

```bash
jupyter notebook Diabetes_prediction__1_.ipynb
```

Run all cells — this generates `trained_model.sav`.

### 2. CLI prediction

```bash
python app.py
```

### 3. Streamlit web app

```bash
streamlit run web.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`), enter the 8 medical values, and click **"Diabetes test result"**.

---

## 📁 Project Structure

```
.
├── Diabetes_prediction__1_.ipynb   # Training notebook
├── trained_model.sav               # Pickled SVM model
├── app.py                          # CLI predictor
├── web.py                          # Streamlit web app
├── diabetes.csv                    # Dataset
└── README.md
```

---

## 📝 Notes

- **Scaler not saved.** The notebook trains the SVM on scaled data, but `StandardScaler` isn't pickled alongside the model. `app.py` and `web.py` pass **raw** values to the model, which will hurt prediction accuracy. Fix: pickle the scaler too and apply it before `predict()`.
- **Streamlit inputs are strings.** `st.text_input` returns strings; cast them to `float` before passing to the model, otherwise the model call will fail.
- **Dataset is mildly imbalanced** (~65% non-diabetic, ~35% diabetic). Accuracy is okay as a metric, but precision/recall/F1 give a fuller picture.
- **`scaler.fit_transform` on a single input** (used in one notebook cell) is incorrect — it refits the scaler on one row. Use the saved scaler's `.transform()` instead.

---

## 📄 License

Free to use for learning and personal projects.
