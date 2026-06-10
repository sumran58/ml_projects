
# 🎬 Movie Recommendation System

A content-based movie recommendation system that suggests similar movies using **TF-IDF vectorization** and **cosine similarity** on movie metadata.

---

## 📋 Overview

This project recommends movies similar to a user's favorite by analyzing the combined textual content of genres, keywords, tagline, cast, and director. It uses TF-IDF to convert text into vectors and cosine similarity to rank the closest matches.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | `movies.csv` (TMDB 5000 Movies dataset) |
| **Features used** | `genres`, `keywords`, `tagline`, `cast`, `director` |

These five fields are merged into a single text feature representing each movie's "content fingerprint."

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🔁 Workflow

1. **Load Data** — read `movies.csv` into a pandas DataFrame.
2. **Explore** — check shape, info, and missing values.
3. **Select Features** — pick `genres`, `keywords`, `tagline`, `cast`, `director`.
4. **Handle Missing Values** — fill nulls with empty strings.
5. **Combine Features** — concatenate the five fields into one text column.
6. **Vectorize** — apply **TF-IDF** to convert text into numerical feature vectors.
7. **Compute Similarity** — calculate the cosine similarity matrix across all movies.
8. **Match User Input** — use `difflib.get_close_matches` for fuzzy matching of the user's typed title.
9. **Rank & Recommend** — sort movies by similarity score and return the top 29 closest matches.

---

## 🤖 Approach

**Content-Based Filtering** — recommends movies by comparing the textual metadata of titles, not user ratings.

- **TfidfVectorizer** — converts combined text into TF-IDF weighted vectors.
- **Cosine Similarity** — measures the angle between vectors; higher = more similar.

---

## 🚀 Usage

```bash
jupyter notebook movie_recommendation_system.ipynb
```

Place `movies.csv` in the same directory (or update the path), run all cells, and enter a favorite movie name when prompted.

**Example:**
```
enter your favourite movie name: Iron Man

Movie suggested for you:
1. Iron Man 2
2. Iron Man 3
3. Avengers: Age of Ultron
...
```

---

## 📁 Project Structure

```
.
├── movie_recommendation_system.ipynb   # Main notebook
├── movies.csv                          # Dataset
└── README.md
```

---

## 📝 Notes

- The dataset path `/content/movies.csv` is Colab-specific — update for local runs.
- Combined features use `''` (empty separator) instead of `' '` (space), which can fuse trailing/leading tokens. Replacing with `' '` improves matching slightly.
- Recommendations are based purely on metadata — not user preferences or ratings.

---

## 📄 License

Free to use for learning and personal projects.
