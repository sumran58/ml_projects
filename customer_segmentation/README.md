
# 🛍️ Customer Segmentation using K-Means Clustering

Groups mall customers into segments based on **Annual Income** and **Spending Score** using **K-Means Clustering**.

---

## 📁 Files
```
customer_segmentation.ipynb
Mall_Customers.csv
```

## 🛠️ Requirements
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 📊 Dataset
- **Source:** Mall Customers dataset
- **Features used:** `Annual Income (k$)`, `Spending Score (1–100)`
- **No missing values** ✅
- Only columns 3 & 4 are used: `X = df.iloc[:, [3, 4]].values`

## ⚙️ Approach

### 1. Elbow Method (Optimal Clusters)
WCSS (Within-Cluster Sum of Squares) is computed for k = 1 to 10 and plotted to find the elbow point → **k = 5** selected.

```python
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

### 2. Final Model
```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)
```

## 📈 Cluster Visualization
5 clusters plotted with distinct colors + yellow centroids on a scatter plot of Income vs Spending Score.

| Cluster | Likely Segment |
|---|---|
| 1 | Low income, Low spending |
| 2 | Low income, High spending |
| 3 | Medium income, Medium spending |
| 4 | High income, Low spending |
| 5 | High income, High spending |

## 🔮 Future Work
- Use all features (age, gender) with dimensionality reduction (PCA)
- Try DBSCAN or Hierarchical Clustering
- Build a customer profile dashboard with Streamlit
