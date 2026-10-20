# Insurance Customer Segmentation using K-Means
# Dataset: insurance.csv

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load dataset (keep insurance.csv in the same folder as this script)
df = pd.read_csv("insurance.csv")

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# 2. Handle missing values, if any
df = df.dropna().copy()

# 3. Encode categorical columns
encoded_df = df.copy()
for column in ["sex", "smoker", "region"]:
    encoded_df[column] = LabelEncoder().fit_transform(
        encoded_df[column].astype(str)
    )

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(encoded_df)

# 5. Elbow Method
inertias = []
k_values = range(2, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150)
plt.show()

# 6. Fit K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 7. Evaluate clustering with Silhouette Score
score = silhouette_score(X_scaled, df["Cluster"])
print(f"\nSilhouette Score (4 clusters): {score:.4f}")

# 8. Show cluster sizes and summaries
print("\nNumber of customers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nAverage numeric values by cluster:")
print(df.groupby("Cluster")[["age", "bmi", "children", "charges"]].mean().round(2))

# 9. PCA visualization (2D)
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    components[:, 0],
    components[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.7
)
plt.title("Insurance Customer Segmentation (K-Means + PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.savefig("customer_clusters_pca.png", dpi=150)
plt.show()

# 10. Save clustered dataset
df.to_csv("insurance_with_clusters.csv", index=False)
print("\nSaved: insurance_with_clusters.csv")
print("Saved: elbow_method.png and customer_clusters_pca.png")
