# ===============================
# Agglomerative Hierarchical Clustering
# Wine Quality Dataset (Red + White)
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------
# 1. Load Datasets
# -------------------------------
red = pd.read_csv("../admin1/Downloads/wine+quality/winequality-red.csv", sep=';')
white = pd.read_csv("../admin1/Downloads/wine+quality/winequality-white.csv", sep=';')



# -------------------------------
# 2. Concatenate Datasets
# -------------------------------
df = pd.concat([red, white], axis=0, ignore_index=True)

print("Combined Dataset Shape:", df.shape)

# -------------------------------
# 3. Prepare Data for Clustering
# -------------------------------
# Remove target and non-numeric column
X = df.drop(['quality'], axis=1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. Create Linkage Matrix
# -------------------------------
linked = linkage(X_scaled, method='ward')

# -------------------------------
# 5. Plot Dendrogram
# -------------------------------
plt.figure(figsize=(14, 6))
dendrogram(
    linked,
    truncate_mode='level',   # makes dendrogram readable
    p=5
)
plt.title("Dendrogram of Combined Red & White Wine Dataset")
plt.xlabel("Wine Samples")
plt.ylabel("Euclidean Distance")
plt.show()
