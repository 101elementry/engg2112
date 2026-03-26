import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
df.head()

df["quality_band"] = pd.cut(df["alcohol"], bins=[0, 12, 13.5, 20], labels=["low", "medium", "high"])
df["has_high_malic"] = np.where(df["malic_acid"] > df["malic_acid"].median(), "yes", "no")
df.dtypes

df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

df_missing = df.copy()
rng = np.random.default_rng(42)
for col in ["alcohol", "magnesium", "color_intensity"]:
    idx = rng.choice(df_missing.index, size=12, replace=False)
    df_missing.loc[idx, col] = np.nan
df_missing.isnull().sum()


mean_imputer = SimpleImputer(strategy="mean")
median_imputer = SimpleImputer(strategy="median")
df_mean = df_missing.copy()
df_median = df_missing.copy()
num_cols = df_missing.select_dtypes(include=np.number).columns
df_mean[num_cols] = mean_imputer.fit_transform(df_missing[num_cols])
df_median[num_cols] = median_imputer.fit_transform(df_missing[num_cols])


cat_cols = ["quality_band", "has_high_malic"]
num_cols = [c for c in df_missing.columns if c not in cat_cols + ["target"]]
preprocessor = ColumnTransformer([("num", SimpleImputer(strategy="median"), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

X = df[num_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = []
k_values = range(2, 8)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))
list(zip(k_values, scores))

plt.plot(list(k_values), scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.show()

X = df.drop(columns=["target", "quality_band", "has_high_malic"], errors="ignore")
y = df["target"]
for test_size in [0.2, 0.3, 0.4]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    print(test_size, X_train.shape, X_test.shape)