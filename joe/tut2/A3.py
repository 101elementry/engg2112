import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
plt.style.use('seaborn-v0_8-whitegrid')

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(df.head())

petal_lengths = df['petal length (cm)'].to_numpy()

print('Mean: ', np.mean(petal_lengths))
print('Median: ', np.median(petal_lengths))
print('Std: ', np.std(petal_lengths))
print('Min/Max: ', np.min(petal_lengths), np.max(petal_lengths))

petal_length_by_species = df.groupby('species')['petal length (cm)'].mean()
print(petal_length_by_species)

norm = (petal_lengths-petal_lengths.min())/(petal_lengths.max()-petal_lengths.min())
print("normalised mean: ", norm)

print("percentiles: ", np.percentile(petal_lengths, [25,50,75]))
print("conparison: ", df.describe())