import pandas as pd
print(pd)
df = pd.read_csv("Datasets/nsw_road_crash_data_2016-2020_crash.csv")

# 1. Remove completely empty rows
df = df.dropna(how="all")

# 2. Remove rows with ANY missing values
df = df.dropna()

# 3. Remove columns that are entirely empty
df = df.dropna(axis=1, how="all")

print(df.shape)