import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

# Reproducibility seed
SEED = 42

# Load dataset
df = pd.read_csv(
    "/Users/connorwherry/Library/Mobile Documents/com~apple~CloudDocs/Study/Sem1Yr3/ENGG2112/engg2112/joe/project/data/nsw_road_crash_data_merged_removed_duplicates.csv",
    low_memory=False
)
# df = pd.read_csv()
# Remove rows with missing target
df = df.dropna(subset=['Degree of crash'])

# Target
y = df['Degree of crash']

# Features
X = df.drop(columns=[
    'Degree of crash',
    'Degree of crash - detailed',
    'Crash ID',
    'No. killed',
    'No. seriously injured',
    'No. moderately injured',
    'No. minor-other injured'
])

# Encode categorical variables
X = pd.get_dummies(X)

# Handle missing values in features
X = X.fillna(0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Train model
dt_default = DecisionTreeClassifier(random_state=SEED)
dt_default.fit(X_train, y_train)

# Predict
y_pred_default = dt_default.predict(X_test)

# Results
print(f'Accuracy: {accuracy_score(y_test, y_pred_default):.4f}')
print(f'Max depth: {dt_default.get_depth()}')