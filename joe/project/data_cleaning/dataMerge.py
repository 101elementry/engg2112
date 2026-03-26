import pandas as pd
import os

filePath = os.path.dirname(__file__)
dataFolder = os.path.abspath(os.path.join(filePath, '..', 'data'))

files = os.listdir(dataFolder)

csvFiles = []
for file in files:
    if file.endswith('.csv'):
        csvFiles.append(file)

dfs = []

for file in csvFiles:
    filePath = os.path.join(dataFolder, file)
    df = pd.read_csv(filePath, encoding = 'latin1')
    dfs.append(df)

mergedData = pd.concat(dfs)

outputFile = os.path.join(dataFolder, 'nsw_road_crash_data_merged.csv')
mergedData.to_csv(outputFile, index = False)