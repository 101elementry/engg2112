import pandas as pd
import os

filePath = os.path.dirname(__file__)
dataFolder = os.path.abspath(os.path.join(filePath, '..', 'data'))

inputFile = os.path.join(dataFolder, 'nsw_road_crash_data_merged.csv')

df = pd.read_csv(inputFile, encoding='latin1')

dfCleaned = df.drop_duplicates(subset=['Crash ID'])

outputFile = os.path.join(dataFolder, 'nsw_road_crash_data_merged_removed_duplicates.csv')

dfCleaned.to_csv(outputFile, index=False)