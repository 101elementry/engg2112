
import torch
import kagglehub

# Download latest version
path = kagglehub.dataset_download("karmansinghbains/ttc-delays-and-routes-2023")

print("Path to dataset files:", path)