
import torch

import kagglehub

# Download latest version
path = kagglehub.dataset_download("khushikyad001/public-transport-delays-with-weather-and-events")

print("Path to dataset files:", path)