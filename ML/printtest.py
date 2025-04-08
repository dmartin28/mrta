import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

from ML.synergy_model import SynergyModel

# Load the validation data
val_data = pd.read_csv('data/val/mrta_val.csv')

# Separate features and target
X_val = val_data.iloc[:, :-1].values
y_val = val_data.iloc[:, -1].values

# Standardize the features
scaler = StandardScaler()
X_val = scaler.fit_transform(X_val)

# Convert to PyTorch tensors
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val).reshape(-1, 1)

# Load the saved model
model = SynergyModel(X_val.shape[1])
model.load_state_dict(torch.load('best_linear_nn_model.pth'))
model.eval()

# Function to get random samples
def get_random_samples(X, y, num_samples):
    indices = random.sample(range(len(X)), num_samples)
    return X[indices], y[indices]

# Get 10 random samples
num_samples = 100
X_samples, y_samples = get_random_samples(X_val, y_val, num_samples)

# Make predictions
with torch.no_grad():
    predictions = model(X_samples)

# Print results
print("Sample, Target Value, Predicted Value, Absolute Error")
print("-" * 50)
for i in range(num_samples):
    target = y_samples[i].item()
    pred = predictions[i].item()
    abs_error = abs(target - pred)
    print(f"{i+1}, {target:.2f}, {pred:.2f}, {abs_error:.2f}")

# Calculate and print overall metrics
mse = nn.MSELoss()(predictions, y_samples)
mae = nn.L1Loss()(predictions, y_samples)
print("\nOverall Metrics:")
print(f"Mean Squared Error: {mse.item():.4f}")
print(f"Mean Absolute Error: {mae.item():.4f}")