import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import pickle
from scipy.stats import pearsonr

from ML.synergy_model import SynergyModel

# Load the validation data
val_data = pd.read_csv('data/val/mrta_val.csv')

# Separate features and target
X_val = val_data.iloc[:, :-1].values
y_val = val_data.iloc[:, -1].values

# Load the scaler from the pickle file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"X_val unscaled: {X_val}")
X_val = scaler.transform(X_val)  # Use transform instead of fit_transform
print(f"X_val scaled: {X_val}")

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

# Get random samples
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

# Calculate correlation
y_true = y_samples.numpy().flatten()
y_pred = predictions.numpy().flatten()
correlation, _ = pearsonr(y_true, y_pred)
print(f"Correlation between predictions and true values: {correlation:.4f}")

# Calculate R-squared
y_mean = np.mean(y_true)
ss_tot = np.sum((y_true - y_mean)**2)
ss_res = np.sum((y_true - y_pred)**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared:.4f}")