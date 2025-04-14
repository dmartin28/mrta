"""
MRTA Synergy Model Evaluation Script

This script evaluates a trained neural network model that predicts the synergy (additional reward)
obtained by merging two clusters in a Multi-Robot Task Allocation (MRTA) problem.
It loads the validation data, applies scaling, makes predictions using the trained model,
and calculates various performance metrics including MSE, MAE, correlation, and R-squared.

The model architecture is defined in the SynergyModel class (imported from ML.synergy_model).
"""

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
import ML.ML_utils as utils

# Constants
NUM_SAMPLES = 100
MODEL_PATH = 'best_linear_nn_model.pth'
SCALER_PATH = 'scaler.pkl'
VAL_DATA_PATH = 'data/val/mrta_val.csv'

def get_random_samples(X, y, num_samples):
    """Get random samples from the dataset."""
    indices = random.sample(range(len(X)), num_samples)
    return X[indices], y[indices]

def evaluate_model(model, X_samples, y_samples):
    """Evaluate the model and print results."""
    with torch.no_grad():
        predictions = model(X_samples)

    print("Sample, Target Value, Predicted Value, Absolute Error")
    print("-" * 50)
    for i in range(len(y_samples)):
        target = y_samples[i].item()
        pred = predictions[i].item()
        abs_error = abs(target - pred)
        print(f"{i+1}, {target:.2f}, {pred:.2f}, {abs_error:.2f}")

    # Calculate overall metrics
    mse = nn.MSELoss()(predictions, y_samples)
    mae = nn.L1Loss()(predictions, y_samples)
    print("\nOverall Metrics:")
    print(f"Mean Squared Error: {mse.item():.4f}")
    print(f"Mean Absolute Error: {mae.item():.4f}")

    # Calculate correlation and R-squared
    y_true = y_samples.numpy().flatten()
    y_pred = predictions.numpy().flatten()
    correlation, _ = pearsonr(y_true, y_pred)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"Correlation between predictions and true values: {correlation:.4f}")
    print(f"R-squared: {r_squared:.4f}")

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X_val, y_val = utils.load_and_preprocess_data(VAL_DATA_PATH, SCALER_PATH)

    # Convert to PyTorch tensors
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).reshape(-1, 1)

    # Load the saved model
    model = SynergyModel(X_val.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Get random samples and evaluate
    X_samples, y_samples = get_random_samples(X_val, y_val, NUM_SAMPLES)
    evaluate_model(model, X_samples, y_samples)