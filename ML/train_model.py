"""
MRTA Synergy Model Training Script

This script trains a neural network model to predict the synergy (additional reward)
obtained by merging two clusters in a Multi-Robot Task Allocation (MRTA) problem.
It loads training and validation data, preprocesses it using StandardScaler,
trains the model, and saves the best performing model based on validation loss.

The model architecture is defined in the SynergyModel class (imported from ML.synergy_model).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

from ML.synergy_model import SynergyModel

# Constants
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
BEST_MODEL_PATH = 'best_linear_nn_model.pth'
SCALER_FILENAME = "scaler.pkl"

# Load and preprocess data
def load_and_preprocess_data(train_path, val_path):
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val, scaler

# Convert data to PyTorch tensors
def to_tensor(X, y):
    return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

# Training loop
def train_model(model, X_train, y_train, X_val, y_val):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            batch_X = X_train[i:i+BATCH_SIZE]
            batch_y = y_train[i:i+BATCH_SIZE]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / (X_train.shape[0] // BATCH_SIZE)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {val_loss.item():.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Model saved with validation loss: {best_val_loss:.4f}")

    return best_val_loss

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_val, y_val, scaler = load_and_preprocess_data(
        'data/train/mrta_train.csv',
        'data/val/mrta_val.csv'
    )

    # Save the scaler
    with open(SCALER_FILENAME, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print(f"Scaler saved as '{SCALER_FILENAME}'")

    # Convert to PyTorch tensors
    X_train, y_train = to_tensor(X_train, y_train)
    X_val, y_val = to_tensor(X_val, y_val)

    # Instantiate and train the model
    model = SynergyModel(X_train.shape[1])
    best_val_loss = train_model(model, X_train, y_train, X_val, y_val)

    print(f"Training complete. Best model saved as '{BEST_MODEL_PATH}' "
          f"with validation loss: {best_val_loss:.4f}")