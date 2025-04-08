import torch
import torch.nn as nn
import torch.optim as optim

# Define the model architecture (make sure it matches the saved model)
class SynergyModel(nn.Module):
    def __init__(self, input_size):
        super(SynergyModel, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.linear3 = nn.Linear(input_size, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x