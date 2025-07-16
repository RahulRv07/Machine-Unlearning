import torch.nn as nn
import torch.nn.functional as F

class BreastCancerNet(nn.Module):
    def __init__(self, input_size=30):  # 30 features after dropping 'id' and 'diagnosis'
        super(BreastCancerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Binary classification (0 or 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits, apply CrossEntropyLoss externally
