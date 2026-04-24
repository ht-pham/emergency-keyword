import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        # First convolutional layer: input channels = 12 (for MFCC), output channels = 16, kernel size = 3
        self.conv1 = nn.Conv1d(12, 16, kernel_size=3, stride=2, padding=1)
        # Second convolutional layer: input channels = 16, output channels = 32, kernel size = 3
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, num_classes)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = x.squeeze(-1)
        x = self.fc1(x)
        return x
    
    
    def loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            x = F.softmax(x, dim=1)
        return x