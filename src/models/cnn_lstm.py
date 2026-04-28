import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN_LSTM, self).__init__()

        # First convolutional layer: input channels = 12 (for MFCC), output channels = 16, kernel size = 3
        self.conv1 = nn.Conv1d(12, 16, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: input channels = 16, output channels = 32, kernel size = 3
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2) # best: 0.2
        self.fc = nn.Linear(64,num_classes)
    
    def __str__(self):
        return 'CNN_LSTM'

    def forward(self, x):
        # x: (batch,12,298)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # reshape for LSTM
        x = x.permute(0,2,1)
        
        x, _ = self.lstm(x)
        
        # Take the last time step's output
        x=x[:, -1, :]
        x = self.dropout(x)
        
        out = self.fc(x)  

        return out
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            x = F.softmax(x, dim=1)
        return x