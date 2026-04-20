import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=3):
        super(CNN_LSTM, self).__init__()
        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 128) 
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), x.size(1), -1)  # Reshape for LSTM
        
        h0 = torch.zeros(2, x.size(0), 128).to(x.device)  # Initial hidden state
        c0 = torch.zeros(2, x.size(0), 128).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            x = F.softmax(x, dim=1)
        return x