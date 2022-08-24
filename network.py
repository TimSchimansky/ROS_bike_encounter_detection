"""import torch
import torch.nn as nn
#import torch.nn.functional as F

import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, width=100, n_vars=6, n_classes=7, conv_kernel_size=5, conv_filters=3, lstm_units=3):
        super(NeuralNet, self).__init__()

        # Calc base vars
        n_vars_0 = n_vars
        n_vars_1 = n_vars_0 * conv_filters
        n_vars_2 = n_vars_1 * conv_filters

        lstm_input_width = 100 - 4 * int(5 / 2)

        self.conv_1 = nn.Conv1d(in_channels=n_vars_0, out_channels=n_vars_1, kernel_size=conv_kernel_size, padding='valid')
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(in_channels=n_vars_1, out_channels=n_vars_2, kernel_size=conv_kernel_size, padding='valid')
        self.relu_2 = nn.ReLU()
        self.lstm_1 = nn.LSTM(input_size=lstm_input_width, hidden_size=5, num_layers=lstm_units, dropout=0.1)
        self.line_1 = nn.Linear(5, n_classes)
        self.soft_1 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv_1(torch.transpose(x, 1,2))
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x, _ = self.lstm_1(x)
        x = self.line_1(x)
        x = self.soft_1(x)
        return x"""

