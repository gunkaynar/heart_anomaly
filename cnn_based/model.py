import torch.nn as nn
import torch


class AnomalyModel(torch.nn.Module):

    def __init__(self):
        super(AnomalyModel, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=360, out_channels=16, kernel_size=13, padding='same', stride=1)
        self.activation = torch.nn.ReLU()
        self.pool1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15,padding='same', stride=1)
        self.cnn3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=17, padding='same', stride=1)
        self.cnn4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=19, padding='same', stride=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_features=128, out_features=35)
        self.linear2 = nn.Linear(in_features=35, out_features=5)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.cnn3(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.cnn4(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
