import torch.nn as nn
import torch
import numpy as np

from torch.autograd import Variable
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def shuffle_torch(x, y):
    p = np.random.permutation(x.shape[0])
    return x[p], y[p]


def batch_generator_torch(x, y, batch_size, shuffle=True):
    if shuffle:
        x, y = shuffle_torch(x, y)
    n_samples = x.shape[0]
    for i in range(0, n_samples, batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]
