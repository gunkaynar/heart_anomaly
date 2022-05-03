import torch
import torch.nn as nn

import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pywt

#train_on_gpu = torch.cuda.is_available()
train_on_gpu = False
print(train_on_gpu)

def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.01  # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')

    return datarec

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        if (i+tw+tw > L):
            break
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+tw]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def create_dataloader(data_entered, train_window, batch_size):
  data = create_inout_sequences(data_entered, train_window)
  data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=batch_size,
                                          shuffle=False, drop_last= True)
  return data_loader

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_dim=100, n_layers=5, output_size=1, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.lr = lr

        ## define the LSTM
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=n_layers, dropout=drop_prob, batch_first=True)

        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## define the final, fully-connected output layer
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, input_seq, hidden):
        lstm_out, hidden = self.lstm(input_seq, hidden)
        out = self.dropout(lstm_out)
        out = out.contiguous().view(-1, self.hidden_dim)

        # fully-connected layer
        out = self.linear(out)

        batch_size = input_seq.size(0)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

def test(model, test_loader, batch_size):
    if (train_on_gpu):
        model.cuda()
    arr = []
    with torch.no_grad():
        model.eval()
        criterion = nn.L1Loss()
        total = 0
        total_loss = 0
        model.eval()
        h = net.init_hidden(batch_size)
        for data, target in tqdm(test_loader):
            data = torch.unsqueeze(data, 2)

            if (train_on_gpu):
                data, target = data.cuda(), target.cuda()

            h = tuple([each.data for each in h])

            #h = torch.stack(h)

            outputs, h = model(data, h)

            #temp = np.abs(outputs.cpu().detach().numpy() - target.cpu().detach().numpy())
            #print(torch.flatten(target))
            loss = criterion(torch.flatten(outputs), torch.flatten(target))
            for i in range(batch_size):
                arr.append(loss)
            total += target.size(0)
            total_loss += loss.item()
            print(loss.item())

        print('Test Loss of the model: {}'.format(total_loss/total))

    return arr



train_window = 850  # 3 cardiac cycles in each sequence
# train_data = create_inout_sequences(ecg_train, train_window) 129900
batch_size = 64

ecg_val = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/106', sampfrom= 259900, sampto= 299900 , channels= [0])
ecg_ann = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/106', extension='atr', sampfrom= 259900, sampto= 299900, shift_samps=True)
wfdb.plot_wfdb(ecg_val, ecg_ann,plot_sym=True)

ecg_val = denoise(ecg_val.p_signal[:,0])

print(ecg_ann.sample)

wfdb.plot_items(ecg_val, ann_samp=[ecg_ann.sample])
ecg_val = torch.FloatTensor(ecg_val).view(-1)

seq_length = train_window
val_data_loader = create_dataloader(ecg_val, seq_length, batch_size)


hidden_dim=32
n_layers=3

net = LSTM(hidden_dim= hidden_dim, n_layers= n_layers)
print(net)

net.load_state_dict(torch.load('rnn_20_epoch.net')["state_dict"])

an = test(net, val_data_loader, batch_size)

an = [a_ if a_ > 0.30 else 0 for a_ in an]

plt.plot(an)
plt.show()

tp = 0
pos = 0
fn = 0

for i in range(len(ecg_ann.sample)):
    ts = ecg_ann.sample[i]
    typ = ecg_ann.symbol[i]
    if typ != "N":
        pos += 1
        #print(ts)
        if an[max(ts - (2 * train_window), 0)] != 0:
            tp += 1
        else:
            fn += 1

print(tp, pos, fn)

all = []
for i in range(0,len(an),batch_size):
    if an[i] != 0:
        all.append(i)


precision = tp / (tp + fn)
recall = tp / len(all)

print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))