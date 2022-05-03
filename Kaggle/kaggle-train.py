import torch
import torch.nn as nn

import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.01  # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')

    return datarec

#ecg_record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/106', channels= [0])

ecg_record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/106', sampfrom= 0, sampto= 108300, channels= [0])

ecg = denoise(ecg_record.p_signal[:,0])
train_len = 108000

ecg_train = ecg[0: train_len]
ecg_test = ecg[train_len:]

ecg_train = torch.FloatTensor(ecg_train).view(-1)
ecg_test = torch.FloatTensor(ecg_test).view(-1)

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
                                          shuffle=True, drop_last= True)
  return data_loader

train_window = 850  # 3 cardiac cycles in each sequence
# train_data = create_inout_sequences(ecg_train, train_window)
batch_size = 64


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

ecg_val = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/100', sampfrom= 108300, sampto= 129900, channels= [0])
ecg_val = denoise(ecg_val.p_signal[:,0])

wfdb.plot_items(ecg_val)
ecg_val = torch.FloatTensor(ecg_val).view(-1)

seq_length = train_window
train_data_loader = create_dataloader(ecg_train, seq_length, batch_size)
val_data_loader = create_dataloader(ecg_val, seq_length, batch_size)


def train(net, train_data_loader, val_data_loader, epochs=20, batch_size=batch_size, lr=0.1, clip=5, print_every=10):
    ''' Training a network

        Arguments
        ---------

        net: LSTM network
        data: ecg data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()

    if (train_on_gpu):
        net.cuda()

    counter = 0
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in train_data_loader:
            counter += 1
            x = torch.unsqueeze(x, 2)

            inputs, targets = x, y

            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(torch.flatten(output), torch.flatten(targets))
            #loss = criterion(output, targets.view(batch_size))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in val_data_loader:
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    x = torch.unsqueeze(x, 2)

                    inputs, targets = x, y
                    if (train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(torch.flatten(output), torch.flatten(targets))

                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

hidden_dim=32
n_layers=3

net = LSTM(hidden_dim= hidden_dim, n_layers= n_layers)
print(net)

train_on_gpu = torch.cuda.is_available()
print(train_on_gpu)

batch_size = 64
seq_length = train_window
n_epochs = 1

# model training
# train(net, ecg_train, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

train(net, train_data_loader, val_data_loader, epochs=n_epochs, batch_size=batch_size, lr=0.0001, print_every=10)

model_name = 'rnn_20_epoch.net'

checkpoint_20E = {'n_hidden': net.hidden_dim,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict()}

with open(model_name, 'wb') as f:
    torch.save(checkpoint_20E, f)