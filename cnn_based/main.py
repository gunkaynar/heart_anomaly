import numpy as np
import pandas as pd
import os
from model import *
import matplotlib.pyplot as plt
import csv
import itertools
import collections
import torch.optim as optim

from scipy import stats
import pywt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04  # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')

    return datarec


path = '/Users/gunkaynar/Desktop/Courses/CS_550/heart_anomaly/dataset/'
window_size = 180
maximum_counting = 10000

classes = ['N', 'S', 'F', 'V', 'Q']
n_classes = len(classes)
count_classes = [0] * n_classes

X = list()
y = list()

# Read files
filenames = next(os.walk(path))[2]

# Split and save .csv , .txt
records = list()
annotations = list()
filenames.sort()

# segrefating filenames and annotations
for f in filenames:
    filename, file_extension = os.path.splitext(f)

    # *.csv
    if (file_extension == '.csv'):
        records.append(path + filename + file_extension)

    # *.txt
    else:
        annotations.append(path + filename + file_extension)

# Records
for r in range(0, len(records)):
    signals = []

    with open(records[r], 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')  # read CSV file\
        row_index = -1
        for row in spamreader:
            if (row_index >= 0):
                signals.insert(row_index, int(row[1]))
            row_index += 1

    # Plot an example to the signals
    if r == 1:
        # Plot each patient's signal
        plt.title(records[1] + " Wave")
        plt.plot(signals[0:700])
        plt.show()

    signals = denoise(signals)
    # Plot an example to the signals
    if r == 1:
        # Plot each patient's signal
        plt.title(records[1] + " wave after denoised")
        plt.plot(signals[0:700])
        plt.show()

    signals = stats.zscore(signals)
    # Plot an example to the signals
    if r == 1:
        # Plot each patient's signal
        plt.title(records[1] + " wave after z-score normalization ")
        plt.plot(signals[0:700])
        plt.show()

    # Read anotations: R position and Arrhythmia class
    example_beat_printed = False
    with open(annotations[r], 'r') as fileID:
        data = fileID.readlines()
        beat = list()

        for d in range(1, len(data)):  # 0 index is Chart Head
            splitted = data[d].split(' ')
            splitted = filter(None, splitted)
            next(splitted)  # Time... Clipping
            pos = int(next(splitted))  # Sample ID
            arrhythmia_type = next(splitted)  # Type
            if (arrhythmia_type in classes):
                arrhythmia_index = classes.index(arrhythmia_type)
                #                 if count_classes[arrhythmia_index] > maximum_counting: # avoid overfitting
                #                     pass
                #                 else:
                count_classes[arrhythmia_index] += 1
                if (window_size <= pos and pos < (len(signals) - window_size)):
                    beat = signals[pos - window_size:pos + window_size]  ## REPLACE WITH R-PEAK DETECTION
                    # Plot an example to a beat
                    """
                    if r == 1 and not example_beat_printed:
                        plt.title("A Beat from " + records[1] + " Wave")
                        plt.plot(beat)
                        plt.show()
                        example_beat_printed = True
                    """
                    X.append(beat)
                    y.append(arrhythmia_index)

# data shape
print(np.shape(X), np.shape(y))

for i in range(0, len(X)):
    X[i] = np.append(X[i], y[i])
#         X[i].append(y[i])

print(np.shape(X))

X_train_df = pd.DataFrame(X)
per_class = X_train_df[X_train_df.shape[1] - 1].value_counts()
print(per_class)
plt.figure(figsize=(20, 10))
my_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(per_class, labels=['N', 'S', 'F', 'V', 'Q'],
        colors=['tab:blue', 'tab:orange', 'tab:purple', 'tab:olive', 'tab:green'], autopct='%1.1f%%')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

df_1 = X_train_df[X_train_df[X_train_df.shape[1] - 1] == 1]
df_2 = X_train_df[X_train_df[X_train_df.shape[1] - 1] == 2]
df_3 = X_train_df[X_train_df[X_train_df.shape[1] - 1] == 3]
df_4 = X_train_df[X_train_df[X_train_df.shape[1] - 1] == 4]
# df_5=X_train_df[X_train_df[X_train_df.shape[1]-1]==5]
df_0 = (X_train_df[X_train_df[X_train_df.shape[1] - 1] == 0]).sample(n=5000, random_state=42)

df_1_upsample = resample(df_1, replace=True, n_samples=5000, random_state=122)
df_2_upsample = resample(df_2, replace=True, n_samples=5000, random_state=123)
df_3_upsample = resample(df_3, replace=True, n_samples=5000, random_state=124)
df_4_upsample = resample(df_4, replace=True, n_samples=5000, random_state=125)
# df_5_upsample=resample(df_5,replace=True,n_samples=5000,random_state=126)

# X_train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample,df_5_upsample])
X_train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

per_class = X_train_df[X_train_df.shape[1] - 1].value_counts()
print(per_class)
plt.figure(figsize=(20, 10))
my_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(per_class, labels=['N', 'S', 'F', 'V', 'Q'],
        colors=['tab:blue', 'tab:orange', 'tab:purple', 'tab:olive', 'tab:green'], autopct='%1.1f%%')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

train, test = train_test_split(X_train_df, test_size=0.20)

print("X_train : ", np.shape(train))
print("X_test  : ", np.shape(test))


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


target_train = train[train.shape[1] - 1]
target_test = test[test.shape[1] - 1]
train_y = to_categorical(target_train)
test_y = to_categorical(target_test)
print(np.shape(train_y), np.shape(test_y))

train_x = train.iloc[:, :train.shape[1] - 1].values
test_x = test.iloc[:, :test.shape[1] - 1].values
train_x = train_x.reshape(len(train_x), train_x.shape[1], 1)
test_x = test_x.reshape(len(test_x), test_x.shape[1], 1)
print(np.shape(train_x), np.shape(test_x))

pickle.dump(train_x, open("train_x.p", "wb"))
pickle.dump(test_x, open("test_x.p", "wb"))
pickle.dump(train_y, open("train_y.p", "wb"))
pickle.dump(test_y, open("test_y.p", "wb"))

net = AnomalyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0

    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(train_x)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    print(f'[{epoch + 1}] loss: {running_loss :.3f}')
    running_loss = 0.0

print('Finished Training')
