import numpy as np
import pandas as pd
from merlion.utils import TimeSeries
from merlion.utils import time_series
import re
import pywt

def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.01  # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')

    return datarec

def standardize(data):
    """
    Standardize the data.
    """
    return (data - np.mean(data)) / np.std(data)

def read_data(path, **kwargs):
    """
    Read data from a csv file.
    """
    df = pd.read_csv(path + '.csv', **kwargs)
    lb = read_labels(path, len(df))

    df = df.head(50000)
    df['value'] = denoise(df["'MLII'"])
    df['value'] = standardize(df['value'])
    ts_data = TimeSeries.from_pd(df["value"], freq="1D")
    lb = lb.head(50000)
    ts_label = TimeSeries.from_pd(lb['anomaly'], freq="1D")
    print(ts_data)
    print(ts_label)

    return ts_data, ts_label

def read_labels(path, length):
    f = open(path + 'annotations.txt', "r").read()
    ts = f.split('\n')[1:-1]
    for i in range(len(ts)):
        ts[i] = re.sub("\s+", '\t', ts[i])
        ts[i] = ts[i].split('\t')
        #print(ts[i])

    lb = np.zeros(length)
    for i in range(len(ts)):
        if ts[i][3] != 'N':
            #print("check")
            for j in range(int(ts[i][2]) -100, int(ts[i][2]) +100):
                lb[j] = 1

    #print(np.count_nonzero(lb) / 100)
    df = pd.DataFrame(lb, columns=['anomaly'])

    return df

#df = read_data('dataset/100')
#print(df)