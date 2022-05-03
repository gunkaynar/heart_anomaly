from merlion.utils import TimeSeries
from ts_datasets.anomaly import NAB
from mutils import *
import sys
import os


# Data loader returns pandas DataFrames, which we convert to Merlion TimeSeries
time_series, metadata = NAB(subset="realKnownCause")[3]
train_data, train_labels = read_data('C:/Users/TR/Desktop/CS550 Proje/heart_anomaly/dataset/101')
test_data, test_labels = read_data('C:/Users/TR/Desktop/CS550 Proje/heart_anomaly/dataset/101')
test_labels_tmp = TimeSeries.from_pd(metadata.anomaly[~metadata.trainval])
"""

time_series, metadata = NAB(subset="realKnownCause")[3]
train_data = TimeSeries.from_pd(time_series[metadata.trainval])
test_data = TimeSeries.from_pd(time_series[~metadata.trainval])
test_labels = TimeSeries.from_pd(metadata.anomaly[~metadata.trainval])
"""

print("step 1")
from merlion.models.anomaly.vae import VAE, VAEConfig
from merlion.models.anomaly.autoencoder import AutoEncoder, AutoEncoderConfig
model = VAE(VAEConfig(sequence_len=30, latent_size=2,encoder_hidden_sizes=(25, 10, 2), decoder_hidden_sizes=(2, 10, 25)))
model.train(train_data=train_data)
print("step 1-5")
test_pred = model.get_anomaly_label(test_data)

print("step 2")
from merlion.plot import plot_anoms
import matplotlib.pyplot as plt
fig, ax = model.plot_anomaly(time_series=test_data)
plot_anoms(ax=ax, anomaly_labels=test_labels)
plt.show()

from merlion.evaluate.anomaly import TSADMetric
p = TSADMetric.Precision.value(ground_truth=test_labels, predict=test_pred)
r = TSADMetric.Recall.value(ground_truth=test_labels, predict=test_pred)
f1 = TSADMetric.F1.value(ground_truth=test_labels, predict=test_pred)
mttd = TSADMetric.MeanTimeToDetect.value(ground_truth=test_labels, predict=test_pred)
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}\n"
      f"Mean Time To Detect: {mttd}")