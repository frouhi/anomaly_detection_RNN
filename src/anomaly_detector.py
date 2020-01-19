#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv("../data/klines_1h_BTCUSDT.csv")  # reads the data to a pandas data frame

feature_names = ["open","high","low","close","volume"]
features = df[feature_names]
features.index = df["time"]
features.head()
features.plot(subplots=True)
plt.show()
# In the following block we standardize the dataset using the mean and standard deviation of the training data
train_num = int(df["time"].size*0.8)  # we will use 90% of our data for training
dataset = features.values
data_mean = dataset[:train_num].mean(axis=0)
data_std = dataset[:train_num].std(axis=0)
dataset = (dataset-data_mean)/data_std


# This function breaks the data down to multiple windows of training data and targets
def prepare_multivariate_data(dataset, target, start_index, end_index, history_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - 0

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(dataset[indices])

        labels.append(target[i+0])

    return np.array(data), np.array(labels)


x_train, y_train = prepare_multivariate_data(dataset, dataset[:, 1], 0, train_num, 24)
x_val, y_val = prepare_multivariate_data(dataset, dataset[:, 1], train_num, None, 24)

batch_size = 128
buffer_size = 5000

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(batch_size).repeat()

# We build the model in the following block
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32,input_shape=x_train[0].shape))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

single_step_history = model.fit(train_data, epochs=10,
                                            steps_per_epoch=500,
                                            validation_data=val_data,
                                            validation_steps=100)

