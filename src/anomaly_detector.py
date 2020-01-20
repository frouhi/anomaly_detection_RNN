import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
In this script, first we build a Recurrent Neural Network (RNN) to forecast the financial times series.
Note that the financial time series are multivariate.
'''

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
def prepare_multivariate_data(dataset, start_index, end_index, history_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - 0

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(dataset[indices])

        labels.append(dataset[i+0,0:4])

    return np.array(data), np.array(labels)


x_train, y_train = prepare_multivariate_data(dataset, 0, train_num, 24)
x_val, y_val = prepare_multivariate_data(dataset, train_num, None, 24)

batch_size = 128
buffer_size = 5000

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(batch_size).repeat()

# We build the model in the following block
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32,input_shape=x_train[0].shape))
model.add(tf.keras.layers.Dense(4))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

history = model.fit(train_data, epochs=10,steps_per_epoch=250,
                                            validation_data=val_data,
                                            validation_steps=50)

# Plot the history
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

#model.save("../data/saved_model/RNN_model")
# Next section finds anomalies and plots them
times = features.index[train_num:]
anomaly_times = np.array([])
ys = np.array([])
num = 0
for x, y in val_data.take(30):
    predictions = model.predict(x)
    ys = np.append(ys,y.numpy()[:,3])
    for i in range(0,len(x)):
        if (abs(predictions[i,:]-y[i,:])/y[i,:]>0.1).numpy().all():
            anomaly_times = np.append(anomaly_times, num)
        num += 1

indexes = np.array(range(0,num))
plt.figure()
plt.title("Price Anomalies")
plt.plot(indexes,ys,"g")
for xc in anomaly_times:
    plt.axvline(xc)
plt.show()