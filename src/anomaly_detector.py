import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


model = tf.keras.models.load_model("../data/saved_model/RNN_model")
model.summary()

# for x, y in val_data_single.take(3):
#   plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
#                     single_step_model.predict(x)[0]], 12,
#                    'Single Step Prediction')
#   plot.show()