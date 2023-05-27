import numpy as np
import pandas as pd
import scipy.io
import math
import os
import ntpath
import sys
import logging
import time
import sys
import random

from importlib import reload

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Masking

from tensorflow.keras.utils import plot_model

IS_TRAINING = True

from data_processing.nasa_random_data import NasaRandomizedData
from data_processing.prepare_rul_data import RulHandler


# ### Config logging

round_it = sys.argv[1]


reload(logging)
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')

nasa_data_handler = NasaRandomizedData("./")
rul_handler = RulHandler()

train_x = np.load("data_preprocessing/all_processed_data/full_train_x.npy")
train_y = np.load("data_preprocessing/all_processed_data/full_train_y.npy")
test_x = np.load("data_preprocessing/all_processed_data/full_test_x.npy")
test_y = np.load("data_preprocessing/all_processed_data/full_test_y.npy")
val_x = np.load("data_preprocessing/all_processed_data/full_val_x.npy")
val_y = np.load("data_preprocessing/all_processed_data/full_val_y.npy")

# # Model training


if IS_TRAINING:
    EXPERIMENT = "lstm_autoencoder_rul_nasa_randomized"

    experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S") + '_' + EXPERIMENT
    print(experiment_name)

    # Model definition

    opt = tf.keras.optimizers.Adam(lr=0.000003)

    print(train_x.shape[1])
    print(train_x.shape[2])
    
    model = Sequential()
    model.add(layers.Input(shape=(train_x.shape[1], train_x.shape[2])))
    model.add(layers.Conv1D(64, kernel_size=8, strides=4, activation='relu',
                    kernel_regularizer=regularizers.l2(0.0002)))
    model.add(layers.Conv1D(32, kernel_size=4, strides=2, activation='relu',
                    kernel_regularizer=regularizers.l2(0.0002)))
    model.add(layers.Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0002)))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0002)))
    model.add(Dense(1, activation='linear'))
    model.summary()
    
    #plot_model(model, "cnn_graph.png")

    model.compile(optimizer=opt, loss='huber', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])


if IS_TRAINING:
    history = model.fit(train_x, train_y,
                                epochs=60, 
                                batch_size=32, 
                                verbose=1,
                                validation_data=(val_x, val_y)
                               )


print(history.history)
np.save('base_history_of_60ep_long_run.npy', history.history)

results = model.evaluate(test_x, test_y, return_dict = True)
print(results)

print("DONE! :)")
