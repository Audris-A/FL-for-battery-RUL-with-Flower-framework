# Get the full processed data set just like in the mentioned publication

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

from data_processing.nasa_random_data import NasaRandomizedData
from data_processing.prepare_rul_data import RulHandler

reload(logging)
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')

IS_TRAINING = True

data_path = "../"


### Training data ###
train_names_10k = [
        'RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW13',
        'RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW14',
        'RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW15',

        'RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW21',
        'RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW22',

        'RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW25',
        'RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW26',
        'RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW27',
]

train_names_20k = [
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW1',
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW2',
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW7',

        'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW4',
        'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW5',
]


train_names_100k = [
        'RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW17',
        'RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW18',
        'RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW19',
]
#######################


### Test data ###
test_names_10k = [
        'RW_Skewed_Low_Room_Temp_DataSet_2Post/data/Matlab/RW16',
        'RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW24',
        'RW_Skewed_High_40C_DataSet_2Post/data/Matlab/RW28',
]

test_names_20k = [
        'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW8',
        'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW6',
]

test_names_no_100k = [
        'RW_Skewed_High_Room_Temp_DataSet_2Post/data/Matlab/RW20',
]
#######################

### Validation data ###
val_names_10k = [
        'RW_Skewed_Low_40C_DataSet_2Post/data/Matlab/RW23',
]
#######################

# See the publication source for this code
nasa_data_handler = NasaRandomizedData(data_path)
rul_handler = RulHandler()


## DATA PREPARATION
#
# Getting data splitted by frequency
(train_x_10k, train_y_soh_10k, test_x_10k, test_y_soh_10k, 
  battery_range_cycle_train_10k, battery_range_cycle_test_10k,
  time_train_10k, time_test_10k, current_train_10k, current_test_10k,
  val_x_10k, val_y_soh,
  val_battery_range,
  time_val, current_val
  ) = nasa_data_handler.get_discharge_whole_cycle_future(train_names_10k, test_names_10k, val_names_10k)
(train_x_20k, train_y_soh_20k, test_x_20k, test_y_soh_20k,
  battery_range_cycle_train_20k, battery_range_cycle_test_20k,
  time_train_20k, time_test_20k, current_train_20k, current_test_20k
  ) = nasa_data_handler.get_discharge_whole_cycle_future(train_names_20k, test_names_20k)
(train_x_100k, train_y_soh_100k, test_x_no_100k, test_y_soh_100k,
  battery_range_cycle_train_100k, battery_range_cycle_test_100k,
  time_train_100k, time_test_100k, current_train_100k, current_test_100k
  ) = nasa_data_handler.get_discharge_whole_cycle_future(train_names_100k, test_names_no_100k)


# let data have same length by sampling and unify them
train_x_20k = train_x_20k[:, ::2, :]
test_x_20k = test_x_20k[:, ::2, :]
train_x_100k = train_x_100k[:, ::10, :]
max_lenght = max(train_x_10k.shape[1], test_x_10k.shape[1], train_x_20k.shape[1], test_x_20k.shape[1], train_x_100k.shape[1], test_x_no_100k.shape[1])

train_x = np.zeros((
        train_x_10k.shape[0] + train_x_20k.shape[0] + train_x_100k.shape[0],
        max_lenght,
        train_x_10k.shape[2]))
train_x[:train_x_10k.shape[0], :train_x_10k.shape[1], :] = train_x_10k
train_x[train_x_10k.shape[0]:train_x_10k.shape[0]+train_x_20k.shape[0], :train_x_20k.shape[1], :] = train_x_20k
train_x[train_x_10k.shape[0]+train_x_20k.shape[0]:, :train_x_100k.shape[1], :] = train_x_100k

val_x = np.zeros((
        val_x_10k.shape[0],
        max_lenght,
        val_x_10k.shape[2]))
val_x[:val_x_10k.shape[0], :val_x_10k.shape[1], :] = val_x_10k

test_x = np.zeros((
        test_x_10k.shape[0] + test_x_20k.shape[0] + test_x_no_100k.shape[0],
        max_lenght,
        test_x_10k.shape[2]))
test_x[:test_x_10k.shape[0], :test_x_10k.shape[1], :] = test_x_10k
test_x[test_x_10k.shape[0]:test_x_10k.shape[0]+test_x_20k.shape[0], :test_x_20k.shape[1], :] = test_x_20k
test_x[test_x_10k.shape[0]+test_x_20k.shape[0]:, :test_x_no_100k.shape[1], :] = test_x_no_100k

print("train shape {}".format(train_x.shape))
print("val shape {}".format(val_x.shape))
print("test shape {}".format(test_x.shape))

train_x = train_x[:,:11800,:]
val_x = val_x[:,:11800,:]
test_x = test_x[:,:11800,:]
print("cut train shape {}".format(train_x.shape))
print("cut val shape {}".format(val_x.shape))
print("cut test shape {}".format(test_x.shape))


train_names = train_names_10k + train_names_20k + train_names_100k
test_names = test_names_10k + test_names_20k + test_names_no_100k

battery_range_cycle_train_20k += battery_range_cycle_train_10k[-1]
battery_range_cycle_train_100k += battery_range_cycle_train_20k[-1]
train_battery_range = np.concatenate((battery_range_cycle_train_10k, battery_range_cycle_train_20k, battery_range_cycle_train_100k), axis=None)
battery_range_cycle_test_20k += battery_range_cycle_test_10k[-1]
battery_range_cycle_test_100k += battery_range_cycle_test_20k[-1]
test_battery_range = np.concatenate((battery_range_cycle_test_10k, battery_range_cycle_test_20k, battery_range_cycle_test_100k), axis=None)

max_lenght = max(time_train_10k.shape[1], time_train_20k.shape[1], time_train_100k.shape[1], time_test_10k.shape[1], time_test_20k.shape[1], time_test_100k.shape[1])
#
time_train = np.zeros((
        time_train_10k.shape[0] + time_train_20k.shape[0] + time_train_100k.shape[0],
        max_lenght))
time_train[:time_train_10k.shape[0], :time_train_10k.shape[1]] = time_train_10k
time_train[time_train_10k.shape[0]:time_train_10k.shape[0]+time_train_20k.shape[0], :time_train_20k.shape[1]] = time_train_20k
time_train[time_train_10k.shape[0]+time_train_20k.shape[0]:, :time_train_100k.shape[1]] = time_train_100k
time_test = np.zeros((
        time_test_10k.shape[0] + time_test_20k.shape[0] + time_test_100k.shape[0],
        max_lenght))
time_test[:time_test_10k.shape[0], :time_test_10k.shape[1]] = time_test_10k
time_test[time_test_10k.shape[0]:time_test_10k.shape[0]+time_test_20k.shape[0], :time_test_20k.shape[1]] = time_test_20k
time_test[time_test_10k.shape[0]+time_test_20k.shape[0]:, :time_test_100k.shape[1]] = time_test_100k
#
current_train = np.zeros((
        current_train_10k.shape[0] + current_train_20k.shape[0] + current_train_100k.shape[0],
        max_lenght))
current_train[:current_train_10k.shape[0], :current_train_10k.shape[1]] = current_train_10k
current_train[current_train_10k.shape[0]:current_train_10k.shape[0]+current_train_20k.shape[0], :current_train_20k.shape[1]] = current_train_20k
current_train[current_train_10k.shape[0]+current_train_20k.shape[0]:, :current_train_100k.shape[1]] = current_train_100k
current_test = np.zeros((
        current_test_10k.shape[0] + current_test_20k.shape[0] + current_test_100k.shape[0],
        max_lenght))
current_test[:current_test_10k.shape[0], :current_test_10k.shape[1]] = current_test_10k
current_test[current_test_10k.shape[0]:current_test_10k.shape[0]+current_test_20k.shape[0], :current_test_20k.shape[1]] = current_test_20k
current_test[current_test_10k.shape[0]+current_test_20k.shape[0]:, :current_test_100k.shape[1]] = current_test_100k

train_y_soh = np.concatenate((train_y_soh_10k, train_y_soh_20k, train_y_soh_100k), axis=None)
test_y_soh = np.concatenate((test_y_soh_10k, test_y_soh_20k, test_y_soh_100k), axis=None)

print("train names shape {}".format(len(train_names)))
print("test names shape {}".format(len(test_names)))
print("train battery range {}".format(train_battery_range))
print("test battery range {}".format(test_battery_range))
print("train time shape {}".format(time_train.shape))
print("test time shape {}".format(time_test.shape))
print("train current shape {}".format(current_train.shape))
print("test current shape {}".format(current_test.shape))
print("train y soh shape {}".format(train_y_soh.shape))
print("test y soh shape {}".format(test_y_soh.shape))

CAPACITY_THRESHOLDS = None
NOMINAL_CAPACITY = 2.2
N_CYCLE = 1000 # CNN had 1000, but LSTM had 500
WARMUP_TRAIN = 15
WARMUP_TEST = 30

#preparing y
train_y = rul_handler.prepare_y_future(train_names, train_battery_range, train_y_soh, current_train, time_train, CAPACITY_THRESHOLDS, capacity=NOMINAL_CAPACITY)
del globals()["current_train"]
del globals()["time_train"]
val_y = rul_handler.prepare_y_future(val_names_10k, val_battery_range, val_y_soh, current_val, time_val, CAPACITY_THRESHOLDS, capacity=NOMINAL_CAPACITY)
del globals()["current_val"]
del globals()["time_val"]
test_y = rul_handler.prepare_y_future(test_names, test_battery_range, test_y_soh, current_test, time_test, CAPACITY_THRESHOLDS, capacity=NOMINAL_CAPACITY)
del globals()["current_test"]
del globals()["time_test"]

x_norm = rul_handler.Normalization()
x_norm.fit(train_x)
train_x = x_norm.normalize(train_x)
val_x = x_norm.normalize(val_x)
test_x = x_norm.normalize(test_x)


## COMPRESSING X USING THE AUTOENCODER
#
# Model definition

opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
LATENT_DIM_LOCAL = 7
LATENT_DIM_GLOBAL = 7

class Autoencoder(Model):
    def __init__(self, latent_dim_local, latent_dim_global):
        super(Autoencoder, self).__init__()
        self.latent_dim_local = latent_dim_local
        self.latent_dim_global = latent_dim_global

        encoder_inputs = layers.Input(shape=(train_x.shape[1], train_x.shape[2]))
        encoder_conv1 = layers.Conv1D(filters=16, kernel_size=10, strides=5, activation='relu', padding='same')(encoder_inputs)
        encoder_pool1 = layers.MaxPooling1D(5, padding='same')(encoder_conv1)
        encoder_conv2 = layers.Conv1D(filters=8, kernel_size=4, strides=2, activation='relu', padding='same')(encoder_pool1)
        encoder_pool2 = layers.MaxPooling1D(4, padding='same')(encoder_conv2)
        encoder_flat1 = layers.Flatten()(encoder_pool1)
        encoder_flat2 = layers.Flatten()(encoder_pool2)
        encoder_dense_local = layers.Dense(self.latent_dim_local, activation='relu')(encoder_flat1)
        encoder_dense_global = layers.Dense(self.latent_dim_global, activation='relu')(encoder_flat2)
        encoder_concat = layers.concatenate([encoder_dense_local, encoder_dense_global])
        self.encoder = Model(inputs=encoder_inputs, outputs=encoder_concat)

        decoder_inputs = layers.Input(shape=(self.latent_dim_local+self.latent_dim_global,))
        decoder_dense1 = layers.Dense(59*16, activation='relu')(decoder_inputs)
        decoder_reshape1 = layers.Reshape((59, 16))(decoder_dense1)
        decoder_upsample1 = layers.UpSampling1D(4)(decoder_reshape1)
        decoder_convT1 = layers.Conv1DTranspose(filters=8, kernel_size=4, strides=2, activation='relu', padding='same')(decoder_upsample1)
        decoder_upsample2 = layers.UpSampling1D(5)(decoder_convT1)
        decoder_convT2 = layers.Conv1DTranspose(filters=16, kernel_size=10, strides=5, activation='relu', padding='same')(decoder_upsample2)
        decoder_outputs = layers.Conv1D(3, kernel_size=3, activation='relu', padding='same')(decoder_convT2)
        self.decoder = Model(inputs=decoder_inputs, outputs=decoder_outputs)



    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(LATENT_DIM_LOCAL, LATENT_DIM_GLOBAL)
autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
autoencoder.encoder.summary()
autoencoder.decoder.summary()

# read previously generated weights
autoencoder.load_weights(data_path + '../autoencoder_results/autoencoder_weights')


# compression
train_x = autoencoder.encoder(train_x).numpy()
val_x = autoencoder.encoder(val_x).numpy()
test_x = autoencoder.encoder(test_x).numpy()
print("compressed train x shape {}".format(train_x.shape))
print("compressed val x shape {}".format(val_x.shape))
print("compressed test x shape {}".format(test_x.shape))
test_x = test_x[:,~np.all(train_x == 0, axis=0)]#we need same column number of training
val_x = val_x[:,~np.all(train_x == 0, axis=0)]
train_x = train_x[:,~np.all(train_x == 0, axis=0)]
print("compressed train x shape without zero column {}".format(train_x.shape))
print("compressed val x shape without zero column {}".format(val_x.shape))
print("compressed test x shape without zero column {}".format(test_x.shape))


## DATA PREPARATION FOR THE NEURAL NETWORK
#
x_norm = rul_handler.Normalization()
x_norm.fit(train_x)
train_x = x_norm.normalize(train_x)
val_x = x_norm.normalize(val_x)
test_x = x_norm.normalize(test_x)

train_x = rul_handler.battery_life_to_time_series(train_x, N_CYCLE, train_battery_range)
val_x = rul_handler.battery_life_to_time_series(val_x, N_CYCLE, val_battery_range)
test_x = rul_handler.battery_life_to_time_series(test_x, N_CYCLE, test_battery_range)

train_x, train_y, train_battery_range, train_y_soh = rul_handler.delete_initial(train_x, train_y, train_battery_range, train_y_soh, WARMUP_TRAIN)
val_x, val_y, val_battery_range, val_y_soh = rul_handler.delete_initial(val_x, val_y, val_battery_range, val_y_soh, WARMUP_TRAIN)
test_x, test_y, test_battery_range, test_y_soh = rul_handler.delete_initial(test_x, test_y, test_battery_range, test_y_soh, WARMUP_TEST)

train_x, train_y, train_battery_range, train_y_soh = rul_handler.limit_zeros(train_x, train_y, train_battery_range, train_y_soh)
val_x, val_y, val_battery_range, val_y_soh = rul_handler.limit_zeros(val_x, val_y, val_battery_range, val_y_soh)
test_x, test_y, test_battery_range, test_y_soh = rul_handler.limit_zeros(test_x, test_y, test_battery_range, test_y_soh)

# first one is SOH, we keep only RUL
train_y = train_y[:,1]
val_y = val_y[:,1]
test_y = test_y[:,1]


## Y NORMALIZATION
#
y_norm = rul_handler.Normalization()
y_norm.fit(train_y)
train_y = y_norm.normalize(train_y)
val_y = y_norm.normalize(val_y)
test_y = y_norm.normalize(test_y)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
print(val_x.shape)
print(val_y.shape)

np.save("all_processed_data/full_train_x.npy", train_x, allow_pickle=False, fix_imports=False)
np.save("all_processed_data/full_train_y.npy", train_y, allow_pickle=False, fix_imports=False)
np.save("all_processed_data/full_test_x.npy", test_x, allow_pickle=False, fix_imports=False)
np.save("all_processed_data/full_test_y.npy", test_y, allow_pickle=False, fix_imports=False)
np.save("all_processed_data/full_val_x.npy", val_x, allow_pickle=False, fix_imports=False)
np.save("all_processed_data/full_val_y.npy", val_y, allow_pickle=False, fix_imports=False)

## DATA DIVISION FOR TEST SETUPS
#
# Min client count = 2
# Max client count = 12
# Step = 1