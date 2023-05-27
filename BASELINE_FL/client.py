import flwr as fl
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Masking

from logging import WARNING, INFO
from flwr.common.logger import log

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from client_class import BatteryClient


# Limit GPU resources otherwise may create odd problems
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]) # 12288
  except RuntimeError as e:
    print(e)

# Each client has a different part of the data but
# with the same size
train_x = np.load(sys.argv[1])
train_y = np.load(sys.argv[2])

epochs = int(sys.argv[3])

# Every client has the same test and validation set
test_x = np.load("../data_preprocessing/all_processed_data/full_test_x.npy")
test_y = np.load("../data_preprocessing/all_processed_data/full_test_y.npy")
val_x = np.load("../data_preprocessing/all_processed_data/full_val_x.npy")
val_y = np.load("../data_preprocessing/all_processed_data/full_val_y.npy")

# CNN model
opt = tf.keras.optimizers.Adam(lr=0.000003)

CNN_client_model = Sequential()
CNN_client_model.add(layers.Input(shape=(train_x.shape[1], train_x.shape[2])))
CNN_client_model.add(layers.Conv1D(64, kernel_size=8, strides=4, activation='relu',
                kernel_regularizer=regularizers.l2(0.0002)))
CNN_client_model.add(layers.Conv1D(32, kernel_size=4, strides=2, activation='relu',
                kernel_regularizer=regularizers.l2(0.0002)))
CNN_client_model.add(layers.Flatten())
CNN_client_model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0002)))
CNN_client_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0002)))
CNN_client_model.add(Dense(1, activation='linear'))
CNN_client_model.summary()

CNN_client_model.compile(optimizer=opt, loss='huber', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

model_weight_template = []
for k in CNN_client_model.get_weights():
    model_weight_template.append(k.shape)

fl.client.start_numpy_client(server_address="localhost:8080", client=BatteryClient(CNN_client_model, 
                                                                                   test_x, 
                                                                                   test_y, 
                                                                                   train_x, 
                                                                                   train_y, 
                                                                                   val_x, 
                                                                                   val_y, 
                                                                                   model_weight_template,
                                                                                   epochs))
