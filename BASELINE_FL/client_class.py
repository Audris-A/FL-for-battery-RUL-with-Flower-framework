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

from data_processing.nasa_random_data import NasaRandomizedData
from data_processing.prepare_rul_data import RulHandler

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from logging import WARNING, INFO
from flwr.common.logger import log

class BatteryClient(fl.client.NumPyClient):
    def __init__(self, model, test_x, test_y, train_x, train_y, val_x, val_y, model_weight_template, epochs):
        self.model = model
        self.test_x = test_x
        self.test_y = test_y
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.model_weight_template = model_weight_template
        self.epochs = epochs

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_x, 
                       self.train_y, 
                       epochs=self.epochs, 
                       batch_size=32,
                       validation_data=(self.val_x, self.val_y))

        return self.model.get_weights(), len(self.train_x), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        loss, mse, mae, mape, rmse = self.model.evaluate(self.test_x, self.test_y)
        
        return loss, len(self.test_x), {"mse": float(mse), "mse": float(mse), "mae": float(mae), "mape": float(mape), "rmse": float(rmse)}