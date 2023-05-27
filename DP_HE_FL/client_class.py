import flwr as fl
import tensorflow as tf
import numpy as np
import sys
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Masking

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.dp import add_gaussian_noise

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
        
        HE = Pyfhel() # Empty creation
        HE.load_context("./context")
        HE.load_public_key("./pub.key")
        HE.load_secret_key("./sec.key")
        HE.load_rotate_key("./rotate.key")

        model_encrypted_weights_as_bytes = []

        # Reshaping the tensors to 1 dimensional numpy ndarray and dividing
        #  larger tensors relative to the HE context n attribute (see more comments below and see HE context gen)
        #  then encrypting each reshaped tensor and transforming them to bytes for sending them to the server.
        for k in self.model.get_weights():
            try:
                vector_length = np.prod(np.array(k.shape))

                reshaped_weight = k.reshape(vector_length,)
                
                # The Polynomial modulus degree (n in the HE context generation) is the limit
                #   for the length of the ndarray, so if we have larger arrays they have to be
                #   divided into smaller ones relative to the (2^15)/2
                #   The large tensor was divided into 8 (the iterator length) ndarrays with dimension 15744. 
                #   This should be defined dynamically, but is hardcoded for now
                if reshaped_weight.shape[0] > 16384:
                    reshaped_weight = reshaped_weight.reshape((8, 15744))
                    
                    for large_weight in reshaped_weight:
                        ptxt_x = HE.encodeFrac(large_weight.astype(np.float64))
                        encrypted_weight = HE.encryptPtxt(ptxt_x)

                        encrypted_weight_as_bytes = encrypted_weight.to_bytes()

                        model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)

                else:
                    ptxt_x = HE.encodeFrac(reshaped_weight.astype(np.float64))
                    encrypted_weight = HE.encryptPtxt(ptxt_x)

                    encrypted_weight_as_bytes = encrypted_weight.to_bytes()

                    model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)
            except Exception as e:
                log(WARNING, str(e))
                log(WARNING, "nFAIL")

        c_b = PyCtxt(pyfhel=HE, bytestring=model_encrypted_weights_as_bytes[0])
        plaintext_weight = HE.decryptFrac(c_b)
        
        return model_encrypted_weights_as_bytes

    def fit(self, parameters, config):
        
        # Parameters already deciphered in the DP wrapper
        self.model.set_weights(parameters)

        history = self.model.fit(self.train_x, self.train_y, epochs=self.epochs, batch_size=32, validation_data=(self.val_x, self.val_y))
        
        return self.model.get_weights(), len(self.train_x), {}


    def evaluate(self, parameters, config):

        HE = Pyfhel() # Empty creation
        HE.load_context("./context")
        HE.load_public_key("./pub.key")
        HE.load_secret_key("./sec.key")
        HE.load_rotate_key("./rotate.key")

        # Reading the received bytes and transforming them to HE cyphertexts
        #   which then are decrypted and reshaped to the original model shape
        decrypted_weights = []
        offset_wit = 0
        for model_weight_shape_obj in self.model_weight_template:

            temp_shape_dproduct = np.prod(np.array(model_weight_shape_obj))

            # The Polynomial modulus degree (n in the HE context generation) is the limit
            #   for the length of the ndarray, so if we have larger arrays they have to be
            #   divided into smaller ones relative to the (2^15)/2
            if temp_shape_dproduct > 16384:
                large_tensor = np.array([])
                for lto in range(0,8):
                    encr_bytes = parameters[offset_wit]
                    c_b = PyCtxt(pyfhel=HE, bytestring=encr_bytes)
                    plaintext_weight = HE.decryptFrac(c_b)

                    # The large tensor was divided into 8 (the iterator length) ndarrays with dimension 15744
                    #  This should be defined dynamically, but is hardcoded for now
                    plaintext_weight = plaintext_weight[:15744]

                    large_tensor = np.concatenate((large_tensor, plaintext_weight))
                    offset_wit += 1

                large_tensor = large_tensor.reshape(model_weight_shape_obj)
                decrypted_weights.append(large_tensor)
            else:
                encr_bytes = parameters[offset_wit]
                c_b = PyCtxt(pyfhel=HE, bytestring=encr_bytes)
                plaintext_weight = HE.decryptFrac(c_b)

                plaintext_weight = plaintext_weight[:temp_shape_dproduct]

                decrypted_weights.append(plaintext_weight.reshape(model_weight_shape_obj))
                offset_wit += 1
        
        self.model.set_weights(decrypted_weights)

        loss, mse, mae, mape, rmse = self.model.evaluate(self.test_x, self.test_y)

        return loss, len(self.test_x), {"mse": float(mse), "mse": float(mse), "mae": float(mae), "mape": float(mape), "rmse": float(rmse)}