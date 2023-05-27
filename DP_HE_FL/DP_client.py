# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for configuring a Flower client for DP."""


import copy
from typing import Dict, Tuple

import numpy as np

from flwr.client.numpy_client import NumPyClient
from flwr.common.dp import add_gaussian_noise, clip_by_l2
from flwr.common.typing import Config, NDArrays, Scalar

from logging import WARNING, INFO
from flwr.common.logger import log

from Pyfhel import Pyfhel, PyPtxt, PyCtxt

class DPFedAvgNumPyClient(NumPyClient):
    """Wrapper for configuring a Flower client for DP."""

    def __init__(self, client: NumPyClient, model_weight_template) -> None:
        super().__init__()
        self.client = client
        self.model_weight_template = model_weight_template

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return self.client.get_properties(config)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.client.get_parameters(config)

    # self, parameters: NDArrays, config: Dict[str, Scalar]
    def fit(
        self, parameters, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        
        HE = Pyfhel() # Empty creation
        HE.load_context("./context")
        HE.load_public_key("./pub.key")
        HE.load_secret_key("./sec.key")
        HE.load_rotate_key("./rotate.key")

        # data preprocessing operations...
        decrypted_weights = []
        offset_wit = 0

        # Reading the received bytes and transforming them to HE cyphertexts
        #   which then are decrypted and reshaped to the original model shape
        for model_weight_shape_obj in self.model_weight_template:

            temp_shape_dproduct = np.prod(np.array(model_weight_shape_obj))

            # The Polynomial modulus degree (n in the HE context generation) is the limit
            #   for the length of the ndarray, so if we have larger arrays they have to be
            #   divided into smaller ones relative to the (2^15)/2
            if temp_shape_dproduct > 16384:
                large_tensor = np.array([])
                
                for lto in range(0, 8):
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

        original_params = copy.deepcopy(decrypted_weights)

        # Getting the updated model from the wrapped client
        updated_params, num_examples, metrics = self.client.fit(decrypted_weights, config)

        # Update = updated model - original model
        update = [np.subtract(x, y) for (x, y) in zip(updated_params, original_params)]

        if "dpfedavg_clip_norm" not in config:
            raise Exception("Clipping threshold not supplied by the server.")
        if not isinstance(config["dpfedavg_clip_norm"], float):
            raise Exception("Clipping threshold should be a floating point value.")

        # Clipping
        update, clipped = clip_by_l2(update, config["dpfedavg_clip_norm"])
        
        if "dpfedavg_noise_stddev" in config:
            if not isinstance(config["dpfedavg_noise_stddev"], float):
                raise Exception(
                    "Scale of noise to be added should be a floating point value."
                )
            # Noising
            # Added a fix for complex number creation
            #  in the default code
            if config["dpfedavg_noise_stddev"] < 0:
                config["dpfedavg_noise_stddev"] = -config["dpfedavg_noise_stddev"]
            update = add_gaussian_noise(update, config["dpfedavg_noise_stddev"])

        for i, _ in enumerate(original_params):
            updated_params[i] = original_params[i] + update[i]

        # Calculating value of norm indicator bit, required for adaptive clipping
        if "dpfedavg_adaptive_clip_enabled" in config:
            if not isinstance(config["dpfedavg_adaptive_clip_enabled"], bool):
                raise Exception(
                    "dpfedavg_adaptive_clip_enabled should be a boolean-valued flag."
                )
            metrics["dpfedavg_norm_bit"] = not clipped
        
        # Reshaping the tensors to 1 dimensional numpy ndarray and dividing
        #  larger tensors relative to the HE context n attribute (see more comments below and see HE context gen)
        #  then encrypting each reshaped tensor and transforming them to bytes for sending them to the server.
        model_encrypted_weights_as_bytes = []
        for k in updated_params:
            vector_length = np.prod(np.array(k.shape))

            reshaped_weight = k.reshape(vector_length,)
            
            # The Polynomial modulus degree (n in the HE context generation) is the limit
            #   for the length of the ndarray, so if we have larger arrays they have to be
            #   divided into smaller ones relative to the (2^15)/2
            #   The large tensor was divided into 8 (the iterator length) ndarrays with dimension 15744
            #   which was. This should be defined dynamically, but is hardcoded for now
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

        return model_encrypted_weights_as_bytes, len(model_encrypted_weights_as_bytes), metrics

    # self, parameters: NDArrays, config: Dict[str, Scalar]
    def evaluate(
        self, parameters, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        return self.client.evaluate(parameters, config)
