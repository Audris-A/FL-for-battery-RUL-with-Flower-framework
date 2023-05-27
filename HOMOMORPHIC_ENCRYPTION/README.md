Activate the environemnt "he_environment" because it contains
direct changes in the Flower library!!!

The changes made include adapting the conversion functions ndarrays_to_parameters
and parameters_to_ndarrays.

If the package somehow is reinstalled and the changes are lost, see the necessary code below:
  The path to file is (...)/flwr/common/parameter.py
============================
from logging import WARNING, INFO
from flwr.common.logger import log

def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = ndarrays #[ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""

    #return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]
    return [tensor for tensor in parameters.tensors]
======================================