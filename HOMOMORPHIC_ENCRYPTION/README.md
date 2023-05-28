This method requires changes to the flower lib.

The changes to be made include adapting the conversion functions ndarrays_to_parameters
and parameters_to_ndarrays.

The path to file is (...)/flwr/common/parameter.py

```
from logging import WARNING, INFO
from flwr.common.logger import log

def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Send the tensors as is."""
    tensors = ndarrays #[ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Send the tensors as is"""

    #return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]
    return [tensor for tensor in parameters.tensors]
```