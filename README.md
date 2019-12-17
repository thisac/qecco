# Bosonic code optimizer for quantum error correction

Simulation of Quantum Optical Neural Networks (QONNs) using a fock-basis encoding for input/output states. QONNs use layers of unitary operations together with nonlinearities to perform arbitrary quantum operations on the input quantum states.

The network can be trained to perform various quantum gates such as the CNOT gate and the Toffoli gate, among others, as well as for error correction. The encoder object encodes a 0 and 1 into a set of logical codes that can be more robust to noise than the naive choice, and the decoder can apply noise to the logical states and then train a network to correct for the same loss. [1]

## Installation:

Clone or download the repository and install via pip:

    $ pip install -e ./ --user

## Usage:
Begin by importing the module.

```python
from qecco import System
```

Either create an encoder-object or a decoder-object. Default parameters are chosen if no parameters are given.

```python
params_encoder =  {
    "num_of_layers": [20],
    "num_of_tests": 1,
    "opt_algorithm": "LBFGS",
    "method": "reck",
    "num_of_evaluations": 1000,
    "ancillae": [0],
    "n_photons": 4,
    "m_modes": 2,
    "code_name": "n4m2",
    "gen_encoder_inputs": True,

    "print_every": 10,
    "verbose": True,
}

params_decoder = {
    "num_of_layers": [180],
    "num_of_tests": 1,
    "opt_algorithm": "LBFGS",
    "method": "reck",
    "loss_function": "bosonic_density_loss",
    "loss_kwargs": {
        "n": 4,
        "m": 2,
        "eta": 0.05
    },
    "num_of_evaluations": 2000,
    "ancillae": [1, 1],
    "n_photons": 4,
    "m_modes": 2,
    "code_name": "n4m2",
    "gen_encoder_inputs": False,

    "print_every": 10,
    "verbose": True,
}

en = System("encoder", parameters=params_encoder)
de = System("decoder", parameters=params_decoder)
```

Next, load and set up the inputs and targets of the optimization, apply loss to the decoder inputs, and then optimize.

```python
en.build_codes()
de.build_codes().apply_loss()

en.optimize()
de.optimize()
```

Both optimizations can also be run on one line as `en.build_codes().optimize()` and `de.build_codes().apply_loss().optimize()` or completely separate. When the optimizations are done, the results can be found in the results object as a dictionary. Use the `dict.keys()` function to print the different result types.

```python
en.results()
de.results()
```

Default parameters are stored in [parameters.json](qecco/system/parameters.json), the different inputs and targets are stored in [codes.json](qecco/system/codes.json).

*More documentation to come some day... hopefully sooner than later.*

[1]: Quantum optical neural networks, Gregory R. Steinbrecher, Jonathan P. Olson, Dirk Englund & Jacques Carolan (https://www.nature.com/articles/s41534-019-0174-7)