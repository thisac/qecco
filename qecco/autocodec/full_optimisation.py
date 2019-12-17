import numpy as np

from qecco import System
from ..system.state_set_optimization import optimize_state_set_lossless as opt


if __name__ == "__main__":
    parameters_encoder = {
        "num_of_layers": 20,
        "print_every": 1,
        "ancillae": [0],
        "simple_cost": False,
        "opt_algorithm": "LBFGS",
        "num_of_evaluations": 2000,
        "save_data": False,
        }
    parameters_decoder = {
        "num_of_layers": 200,
        "print_every": 1,
        "ancillae": [1, 1],
        "opt_algorithm": "LBFGS",
        "num_of_evaluations": 2000,
        "save_data": False,
        }

    en = System("encoder", parameters_encoder)
    de = System("decoder", parameters_decoder)

results_en = opt(
    self.encoding,
    system_list=[self.encoder, self.decoder],
    systems_to_optimize=[True, False],
    guess=guess_en,
    **kwargs,
    )

