import sys

from qecco import Autocodec, System
from qecco.utils import print_array as pa

# NOTE: Same number of layers, ancillae, photons and modes are required
parameters_decoder = {
    "num_of_layers": [20],
    "print_every": 1,
    "ancillae": [0],
    # "opt_algorithm": "BOBYQA",
    "opt_algorithm": "LBFGS",
    "num_of_evaluations": 10,
    }
parameters_encoder = {
    "num_of_layers": [20],
    "print_every": 1,
    "ancillae": [0],
    "simple_cost": False,
    # "opt_algorithm": "BOBYQA",
    "opt_algorithm": "LBFGS",
    "num_of_evaluations": 10,
}
# parameters_decoder.update({"save_data": True})
# parameters_encoder.update({"save_data": True})
de = System("decoder", parameters_decoder)
en = System("encoder", parameters_encoder)

# ##### Show output error ######
# en.build_rhos().optimize()
# de.build_rhos().apply_loss()
# for rho in de.encoding.rhos_inputs:
#     pa(rho)
# sys.exit()
# de.optimize()
# print("Printing encoder/decoder outputs")
# for i, rho in enumerate(de.get_output()):
#     print("Rho", i)
#     pa(rho)

# print("Printing autocodec outputs")
# ac = Autocodec(en, de, save_data=False)
# ac.output_error(de.results["bestX"])
# ##############################

# ####### Run autocodec ########
ac = Autocodec(en, de, save_data=True)
# ac.optimize(epochs=20, pre_opt=(10, 10))
ac.optimize(epochs=200, pre_opt=False)
# ##############################
