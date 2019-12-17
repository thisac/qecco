# TODO: Add save function rhos/bosonic codes in JSON

# TODO: Create a print-out of the system (with inputs, outputs, and ancillae); curses?!
# TODO: Expand main to ask for all important parameter inputs
# TODO: Add nice-exit when pressing ctrl-c
# TODO: Follow same convention for all assertions (e.g. in system.get_output_rhos)

# TODO: Add parallelization

# IDEA: Check what happens if input non-loss states or single-photon-loss states
# E.g input (|40> + |04>) / sqrt(2) and |30> should give same (see photo on cellphone 901210)
# IDEA: Try dropout (remove layers during training)

import sys
from .optimize import optimize, opt_type_inputs

if __name__ == "__main__":
    if len(sys.argv) == 2:
        opt_type = sys.argv[1]
        optimize(opt_type)
    else:
        opt_type_inputs()
