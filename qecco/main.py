# TODO: Add save function rhos/bosonic codes in JSON

# TODO: Get the "autocodec" to work
# TODO: Make decoder able to test over different eta

# TODO: Create a print-out of the system (with inputs, outputs, and ancillae); curses?!
# TODO: Expand main to ask for all important parameter inputs
# TODO: Reorganize the functions
# TODO: Add nice-exit when pressing ctrl-c
# TODO: Follow same convention for all assertions (e.g. in system.get_output_rhos)

# TODO: Add parallelization

import sys
from .optimize import optimize, opt_type_inputs

if __name__ == "__main__":
    if len(sys.argv) == 2:
        opt_type = sys.argv[1]
        optimize(opt_type)
    else:
        opt_type_inputs()
