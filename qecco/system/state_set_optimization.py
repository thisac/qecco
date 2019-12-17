import sys

import numpy as np
import nlopt

from .nlopt_wrapper import nlopt_optimize
from .cost_functions import build_cost

# NOTE: only for testing
# np.random.seed(42)


def optimize_state_set_lossless(
        encoding,
        numLayers=None,
        system_list=[...],
        systems_to_optimize=[True],
        **kwargs
        ):

    for system in system_list:
        if numLayers is None or len(system_list) > 1:
            numLayers_to_use = system.parameters["num_of_layers"][0]
            numLayers = None
        else:
            numLayers_to_use = numLayers

        get_num_phases(system, numLayers_to_use)

        # validMethods = ["clements", "reck"]
        # if system.parameters["method"] not in validMethods:
        #     msg = "Error: method not in {}".format(validMethods)
        #     raise ValueError(msg)

        # # number of unitaries = number of nonlinear layers + 1
        # # number of free parameters per unitary = m*(m-1) for reck

        # if numLayers is None or len(system_list) > 1:
        #     numLayers_to_use = system.parameters["num_of_layers"][0]
        #     numLayers = None
        # else:
        #     numLayers_to_use = numLayers
        # if system.parameters["method"] == "clements":
        #     ppl = system.parameters["m_ancillae"] * system.parameters["m_ancillae"]
        #     numPhases = ppl * numLayers_to_use
        #     system.parameters.update({"num_phases": numPhases})
        # elif system.parameters["method"] == "reck":
        #     numUs = numLayers_to_use + 1   # HELP: What is numUS and why?
        #     ppl = system.parameters["m_ancillae"] * (system.parameters["m_ancillae"] - 1)
        #     numPhases = ppl * numUs
        #     system.parameters.update({"num_phases": numPhases})

        cost = build_cost(
            encoding=encoding,
            numLayers=numLayers,
            system_list=system_list,
            systems_to_optimize=systems_to_optimize,
            )

    lowerBounds = np.array([])
    upperBounds = np.array([])
    guess = np.array([])
    idx = np.where(systems_to_optimize)[0]
    system = system_list[idx[0]]
    for i in idx:
        lowerBounds = np.concatenate((lowerBounds, -4 * np.pi * np.ones((system_list[i].parameters["num_phases"], ))))
        upperBounds = np.concatenate((upperBounds, 4 * np.pi * np.ones((system_list[i].parameters["num_phases"], ))))
        guess = np.concatenate((guess, 2 * np.pi * np.random.random((system_list[i].parameters["num_phases"], )) - np.pi))

    # print(type(system).__name__)
    # print(system.parameters["num_of_layers"])
    # print(system.parameters["num_phases"])

    # run the optimization
    # lowerBounds = -4 * np.pi * np.ones((system.parameters["num_phases"], ))
    # upperBounds = 4 * np.pi * np.ones((system.parameters["num_phases"], ))
    optKwargs = {
        # 'stopval': 1e-5,
        'ftolRel': 1e-8,
        'xtolRel': 1e-8,
        # 'maxTime': 60,    #  1m
        # 'maxTime': 300,   #  5m
        # 'maxTime': 1800,  # 30m
        # 'maxTime': 3600,  #  1h
        # 'maxTime': 86400,  # 24h
        # 'guess': np.zeros((system.parameters["num_phases"], ), dtype=float),
        'guess': guess,
        # 'guess': 2 * np.pi * np.random.random((system.parameters["num_phases"], )) - np.pi,  # random uniform guess over -pi -> pi
        # 'guess': 4 * np.pi * np.random.random((system.parameters["num_phases"], )) - 2 * np.pi,  # random uniform guess over -2pi -> 2pi
        # 'guess': 8 * np.pi * np.random.random((system.parameters["num_phases"], )) - 4 * np.pi,  # random uniform guess over -4pi -> 4pi
        'printEvery': system.parameters["print_every"],
        'maxEval': system.parameters["num_of_evaluations"],
        }

    optKwargs.update((key, value) for key, value in kwargs.items() if value is not None)
    opt_algorithm = get_algorithm(str(system.parameters["opt_algorithm"]))
    results = nlopt_optimize(cost, opt_algorithm, lowerBounds, upperBounds, **optKwargs)
    results.update({"fidelity": cost(results["bestX"], return_fidelity=False)})
    # print(cost(results["bestX"], return_fidelity=False))
    # print(cost(results["bestX"], return_fidelity=True))

    if results['caughtError'] is not None:
        print("Warning: optimization returned an error.")
        print(results['caughtError'])
    if system.parameters["verbose"]:
        print("\nRunning time: {:0.02f} seconds".format(results['runTime']))
        print("Number of cost function evaluations: {:d}".format(results['numEvaluations']))
        print("Cost function evaluations/second: {:0.02f}".format(results['numEvaluations'] / results['runTime']))
        print("Optimization result: {}".format(results['returnCodeMessage']))
        print("Best error: {}".format(results['bestError']))

    return results


def get_num_phases(system, numLayers_to_use, return_val=False):
    validMethods = ["clements", "reck"]
    if system.parameters["method"] not in validMethods:
        msg = "Error: method not in {}".format(validMethods)
        raise ValueError(msg)

    # number of unitaries = number of nonlinear layers + 1
    # number of free parameters per unitary = m*(m-1) for reck

    if system.parameters["method"] == "clements":
        ppl = system.parameters["m_ancillae"] * system.parameters["m_ancillae"]
        numPhases = ppl * numLayers_to_use
    elif system.parameters["method"] == "reck":
        numUs = numLayers_to_use + 1   # HELP: What is numUS and why?
        ppl = system.parameters["m_ancillae"] * (system.parameters["m_ancillae"] - 1)
        numPhases = ppl * numUs

    if return_val:
        return numPhases
    else:
        system.parameters.update({"num_phases": numPhases})


def get_algorithm(opt_algorithm):
    default_algorithm = "LBFGS"
    algorithms = {
        "LBFGS": nlopt.LD_LBFGS,
        "BOBYQA": nlopt.LN_BOBYQA,
        "CRS": nlopt.GN_CRS2_LM,
        "MLSL": nlopt.G_MLSL_LDS
    }

    try:
        opt_function = algorithms[opt_algorithm]
    except KeyError:
        print(f"Warning: {opt_algorithm} not found. Using default ({default_algorithm}).")
        input("Press Enter to continue...")
        opt_function = algorithms[default_algorithm]

    return opt_function
