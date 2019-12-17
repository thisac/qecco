import sys
import time

import autograd.numpy as np
from autograd import grad as agrad
import bosonic as b

from ..utils import print_array as pa
from ..utils import timeit


# TODO: Look into...
# non-ancillae mode qgates in the lossy basis do not work with the cost
# function since all lossy modes (i.e. the ones with less than the full number
# of photons in) are lost, thus loosing needed information.

# @timeit
def build_cost(
        encoding,
        numLayers=None,
        system_list=[...],
        systems_to_optimize=[True],
        ):
    """Build the cost function for the optimizations including full loss

    Parameters:
        encoding (encoding obj):
        numLayers (int):
        system_list (list[system obj]):
        systems_to_optimize (list[bool]):

    Returns:

    """

    if encoding.simple_cost:
        lossy = False
    else:
        lossy = True

    # HELP: Does phi=None or phi=np.pi actually make a difference?
    builds_list = []
    for system in system_list:
        numLayersOld = numLayers
        if numLayers is None:
            numLayers = system.parameters["num_of_layers"][0]
        build_system, _ = b.qonn.build_system_function(
            system.parameters["n_ancillae"],
            system.parameters["m_ancillae"],
            numLayers,
            phi=system.parameters["phi"],
            method=system.parameters["method"],
            lossy=lossy,
            )
        builds_list.append(build_system)
        numLayers = numLayersOld

    # Build all subsystems that are NOT to be optimized over with prior thetas
    S_list = []
    SH_list = []
    for i, system in enumerate(system_list):
        if systems_to_optimize[i] is True:
            S_list.append(None)
            SH_list.append(None)
        else:
            assert type(system).__name__ == "System"
            try:
                S_list.append(builds_list[i](system.results["bestX"]))
            except TypeError:
                guess = 2 * np.pi * np.random.random((system.parameters["num_phases"], )) - np.pi
                S_list.append(builds_list[i](guess))

            SH_list.append(np.conj(S_list[-1].T))

    idxs = np.where(systems_to_optimize)[0]

    if encoding.simple_cost:
        if encoding.bc_inputs is None or encoding.bc_targets is None:
            msg = "Must add a set of bosonic codes"
            raise NameError(msg)

        assert len(system_list) == 1

        input_S = np.copy(encoding.bc_inputs)
        output_S = encoding.bc_targets

        numStates = input_S.shape[1]
        outputH = np.conj(output_S.T)

        def cost_fn(x):
            S = build_system(x)
            cost = 0
            for i in range(numStates):
                cost_part = (
                    1.0 - np.abs(np.dot(outputH[i, :], np.dot(S, input_S[:, i])))**2
                    )
                cost = cost + cost_part
            return cost / numStates
    else:
        if encoding.rhos_inputs is None or encoding.rhos_targets is None:
            msg = "density matrices are not defined"
            raise NameError(msg)
        # tt = []

        # TODO: Check what takes time to run.
        # @timeit
        def cost_fn(x, return_fidelity=False):
            new_x = []
            pre = 0
            post = 0
            for i, system in enumerate(system_list):
                if systems_to_optimize[i]:
                    post = post + system.parameters["num_phases"]
                    new_x.append(x[pre:post])
                    pre = system.parameters["num_phases"]
            x = new_x

            for i, idx in enumerate(idxs):
                S_list[idx] = builds_list[idx](x[i])
                SH_list[idx] = np.conj(S_list[idx].T)

            cost = 0
            min_fid = 1
            for i, single_rho in enumerate(encoding.rhos_inputs):
                rho = np.copy(single_rho)

                for j, system in enumerate(system_list):
                    if len(system_list) > 1:

                        if system.system_type == "decoder" and j > 0:
                            pre_sys = system_list[j - 1]
                            rho_ta = pre_sys.encoding.lossless_to_targets(rho)
                            rho = system.apply_loss(rho=rho_ta, verbose=False)

                        rho = np.dot(np.dot(S_list[j], rho), SH_list[j])
                        rho = reset_ancillae(rho, system)
                    else:
                        rho = np.dot(np.dot(S_list[j], rho), SH_list[j])

                rho = system_list[-1].encoding.lossless_to_targets(rho)

                fidelity = np.abs(np.trace(np.dot(encoding.rhos_targets[i], rho)))
                if fidelity < min_fid:
                    min_fid = fidelity
                cost = cost + (1 - fidelity * np.abs(np.trace(rho)))

            if return_fidelity:
                return min_fid
            else:
                return cost / len(encoding.rhos_inputs)

        # # @timeit
        # def small_cost_fn(x):
        #     S_list = build_system(x)
        #     SH_list = np.conj(S_list.T)

        #     cost = 0
        #     for i, single_rho in enumerate(encoding.rhos_inputs):
        #         rho = np.dot(np.dot(S_list, single_rho), SH_list)

        #         rho = system_list[0].encoding.lossless_to_targets(rho)
        #         fidelity = np.abs(np.trace(np.dot(encoding.rhos_targets[i], rho)))
        #         cost = cost + (1 - fidelity)

        #     return cost / len(encoding.rhos_inputs)

    cost_grad = agrad(cost_fn)
    # cost_grad = agrad(small_cost_fn)

    # @timeit
    def cost_wrapper(x, grad=np.array([]), return_fidelity=False):
        if grad.size > 0:
            grad[:] = cost_grad(x)
        return cost_fn(x, return_fidelity)
        # return small_cost_fn(x)

    return cost_wrapper


def reset_ancillae(rho, system):
    rho_ta = system.encoding.lossless_to_targets(rho)
    return system.encoding.targets_to_lossless(rho_ta)
