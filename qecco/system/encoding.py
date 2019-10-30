import json
from pathlib import Path
import warnings
from itertools import combinations
import qutip as qt
import sys

import autograd.numpy as np
import bosonic as b

from ..utils import print_array as pa
from ..utils import array_assignment


class Encoding():
    """Density matrices and fock basis encodings

    Load/build density matrices as well as apply loss and transform the density
    matrices between the lossless basis, lossless basis and basises with or
    without ancillae modes/photons. Also load/build simple fock basis encodings
    and translate them to density matrices for use in encoders/decoders.

    Attributes:
        n_photons (int): number of photons (sans ancillae)
        m_modes (int): number of modes (sans ancillae)
        ancillae ([int]): list with ints representing number of photons in each
            mode, e.g. [1,1] means 1 photon each in two ancillae modes
        rhos_lossy ([[complex]]): density matrices with applied loss in the
            lossy fock basis, b.fock.lossy_basis(n_an, m_an), WITH ancillae
            modes
        rhos_lossless ([[complex]]): lossless density matrices in the lossless
            fock basis, b.fock.basis(n_an, m_an), WITH ancillae modes
        rhos_inputs ([[complex]]): density matrices used for input in the
            optimizations. If loss is applied (e.g. for the decoder) the
            rhos_inputs are rhos_lossy + rhos_lossless. Otherwise they are
            density matrix combinations of |01>, |10> or simply rhos_lossless.
        rhos_targets ([[complex]]): target density matrices
        bc_inputs ([[complex]]): the inputs encoded in the pure state fock
            basis
        bc_targets ([[complex]]): the targets encoded in the pure state fock
            basis
        data_in_lossy_basis (bool): Whether the inputs are encoded in the lossy
            basis (True) or not (False). Used when the number of photons differ
            between pure input/output states (e.g. for a qgate without ancillae
            where input one is [0,0] and input two is [0,1]).
        simple_cost (bool): Whether to use the pure state cost function (True)
            or to use the density matrix cost function (False).
    """

    def __init__(self, parameters=None):
        """Initialize all attributes for object

        Parameters:
            paramters (dict): a dictionary with parameters to be set as
            attributes for the object with the set_parameters function. If no
            parameters are passed (None) then all attributes are set to None.
        """

        self.n_photons = None
        self.m_modes = None
        self.ancillae = None

        self.rhos_lossy = None
        self.rhos_lossless = None
        self.rhos_inputs = None
        self.rhos_targets = None

        self.bc_inputs = None
        self.bc_targets = None
        self.data_in_lossy_basis = None
        self.simple_cost = None
        self.verbose = None

        if parameters is not None:
            self.set_parameters(parameters)

    def set_parameters(self, parameters):
        """Set all attributes to corresponding value in parameter dict

        Parameters:
            parameters (dict): a dictionary with parameters to be set as
            attributes for the object. Keys in dict must be included in valid
            parameters to function. All, some or no parameters may be set.
        """

        valid_parameters = ["n_photons", "m_modes", "ancillae", "rhos_lossy",
                            "rhos_lossless", "rhos_targets", "simple_cost",
                            "data_in_lossy_basis"]
        for key in parameters:
            if key not in valid_parameters:
                msg = f"Error: {key} not in {valid_parameters}"
                raise ValueError(msg)

        try:
            self.n_photons = parameters["n_photons"]
        except KeyError:
            pass

        try:
            self.m_modes = parameters["m_modes"]
        except KeyError:
            pass

        try:
            self.ancillae = parameters["ancillae"]
        except KeyError:
            pass

        try:
            self.rhos_lossy = parameters["rhos_lossy"]
        except KeyError:
            pass

        try:
            self.rhos_lossless = parameters["rhos_lossless"]
        except KeyError:
            pass

        try:
            self.rhos_targets = parameters["rhos_targets"]
        except KeyError:
            pass

        try:
            self.simple_cost = parameters["simple_cost"]
        except KeyError:
            pass

        try:
            self.data_in_lossy_basis = parameters["data_in_lossy_basis"]
        except KeyError:
            pass

        try:
            self.verbose = parameters["verbose"]
        except KeyError:
            pass

    def build_bosonic_codes(self, code_name, gen_encoder_inputs=True):
        """ Loads and generates a full set of targets and inputs

        Loads targets, and potentially inputs if needed, for the bosonic codes
        and quantum gates that will the QONN will be optimized for. If needed,
        the function also generates a full set of targets and inputs including
        ancillae modes and photons (that are not necessarily provided in the
        JSON save file).

        Parameters:
            code_name (string): the codes or quantum gate that is loaded
            gen_encoder_inputs (bool): if true, generates the |10> and |01>
                states as inputs for the encoder (not used with decoder), ONLY
                if code is of type "becc".
        """

        codes_dict = self.get_codes(code_name, self.verbose)
        if codes_dict is None:
            raise NameError(f"{code_name} not found.")

        try:
            self.data_in_lossy_basis = codes_dict["lossy_basis"]
        except KeyError:
            pass

        n_an = int(self.n_photons + np.sum(self.ancillae))
        m_an = int(self.m_modes + len(self.ancillae))

        if self.ancillae == [] and self.verbose:
            print(
                "Warning: Be aware that qgates in the lossy basis may alter "
                "number of parameter photons and modes, as well as setting "
                "ancillae to []."
                )
        try:
            assert codes_dict["n_photons"] == self.n_photons
            assert codes_dict["m_modes"] == self.m_modes

        except KeyError:
            warnings.warn("No photons/modes defined in codes.json")

        lossless_basis = np.array(b.fock.basis(self.n_photons, self.m_modes))
        lossless_basis_an = np.array(b.fock.basis(n_an, m_an))
        lossy_basis = np.array(b.fock.lossy_basis(self.n_photons, self.m_modes))
        N = len(lossless_basis_an)

        if codes_dict["type"] == "becc":
            target_1 = np.zeros(N, dtype=complex)
            target_2 = np.zeros(N, dtype=complex)

            for i, el in enumerate(lossless_basis):
                idx = self.rho_idx(el, lossless_basis_an)
                target_1[idx] = codes_dict["targets"][0][i]
                target_2[idx] = codes_dict["targets"][1][i]

            if gen_encoder_inputs:
                input_1 = np.zeros(N, dtype=complex)
                input_2 = np.zeros(N, dtype=complex)

                idx_1 = self.rho_idx([1, 0], lossless_basis_an)
                idx_2 = self.rho_idx([0, 1], lossless_basis_an)

                input_1[idx_1] = input_2[idx_2] = 1

                in_1 = input_1
                in_2 = input_2

                in_3 = (in_1 + in_2) / np.sqrt(2)
                in_4 = (in_1 - in_2) / np.sqrt(2)
                in_5 = (in_1 + 1j * in_2) / np.sqrt(2)
                in_6 = (in_1 - 1j * in_2) / np.sqrt(2)

                expanded_inputs = np.vstack((in_1, in_2, in_3, in_4, in_5, in_6)).T

            ta_1 = target_1
            ta_2 = target_2

            ta_3 = (ta_1 + ta_2) / np.sqrt(2)
            ta_4 = (ta_1 - ta_2) / np.sqrt(2)
            ta_5 = (ta_1 + 1j * ta_2) / np.sqrt(2)
            ta_6 = (ta_1 - 1j * ta_2) / np.sqrt(2)

            expanded_targets = np.vstack((ta_1, ta_2, ta_3, ta_4, ta_5, ta_6)).T

        elif codes_dict["type"] == "qgate":
            # If codes are written in the lossy basis (to include different
            # numbers of photons in different states, e.g. |01> and |11>)
            # transfer the lossy basis vectors into lossless basis vectors
            # with ancillae instead. Then transfer the ancillae photons/modes
            # to non-ancillae photons/modes, i.e. set ancillae = []. Otherwise
            # the cost function will not be happy.
            if self.data_in_lossy_basis:
                num_of_io_states = len(codes_dict["inputs"])
                input_states_an = np.zeros((num_of_io_states, N), dtype=complex)
                target_states_an = np.zeros((num_of_io_states, N), dtype=complex)
                for i, el in enumerate(lossy_basis):
                    idx = self.rho_idx(el, lossless_basis_an)
                    for j in range(len(input_states_an)):
                        input_states_an[j][idx] = codes_dict["inputs"][j][i]
                        target_states_an[j][idx] = codes_dict["targets"][j][i]

                self.n_photons = n_an
                self.m_modes = m_an
                self.ancillae = []
            else:
                assert self.ancillae == []
                input_states_an = codes_dict["inputs"]
                target_states_an = codes_dict["targets"]

            expanded_inputs = np.array(input_states_an, dtype=complex).T
            expanded_targets = np.array(target_states_an, dtype=complex).T

        else:
            raise TypeError("Type not found.")

        if gen_encoder_inputs:
            self.bc_inputs = expanded_inputs
        self.bc_targets = expanded_targets

    def apply_loss_to_rhos(self, loss_function, **loss_kwargs):
        n_an = int(self.n_photons + np.sum(self.ancillae))
        m_an = int(self.m_modes + len(self.ancillae))

        lossy_basis = b.fock.lossy_basis(self.n_photons, self.m_modes)
        lossy_basis_an = b.fock.lossy_basis(n_an, m_an)

        rhos_lossy = []
        for rho in self.rhos_targets:
            # If not in lossy basis => expand to lossy basis;
            # if it is already in the lossy basis, nothing will happen
            rho_expanded = self.expand_rho_to_lossy(rho)

            rho_lossy = loss_function(rho_expanded, **loss_kwargs)

            rho_lossy_an = self.embed(rho_lossy, lossy_basis, lossy_basis_an)
            rhos_lossy.append(rho_lossy_an)

        self.rhos_lossy = rhos_lossy
        self.rhos_inputs = self.rhos_lossy
        return self

    def apply_loss_to_single_rho(self, rho, loss_function, **loss_kwargs):
        n_an = int(self.n_photons + np.sum(self.ancillae))
        m_an = int(self.m_modes + len(self.ancillae))

        lossy_basis = b.fock.lossy_basis(self.n_photons, self.m_modes)
        lossy_basis_an = b.fock.lossy_basis(n_an, m_an)
        rho_expanded = self.expand_rho_to_lossy(rho)

        rho_lossy = loss_function(rho_expanded, **loss_kwargs)
        rho_lossy_an = self.embed_for_autograd(rho_lossy, lossy_basis, lossy_basis_an)

        return rho_lossy_an

    def embed_for_autograd(self, rho, basis_1, basis_2):
        m = len(basis_1[0])
        done_pairs = []
        rho_out = []
        for i, row in enumerate(basis_2):
            try:
                idx_i = basis_1.index(row[:m])
            except ValueError:
                idx_i = None
            row_out = []
            for j, col in enumerate(basis_2):
                try:
                    idx_j = basis_1.index(col[:m])
                except ValueError:
                    idx_j = None

                if (row[:m], col[:m]) not in done_pairs:
                    if idx_i is not None and idx_j is not None:
                        row_out.append(rho[idx_i][idx_j])
                    else:
                        row_out.append(0)
                    done_pairs.append((row[:m], col[:m]))
                else:
                    row_out.append(0)

            rho_out.append(row_out)
        return np.array(rho_out)

    def embed(self, rho, basis_1, basis_2):
        elements_list = self.get_elements(rho)
        rho_an = np.zeros((len(basis_2), len(basis_2)), dtype=complex)

        for element in elements_list:
            state_1 = basis_1[element[0]]
            state_2 = basis_1[element[1]]

            rho_1 = self.rho_idx(state_1, basis_2)
            rho_2 = self.rho_idx(state_2, basis_2)

            if rho_1 is not None and rho_2 is not None:
                # rho_an[rho_1][rho_2] = element[2]
                rho_an = array_assignment(rho_an, element[2], (rho_1, rho_2))
        return rho_an

    def expand_rho_to_lossy(self, rho, ancillae=False):
        n_an = int(self.n_photons + np.sum(self.ancillae))
        m_an = int(self.m_modes + len(self.ancillae))
        if ancillae:
            N = len(b.fock.lossy_basis(n_an, m_an))
        else:
            N = len(b.fock.lossy_basis(self.n_photons, self.m_modes))

        zeroarr_1 = np.zeros((len(rho), N - len(rho)))
        zeroarr_2 = np.zeros((N - len(rho), N))

        rho_1 = np.concatenate((rho, zeroarr_1), axis=1)
        rho_2 = np.concatenate((rho_1, zeroarr_2), axis=0)

        return rho_2

    def targets_to_lossless(self, new_targets):
        """ Transforms target density matrices to lossy density matrices
        """
        n_an = int(self.n_photons + np.sum(self.ancillae))
        m_an = int(self.m_modes + len(self.ancillae))

        lossless_basis = b.fock.basis(self.n_photons, self.m_modes)
        lossy_basis_an = b.fock.lossy_basis(n_an, m_an)
        new_lossless = self.embed_for_autograd(new_targets, lossless_basis, lossy_basis_an)
        return new_lossless

    def lossless_to_targets(self, rho_lossless):
        n_an = int(self.n_photons + np.sum(self.ancillae))
        m_an = int(self.m_modes + len(self.ancillae))

        mask = [np.sum(x) == self.n_photons
                for x in b.fock.lossy_basis(n_an, self.m_modes)]

        for j in range(len(self.ancillae)):
            rho_lossless = b.density.delete_mode(rho_lossless, n_an, m_an - j, -1)

        rho_lossless = rho_lossless[mask, :]
        rho_lossless = rho_lossless[:, mask]
        return rho_lossless

    def update_targets(self, new_targets):
        rhos_lossless = []
        rhos_targets = []
        for rho in new_targets:
            rhos_lossless.append(self.targets_to_lossless(rho))
            rhos_targets.append(rho)

        if self.rhos_lossy is not None:
            self.rhos_lossy = None
            self.rhos_inputs = None

        self.rhos_lossless = rhos_lossless
        self.rhos_targets = rhos_targets

    def pure_to_rho(self, gen_encoder_inputs):
        """Translate the pure state encoding into density matrices.

        Create the six different density matrices (see below) in the lossless
        fock basis, b.fock.basis(n_an, m_an), WITH ancillae modes/photon, and
        save them into object attributes.

        |0_L>
        |1_L>
        |0_L> +  |1_L> / √2
        |0_L> -  |1_L> / √2
        |0_L> + i|1_L> / √2
        |0_L> - i|1_L> / √2
        """

        n_an = int(self.n_photons + np.sum(self.ancillae))
        m_an = int(self.m_modes + len(self.ancillae))

        if gen_encoder_inputs:
            rhos_inputs = []
            for pure_vector in self.bc_inputs.T:
                pv = pure_vector.reshape(-1, 1)
                rho_in = pv @ np.conj(pv).T
                rho_in = self.expand_rho_to_lossy(rho_in, ancillae=True)

                rhos_inputs.append(rho_in)

        mask = [np.sum(x) == self.n_photons for x in b.fock.lossy_basis(n_an, self.m_modes)]
        rhos_lossless = []
        rhos_targets = []
        for pure_vector in self.bc_targets.T:
            pv = pure_vector.reshape(-1, 1)
            rho_ta = pv @ np.conj(pv).T
            rho_ll = self.expand_rho_to_lossy(rho_ta, ancillae=True)

            rhos_lossless.append(rho_ll)

            rho_ta = np.copy(rho_ll)
            for j in range(len(self.ancillae)):
                rho_ta = b.density.delete_mode(rho_ta, n_an, m_an - j, -1)

            rho_ta = rho_ta[mask, :]
            rho_ta = rho_ta[:, mask]

            rhos_targets.append(rho_ta)

        for i in range(len(rhos_targets)):
            if gen_encoder_inputs:
                assert np.isclose(np.trace(rhos_inputs[i]), 1)
                assert np.prod(np.isclose(rhos_inputs[i], np.conj(rhos_inputs[i]).T))

            assert np.isclose(np.trace(rhos_lossless[i]), 1)
            assert np.prod(np.isclose(rhos_lossless[i], np.conj(rhos_lossless[i]).T))
            assert np.isclose(np.trace(rhos_targets[i]), 1)
            assert np.prod(np.isclose(rhos_targets[i], np.conj(rhos_targets[i]).T))

        if gen_encoder_inputs:
            self.rhos_inputs = rhos_inputs
        else:
            self.rhos_inputs = rhos_lossless
        self.rhos_lossless = rhos_lossless
        self.rhos_targets = rhos_targets

    def reset(self):
        self.n_photons = None
        self.m_modes = None
        self.ancillae = None

        self.rhos_lossy = None
        self.rhos_lossless = None
        self.rhos_inputs = None
        self.rhos_targets = None

        self.bc_inputs = None
        self.bc_targets = None

    @staticmethod
    def rho_idx(val, basis):
        end = len(val)
        for i, row in enumerate(basis):
            if str(list(row[:end])) == str(list(val)):
                return i

    @staticmethod
    def get_elements(rho):
        elements_list = []
        for i, row in enumerate(rho):
            for j, el in enumerate(row):
                if el:
                    elements_list.append([i, j, el])
        return elements_list

    @staticmethod
    def find_subarray(array, subarray):
        for i in range(len(array)):
            if array[i:i + len(subarray)] == subarray:
                return i, i + len(subarray)
        warnings.warn("subarray not found in array")
        return None

    @staticmethod
    def get_codes(code_name, verbose=True):
        codes = None
        custom_code_path = Path.cwd() / "codes.json"
        script_path = Path(__file__).parent / "codes.json"
        try:
            with open(custom_code_path) as read_file:
                codes = json.load(read_file)[code_name]
                if verbose:
                    print(f"Found {code_name} in user codes.json")
        except FileNotFoundError:
            try:
                with open(script_path) as read_file:
                    codes = json.load(read_file)[str(code_name)]
                    if verbose:
                        print(f"Found {code_name} in codes.json")
            except KeyError:
                if verbose:
                    print(f"Code {code_name} not found in codes.json")
        except KeyError:
            if verbose:
                print(f"Code {code_name} not found in user codes.json")

        return codes
