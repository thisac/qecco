import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

from ..system.state_set_optimization import optimize_state_set_lossless as opt
from ..system.cost_functions import build_cost
from ..system.encoding import Encoding
from ..utils import print_array as pa
from .. import utils as u

warnings.simplefilter(action='ignore', category=FutureWarning)


# TODO: Do not update targets if exiting with (maxEval (???)) Generic Failure
# TODO: Save data from autocodec in a nice fashion
# TODO: Allow for different number of layers, photons, modes and ancillae
class Autocodec:
    """ Encoder/decoder full system object

    Attributes:
        encoder (system obj): encoder object to be used
        decoder (system obj): decoder object to be used
        save_data (bool): whether to save all data (True) or not (False)
    """

    def __init__(self, encoder, decoder, save_data=False, save_folder=None):
        self.encoder = encoder
        self.decoder = decoder

        self.save_data = save_data
        self.data_folder = None

        # # Use encoder as the main encoding for the autocodec
        # self.encoding = encoder.encoding
        self.encoding = Encoding()
        self.encoding.set_parameters({
            "n_photons": self.encoder.parameters["n_photons"],
            "m_modes": self.encoder.parameters["m_modes"],
            "ancillae": self.encoder.parameters["ancillae"],
            })

        self.update_targets = self.encoding.update_targets

        self.encoder.parameters["save_data"] = False
        self.decoder.parameters["save_data"] = False

        self.epochs = 0

        # NOTE: Has NOT been tested
        # Check if save folder already exists, and prompt if so.
        if self.save_data:
            if save_folder:
                self.data_folder = Path("data") / "autocodecs" / date.today().strftime("%Y%m%d") / save_folder
            else:
                self.data_folder = Path("data") / "autocodecs" / date.today().strftime("%Y%m%d")
            if encoder.results and decoder.results:
                if self.data_folder.exists():
                    print(f"Warning: {self.data_folder} exists. Data may be overwritten!")
                    input("Press Enter to continue...")
                else:
                    self.data_folder.mkdir(parents=True)

                self.pre_save()

    # NOTE: Has NOT been tested
    def pre_save(self):
        # Backup previous data folder (if it exists) as "parameters (old)"
        # and save the new parameters as "parameters".
        save_path_enc = self.data_folder / "1_parameters_enc"
        save_path_dec = self.data_folder / "2_parameters_dec"
        save_path_enc_old = self.data_folder / "1_parameters_enc (old)"
        save_path_dec_old = self.data_folder / "2_parameters_dec (old)"
        if save_path_enc.exists():
            if save_path_enc_old.exists():
                u.remove(save_path_enc_old)
            save_path_enc.rename(save_path_enc_old)
        u.save_dict(save_path_enc, self.encoder.parameters, suppress_warning=True)
        if save_path_dec.exists():
            if save_path_dec_old.exists():
                u.remove(save_path_dec_old)
            save_path_dec.rename(save_path_dec_old)
        u.save_dict(save_path_dec, self.decoder.parameters, suppress_warning=True)

        save_path = self.data_folder / "pre_opt"
        save_path_old = self.data_folder / "pre_opt (old)"
        if save_path.exists():
            if save_path_old.exists():
                u.remove(save_path_old)
            save_path.rename(save_path_old)
        if self.encoder.results and self.decoder.results:
            u.save_dict(save_path / "1_encoder", self.encoder.results, suppress_warning=True)
            u.save_dict(save_path / "2_decoder", self.decoder.results, suppress_warning=True)

        u.save_dict(save_path / "rhos", {
            "start_inputs": self.encoding.rhos_inputs,
            "start_targets": self.encoding.rhos_targets,
            })

    def pre_optimize(self, pre_opt, random_targets, **kwargs):
        """ Pre-optimises the encoder and the decoder

        Optimizes the encoder and then the decoder depending on their
        respective parameters.

        Parameters:
            pre_opt (bool / tuple(int, int)): If bool, then whether to optimize
                the encoder/decoder the number of times set in their respective
                parameters (True) or not pre-optimize at all (False). Note: one
                evaluation will still be made to initialize the
                encoder/decoder.

                If tuple, then optimize the encoder/decoder the number of
                evaluations stated in the tuple as (# for encoder, # for
                decoder).
        """

        # If pre_opt is True then set the tuple to the number of evals stated
        # in parameters, else set number of evals to 1 and temporarily set
        # verbose to False.
        if pre_opt is True:
            pre_opt = (self.encoder.parameters["num_of_evaluations"],
                       self.decoder.parameters["num_of_evaluations"])
        # elif pre_opt is False:

        #     pre_opt = (1, 1)
        #     verb_en = self.encoder.parameters["verbose"]
        #     verb_de = self.decoder.parameters["verbose"]
        #     print_every_en = self.encoder.parameters["print_every"]
        #     print_every_de = self.decoder.parameters["print_every"]
        #     self.encoder.parameters["verbose"] = False
        #     self.decoder.parameters["verbose"] = False
        #     self.encoder.parameters["print_every"] = 99999
        #     self.decoder.parameters["print_every"] = 99999

        assert isinstance(pre_opt, tuple)

        if self.encoder.parameters["verbose"]:
            print("\n--------------------------")
            print("| Pre-optimizing encoder |")
            print("--------------------------\n")
        else:
            print("Preparing states...")

        no_results = True
        while no_results:
            if not random_targets:
                self.encoder.build_rhos()
            self.encoder.optimize(maxEval=pre_opt[0], **kwargs)
            if self.encoder.results["returnCodeMessage"] != "Generic failure":
                no_results = False

        self.encoding.rhos_inputs = np.copy(self.encoder.encoding.rhos_inputs)
        new_targets = self.encoder.get_output()

        if self.encoder.parameters["verbose"]:
            print("\nLogical 0 pre-opt target:")
            pa(new_targets[0])
            print("\nLogical 1 pre-opt target:")
            pa(new_targets[1])

            print("\nTrace of logical 0:")
            print(np.trace(new_targets[0]))
            print("\nTrace of logical 1:")
            print(np.trace(new_targets[1]))
            print("\nOrthogonality between L0 and L1 (should be 0):")
            print(np.trace(new_targets[0] @ new_targets[1]))
            print("\nOrthogonality between L2 and L3 (should be close to 0):")
            print(np.trace(new_targets[2] @ new_targets[3]))
            print("\nOrthogonality between L4 and L5 (should be close to 0):")
            print(np.trace(new_targets[4] @ new_targets[5]))

            # print("\nOrthogonality between L1 and L2 (should be close to 0.5):")
            # print(np.trace(new_targets[1] @ new_targets[2]))
            # print("\nOrthogonality between L3 and L4 (should be close to 0.5):")
            # print(np.trace(new_targets[3] @ new_targets[4]))
            # print("\nOrthogonality between L0 and L2 (should be close to 0.5):")
            # print(np.trace(new_targets[0] @ new_targets[2]))
            # print("\nOrthogonality between L1 and L3 (should be close to 0.5):")
            # print(np.trace(new_targets[1] @ new_targets[3]))
            # print("\nOrthogonality between L2 and L4 (should be close to 0.5):")
            # print(np.trace(new_targets[2] @ new_targets[4]))
            # print("\nOrthogonality between L3 and L5 (should be close to 0.5):")
            # print(np.trace(new_targets[3] @ new_targets[5]))
            # print("\nOrthogonality between L0 and L3 (should be close to 0.5):")
            # print(np.trace(new_targets[0] @ new_targets[3]))
            # print("\nOrthogonality between L1 and L4 (should be close to 0.5):")
            # print(np.trace(new_targets[1] @ new_targets[4]))
            # print("\nOrthogonality between L2 and L5 (should be close to 0.5):")
            # print(np.trace(new_targets[2] @ new_targets[5]))
            # print("\nOrthogonality between L0 and L4 (should be close to 0.5):")
            # print(np.trace(new_targets[0] @ new_targets[4]))
            # print("\nOrthogonality between L1 and L5 (should be close to 0.5):")
            # print(np.trace(new_targets[1] @ new_targets[5]))
            # print("\nOrthogonality between L0 and L5 (should be close to 0.5):")
            # print(np.trace(new_targets[0] @ new_targets[5]))
            # print("\nIs hermitian:")
            # for nt in new_targets:
            #     print(np.isclose(np.conj(nt).T, nt).all())

        self.encoder.update_targets(new_targets)
        self.decoder.update_targets(new_targets)
        self.update_targets(new_targets)

        if self.decoder.parameters["verbose"]:
            print("\n--------------------------")
            print("| Pre-optimizing decoder |")
            print("--------------------------\n")

        no_results = True
        while no_results:
            self.decoder.apply_loss().optimize(maxEval=pre_opt[1], **kwargs)
            if self.decoder.results["returnCodeMessage"] != "Generic failure":
                no_results = False

        if self.decoder.parameters["verbose"]:
            print("\nDecoder logical 0 pre-opt output:")
            pa(self.decoder.get_output()[0])
            print("\nDecoder logical 1 pre-opt output:")
            pa(self.decoder.get_output()[1])
            print("\n\n")

        # # Set back verbose to the prior value stated in the encoder/decoder
        # if pre_opt is False:
        #     self.encoder.parameters["verbose"] = verb_en
        #     self.decoder.parameters["verbose"] = verb_de
        #     self.encoder.parameters["print_every"] = print_every_en
        #     self.decoder.parameters["print_every"] = print_every_de

    def optimize(self, epochs, pre_opt=False, random_targets=True, decoder_first=False, **kwargs):
        """ Optimize the autocodec system

        Alternating between keeping the encoder and the decoder fixed.

        epochs (int): the number of times to optimize the encoder/decoder set
            before stopping
        pre_opt (bool / tuple(int, int)): Whether to pre-optimize the
            encoder/decoder (True) or not (False), or, if tuple, how many
            evaluations to evaluate the (encoder, decoder).
        """
        # have_results = bool(self.encoder.results and self.decoder.results)
        if random_targets:
            self.encoder.build_rhos()
            self.decoder.build_rhos()

            new_targets = u.generate_orth(2, dim=len(self.encoder.encoding.rhos_targets[0]))

            if self.encoder.parameters["verbose"]:
                print("\nLogical 0 start target:")
                pa(new_targets[0])
                print("\nLogical 1 start target:")
                pa(new_targets[1])

                # print("\nTrace of logical 0:")
                # print(np.trace(new_targets[0]))
                # print("\nTrace of logical 1:")
                # print(np.trace(new_targets[1]))
                # print("\nOrthogonality between L0 and L1 (should be 0):")
                # print(np.trace(new_targets[0] @ new_targets[1]))
                # print("\nOrthogonality between L2 and L3 (should be close to 0):")
                # print(np.trace(new_targets[2] @ new_targets[3]))
                # print("\nOrthogonality between L4 and L5 (should be close to 0):")
                # print(np.trace(new_targets[4] @ new_targets[5]))

                # print("\nOrthogonality between L1 and L2 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[2]))
                # print("\nOrthogonality between L3 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[3] @ new_targets[4]))
                # print("\nOrthogonality between L0 and L2 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[2]))
                # print("\nOrthogonality between L1 and L3 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[3]))
                # print("\nOrthogonality between L2 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[2] @ new_targets[4]))
                # print("\nOrthogonality between L3 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[3] @ new_targets[5]))
                # print("\nOrthogonality between L0 and L3 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[3]))
                # print("\nOrthogonality between L1 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[4]))
                # print("\nOrthogonality between L2 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[2] @ new_targets[5]))
                # print("\nOrthogonality between L0 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[4]))
                # print("\nOrthogonality between L1 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[5]))
                # print("\nOrthogonality between L0 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[5]))
                # print("\nIs hermitian:")
                # for nt in new_targets:
                #     print(np.isclose(np.conj(nt).T, nt).all())

            self.encoder.update_targets(new_targets)
            self.decoder.update_targets(new_targets)
            self.encoding.update_targets(np.copy(self.encoder.encoding.rhos_targets))
            self.encoding.rhos_inputs = np.copy(self.encoder.encoding.rhos_inputs)

            if pre_opt:
                self.pre_optimize(pre_opt, random_targets, **kwargs)
        else:
            if pre_opt:
                self.pre_optimize(pre_opt, random_targets, **kwargs)
            else:
                self.encoder.build_rhos()
                self.decoder.build_rhos().apply_loss()
                self.encoding.update_targets(np.copy(self.encoder.encoding.rhos_targets))
                self.encoding.rhos_inputs = np.copy(self.encoder.encoding.rhos_inputs)

                self.quiet(self.encoder.optimize)(maxEval=1)
                self.quiet(self.decoder.optimize)(maxEval=1)

        if "guess_en" in kwargs:
            # Add provided guess into results for encoder
            self.encoder.results["bestX"] = kwargs["guess_en"]

            # Update targets for autocodec, encoder and decoder
            new_targets = self.encoder.get_output()
            self.encoding.update_targets(new_targets)
            self.encoder.encoding.update_targets(new_targets)
            self.decoder.encoding.update_targets(new_targets)
            self.decoder.apply_loss()

            pa(self.encoder.get_output()[0])
            pa(self.encoder.get_output()[1])
        if "guess_de" in kwargs:
            # Add provided guess into results for decoder
            self.decoder.results["bestX"] = kwargs["guess_de"]
            pa(self.decoder.get_output()[0])
            pa(self.decoder.get_output()[1])

        if self.save_data:  # and not have_results:
            self.pre_save()

        self.epochs = epochs
        for i in range(self.epochs):
            print("\n\n###########################")
            print("###########################")
            print("###                     ###")
            print(f"###   Epoch {i + 1:3d} / {self.epochs:3d}   ###")
            print("###                     ###")
            print("###########################")
            print("###########################\n")

            if not decoder_first or i != 0:
                print("\n----------------------")
                print("| Optimizing encoder |")
                print("----------------------\n")
                # Optimize the full system with the decoder locked (i.e. only optimize
                # the encoder part of the system).

                # if pre_opt or i != 0:
                if "guess_en" in kwargs and i == 0:
                    guess_en = kwargs["guess_en"]
                    print("Using provided guess")
                else:
                    try:
                        guess_en = self.encoder.results["bestX"]
                        # print("len guess en", len(guess_en))
                    except KeyError:
                        guess_en = None

                results_en = opt(
                    self.encoding,
                    system_list=[self.encoder, self.decoder],
                    systems_to_optimize=[True, False],
                    guess=guess_en,
                    **kwargs,
                    )

                if results_en["returnCodeMessage"] != "Generic failure":
                    self.encoder.results = results_en

                # Build the output from the prior optimization and use as new targets
                # for both the decoder and the next encoder optimization
                new_targets = self.encoder.get_output()

                print("\nTrace of logical 0:")
                print(np.trace(new_targets[0]))
                print("\nTrace of logical 1:")
                print(np.trace(new_targets[1]))
                print("\nOrthogonality between L0 and L1 (should be 0):")
                print(np.trace(new_targets[0] @ new_targets[1]))
                print("\nOrthogonality between L2 and L3 (should be close to 0):")
                print(np.trace(new_targets[2] @ new_targets[3]))
                print("\nOrthogonality between L4 and L5 (should be close to 0):")
                print(np.trace(new_targets[4] @ new_targets[5]))

                # print("\nOrthogonality between L1 and L2 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[2]))
                # print("\nOrthogonality between L3 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[3] @ new_targets[4]))
                # print("\nOrthogonality between L0 and L2 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[2]))
                # print("\nOrthogonality between L1 and L3 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[3]))
                # print("\nOrthogonality between L2 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[2] @ new_targets[4]))
                # print("\nOrthogonality between L3 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[3] @ new_targets[5]))
                # print("\nOrthogonality between L0 and L3 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[3]))
                # print("\nOrthogonality between L1 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[4]))
                # print("\nOrthogonality between L2 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[2] @ new_targets[5]))
                # print("\nOrthogonality between L0 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[4]))
                # print("\nOrthogonality between L1 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[5]))
                # print("\nOrthogonality between L0 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[5]))
                # print("\nIs hermitian:")
                # for nt in new_targets:
                #     print(np.isclose(np.conj(nt).T, nt).all())

                self.encoding.update_targets(new_targets)

                print("\nEncoder output => New targets")
                pa(new_targets[0])

            print("\n----------------------")
            print("| Optimizing decoder |")
            print("----------------------\n")
            # Optimize the full system with the encoder locked (i.e. only optimize
            # the decoder part of the system).

            # if pre_opt or i != 0:
            if "guess_de" in kwargs and i == 0:
                guess_de = kwargs["guess_de"]
                print("Using provided guess")
            else:
                try:
                    guess_de = self.decoder.results["bestX"]
                    # print("len guess de", len(guess_de))
                except KeyError:
                    guess_de = None

            results_de = opt(
                self.encoding,
                system_list=[self.encoder, self.decoder],
                systems_to_optimize=[False, True],
                guess=guess_de,
                **kwargs,
                )
            if results_de["returnCodeMessage"] != "Generic failure":
                self.decoder.results = results_de
                # # NOTE: Bumps the targets in a random direction
                # # with strength depending on decoder results
                # for i, nt in enumerate(new_targets):
                #     new_vec = (
                #         2 * np.random.random((1, new_targets.shape[0])) - 1
                #         + (2 * np.random.random((1, new_targets.shape[0])) - 1) * 1j
                #         )
                #     new_targets[i] = (
                #         nt + (np.conj(new_vec.T) @ new_vec)
                #         * results_de["bestError"]
                #         )
                #     new_targets = new_targets / np.trace(new_targets)
                self.encoding.update_targets(new_targets)

            if self.save_data:
                save_path = self.data_folder / f"epoch{i:03d}"
                save_path_old = self.data_folder / f"epoch{i:03d} (old)"
                if save_path.exists():
                    if save_path_old.exists():
                        u.remove(save_path_old)
                    save_path.rename(save_path_old)
                u.save_dict(save_path / "1_encoder", self.encoder.results, suppress_warning=True)
                u.save_dict(save_path / "2_decoder", self.decoder.results, suppress_warning=True)

        return self

    def full_opt(self, pre_opt=False, random_targets=True, **kwargs):
        if random_targets:
            self.encoder.build_rhos()
            self.decoder.build_rhos()

            new_targets = u.generate_orth(2, dim=len(self.encoder.encoding.rhos_targets[0]))

            if self.encoder.parameters["verbose"]:
                print("\nLogical 0 start target:")
                pa(new_targets[0])
                print("\nLogical 1 start target:")
                pa(new_targets[1])

                # print("\nTrace of logical 0:")
                # print(np.trace(new_targets[0]))
                # print("\nTrace of logical 1:")
                # print(np.trace(new_targets[1]))
                # print("\nOrthogonality between L0 and L1 (should be 0):")
                # print(np.trace(new_targets[0] @ new_targets[1]))
                # print("\nOrthogonality between L2 and L3 (should be close to 0):")
                # print(np.trace(new_targets[2] @ new_targets[3]))
                # print("\nOrthogonality between L4 and L5 (should be close to 0):")
                # print(np.trace(new_targets[4] @ new_targets[5]))

                # print("\nOrthogonality between L1 and L2 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[2]))
                # print("\nOrthogonality between L3 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[3] @ new_targets[4]))
                # print("\nOrthogonality between L0 and L2 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[2]))
                # print("\nOrthogonality between L1 and L3 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[3]))
                # print("\nOrthogonality between L2 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[2] @ new_targets[4]))
                # print("\nOrthogonality between L3 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[3] @ new_targets[5]))
                # print("\nOrthogonality between L0 and L3 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[3]))
                # print("\nOrthogonality between L1 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[4]))
                # print("\nOrthogonality between L2 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[2] @ new_targets[5]))
                # print("\nOrthogonality between L0 and L4 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[4]))
                # print("\nOrthogonality between L1 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[1] @ new_targets[5]))
                # print("\nOrthogonality between L0 and L5 (should be close to 0.5):")
                # print(np.trace(new_targets[0] @ new_targets[5]))
                # print("\nIs hermitian:")
                # for nt in new_targets:
                #     print(np.isclose(np.conj(nt).T, nt).all())

            self.encoder.update_targets(new_targets)
            self.decoder.update_targets(new_targets)
            self.encoding.update_targets(np.copy(self.encoder.encoding.rhos_targets))
            self.encoding.rhos_inputs = np.copy(self.encoder.encoding.rhos_inputs)

            if pre_opt:
                self.pre_optimize(pre_opt, random_targets, **kwargs)
        else:
            if pre_opt:
                self.pre_optimize(pre_opt, random_targets, **kwargs)
            else:
                self.encoder.build_rhos()
                self.decoder.build_rhos().apply_loss()
                self.encoding.update_targets(np.copy(self.encoder.encoding.rhos_targets))
                self.encoding.rhos_inputs = np.copy(self.encoder.encoding.rhos_inputs)

                self.quiet(self.encoder.optimize)(maxEval=1)
                self.quiet(self.decoder.optimize)(maxEval=1)

        if "guess_en" in kwargs:
            # Add provided guess into results for encoder
            self.encoder.results["bestX"] = kwargs["guess_en"]

            # Update targets for autocodec, encoder and decoder
            new_targets = self.encoder.get_output()
            self.encoding.update_targets(new_targets)
            self.encoder.encoding.update_targets(new_targets)
            self.decoder.encoding.update_targets(new_targets)
            self.decoder.apply_loss()

            pa(self.encoder.get_output()[0])
            pa(self.encoder.get_output()[1])
        if "guess_de" in kwargs:
            # Add provided guess into results for decoder
            self.decoder.results["bestX"] = kwargs["guess_de"]
            pa(self.decoder.get_output()[0])
            pa(self.decoder.get_output()[1])

        if self.save_data:
            self.pre_save()

        results = opt(
            self.encoding,
            system_list=[self.encoder, self.decoder],
            systems_to_optimize=[True, True],
            guess=np.concatenate((kwargs["guess_en"], kwargs["guess_de"])),
            **kwargs,
            )

        new_targets = self.encoder.get_output(results["bestX"][:len(kwargs["guess_en"])])
        self.encoding.update_targets(new_targets)

        pa(new_targets[0])
        pa(new_targets[1])

        if self.save_data:
            save_path = self.data_folder / f"full_opt"
            save_path_old = self.data_folder / f"full_opt (old)"
            if save_path.exists():
                if save_path_old.exists():
                    u.remove(save_path_old)
                save_path.rename(save_path_old)
            u.save_dict(save_path, results, suppress_warning=True)

        return self

    def output_error(self):
        cost = build_cost(
            self.encoder.encoding,
            system_list=[self.encoder, self.decoder],
            systems_to_optimize=[False, False],
            )

        print(cost(None))

    def quiet(self, func):
        def shhh(**kwargs):
            verb_en = self.encoder.parameters["verbose"]
            verb_de = self.decoder.parameters["verbose"]
            print_every_en = self.encoder.parameters["print_every"]
            print_every_de = self.decoder.parameters["print_every"]

            self.encoder.parameters["verbose"] = False
            self.decoder.parameters["verbose"] = False
            self.encoder.parameters["print_every"] = None
            self.decoder.parameters["print_every"] = None

            func(**kwargs)

            self.encoder.parameters["verbose"] = verb_en
            self.decoder.parameters["verbose"] = verb_de
            self.encoder.parameters["print_every"] = print_every_en
            self.decoder.parameters["print_every"] = print_every_de
        return shhh
