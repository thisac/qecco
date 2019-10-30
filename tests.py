import unittest

from qecco import System


class TestEncoder(unittest.TestCase):
    def test_simple_cost(self):
        en_params = {
            "simple_cost": True,
            "ancillae": [0],
            "save_data": False,
            "verbose": False,
            }
        en_simple = System("encoder", en_params)
        en_simple.build_codes().optimize()
        self.assertLess(en_simple.results["bestError"], 1e-13)

    def test_default(self):
        en_params = {
            "print_every": 10,
            "simple_cost": True,
            "ancillae": [0],
            "save_data": False,
            "verbose": False,
            }
        en = System("encoder", en_params)
        en.build_codes().optimize()
        self.assertLess(en.results["bestError"], 1e-13)


class TestDecoder(unittest.TestCase):
    def test_default(self):
        de_params = {
            "print_every": 10,
            "simple_cost": False,
            "ancillae": [0],
            "save_data": False,
            "verbose": False,
            }
        de = System("decoder", de_params)
        de.build_rhos().apply_loss().optimize()
        self.assertAlmostEqual(de.results["bestError"], 0.092746875)


class TestRydberg(unittest.TestCase):
    def test_CNOT(self):
        ryd_params = {
            "num_of_layers": 5,
            "opt_algorithm": "BOBYQA",
            "code_name": "CNOT_rydberg",
            "pulse_sequence": "CNOT",
            "verbose": True,
            "save_data": False,
            }
        ryd = System("rydberg", ryd_params)
        ryd.build_codes().optimize()
        self.assertLess(ryd.results["bestError"], 1e-13)


if __name__ == '__main__':
    unittest.main()
