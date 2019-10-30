import sys

import matplotlib.pyplot as plt
import seaborn
import numpy as np
from qecco.utils import load_data
from qecco import System
from qecco.utils import print_array as pa

seaborn.set()

data = load_data("20191025_en10de10")
en = System("encoder", parameters=data["parameters"][0])
rho = en.build_rhos().get_output(data["best_x"][-4])
print(data["folder_name"][-4])

params = data["parameters"][1]
params.update({"num_of_evaluations": 2000, "ancillae": [1, 1], "num_of_layers": 180})
de = System("decoder", parameters=params)
pa(rho[0])

# params = {
#     "num_of_layers": 20,
#     "num_of_tests": 1,
#     "ancillae": [1, 1],
#     "print_every": 1
#     }
# de = System("decoder", params)

eta_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
distance = np.array(eta_list) / (np.log(10) / 10 * 0.2)
fidelity = []
error = []
guess = None
for eta in eta_list:
    loss_kwargs = {
        "n": 4,
        "m": 2,
        "eta": eta
        }

    print(de.parameters["loss_kwargs"])

    de.update_parameters({"loss_kwargs": loss_kwargs})
    de.update_targets(rho)

    # de.build_rhos().apply_loss().optimize(guess=guess)
    de.apply_loss().optimize(guess=guess)
    guess = np.copy(de.results["bestX"])

    error.append(de.results["bestError"])
    fidelity.append(1 - de.results["bestError"])

print(1 - np.array(fidelity))
print(eta_list)
print(distance)

np.save("error", error)
np.save("eta_list", eta_list)
np.save("fidelity", fidelity)
np.save("distance", distance)

plt.plot(np.append([0], distance), np.append([1], fidelity))
plt.plot(np.append([0], distance), np.append([1], 1 - 6 * np.array(eta_list)**2), label="Analytic (Chuang et. al)")

plt.show()
