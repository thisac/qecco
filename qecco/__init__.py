from .utils import print_array, save_dict, load_dict, load_data
from .optimize import optimize

from .system.system import System
from .autocodec.autocodec import Autocodec

__all__ = [
    "__title__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "print_array",
    "save_dict",
    "load_dict",
    "load_data",
    "optimize",
    "System",
    "Autocodec",
    "system",
    ]

__title__ = "Quantum Error Correcting Code Optimizer"
__version__ = "0.2"
__description__ = "Optimizer for error correction on Quantum Optical Neural Networks (QONNs)"
__url__ = "https://github.com/thisac/qecco"

__author__ = "Theodor Isacsson"
__email__ = "isacsson@mit.edu"

__license__ = "MIT"
