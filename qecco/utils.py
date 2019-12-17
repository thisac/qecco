from pathlib import Path
from scipy.stats import ortho_group
# import warnings
import time

import numpy as np
import bosonic as b

# warnings.simplefilter(action='ignore', category=ComplexWarning)


def print_a(array):
    nparray = np.array(array)
    arr_string = "      "
    arr_line = "       "
    for i, _ in enumerate(nparray[0]):
        arr_string = arr_string + " {:4d} ".format(i)
        arr_line = arr_line + "------"
    print("\n" + arr_string)
    print(arr_line)
    for i, arr in enumerate(nparray):
        arr_string = ""
        for j in arr:
            j = np.real(j)
            if np.isclose(int(j), j, atol=0.01):
                arr_string = arr_string + " {:4d} ".format(int(j))
            else:
                arr_string = arr_string + " {:.2f} ".format(j)
        print("{:3d} : {}".format(i, arr_string))
    print("\n")


def print_array(array2d):
    array2d = np.array(array2d)
    int_array = np.array(
        [[np.real(complex(f"{i:.2f}")).is_integer()
            and np.imag(complex(f"{i:.2f}")).is_integer()
            for i in row] for row in array2d])
    idx_0 = np.where(int_array)
    top_row = np.arange(len(array2d[0]))
    top_str = [""] + [f"{e:2d}" for e in top_row]

    str_mat = np.array([[f"{e:.2f}" for e in row] for row in array2d])

    str_mat[idx_0] = np.array([int(round(np.real(i))) for i in array2d[idx_0]])
    lens = np.max([list(map(len, col)) for col in zip(*str_mat)], axis=1)
    fmt = "{{:>{}}} :  ".format(2) + "  ".join("{{:>{}}}".format(x + 1) for x in lens)
    table = [fmt.format(i, *row) for i, row in enumerate(str_mat)]

    fmt_top = "{{:>{}}}    ".format(2) + "  ".join("{{:>{}}}".format(x + 1) for x in lens)
    top_table = fmt_top.format(*top_str)
    top_print = "   " + "".join(top_table)[3:]

    print("\n", top_print)
    print("     " + "-" * (len(top_print) - 4))
    print("\n".join(table))


def print_idx(rho):
    idx = np.where(rho != 0)
    print_array(np.vstack((idx, rho[idx])).T)


def remove(remove_path, keep_dir=False):
    if remove_path.is_dir():
        for child in remove_path.iterdir():
            remove(child)
        if not keep_dir:
            remove_path.rmdir()
    else:
        remove_path.unlink()


def save_dict(save_path, *args, suppress_warning=False, remove_dir=False):
    save_path = Path(save_path)
    data_dict = {}
    for a_dict in args:
        data_dict = {**data_dict, **a_dict}

    if save_path.exists():
        if not suppress_warning:
            print(f"Warning: {save_path} exists. Data will be overwritten!")
            input("Press Enter to continue...")
        if remove_dir:
            remove(save_path)
            save_path.mkdir(parents=True)
    else:
        save_path.mkdir(parents=True)

    for key, value in data_dict.items():
        np.save(save_path / f"{key}", value)


def load_dict(load_path):
    load_path = Path(load_path)
    data_dict = {}

    for child in load_path.iterdir():
        try:
            value = np.load(child, allow_pickle=True).item()
        except ValueError:
            value = np.load(child, allow_pickle=True)
        key = child.stem
        data_dict[key] = value

    return data_dict


def find_folder(search_folder, name, print_all=False):
    """Finds all files/folder with {name} in their names

    Iterates through a folder finding all files/folder including {name} in
    their names returning all the files/folders fitting this criteria

    Parameters:
        search_folder (string): the folder in which to search
        name (string): the string to search for
        print_all (bool): whether to print a list of all hits

    Returns (list): a list with the paths to all files/folders with {name} in
    """
    child_list = []
    if list(Path(search_folder).glob("*" + name + "*")):
        child_list.append(list(Path(search_folder).glob("*" + name + "*")))

    def iter_folder(search_folder):
        try:
            for child in Path(search_folder).iterdir():
                if list(child.glob("*" + name + "*")):
                    child_list.append(list(child.glob("*" + name + "*")))
                else:
                    iter_folder(child)
            return None
        except NotADirectoryError:
            return None

    iter_folder(search_folder)
    if print_all:
        for el in child_list:
            for em in el:
                print(em)

    return np.array(child_list).flatten()


def exist_in_folder(search_folder, name):
    if list(find_folder(search_folder, name, print_all=False)):
        return True
    else:
        return False


def load_data(folder_name, build_S=False):
    data_directory = find_folder("data", folder_name)[0]
    print("data dir:", data_directory)
    parameter_paths_tmp = find_folder(data_directory, "parameter")
    parameter_paths = []
    for pp in parameter_paths_tmp:
        if pp.is_dir() and "(old)" not in str(pp):
            parameter_paths.append(pp)

    ####################################################################
    # for p in parameter_paths:
    #     print(p)
    ####################################################################

    parameters_list = [load_dict(pp) for pp in parameter_paths]

    if exist_in_folder(data_directory, "(best)"):
        s = "(best)"
    else:
        s = ""

    layers = [el["num_of_layers"] for el in parameters_list]
    error_arrays = [
        np.load(el) for el in find_folder(data_directory, "errorArray.npy")
        if s in str(el) and "(old)" not in str(el) and "pre" not in str(el)
        ]
    run_times = [
        np.load(el).item() for el in find_folder(data_directory, "runTime.npy")
        if s in str(el) and "(old)" not in str(el) and "pre" not in str(el)
        ]
    best_errors = [
        np.load(el).item() for el in find_folder(data_directory, "bestError.npy")
        if s in str(el) and "(old)" not in str(el) and "pre" not in str(el)
        ]
    fidelity = [
        np.load(el).item() for el in find_folder(data_directory, "fidelity.npy")
        if s in str(el) and "(old)" not in str(el) and "pre" not in str(el)
        ]
    guess = [
        np.load(el) for el in find_folder(data_directory, "guess.npy")
        if s in str(el) and "(old)" not in str(el) and "pre" not in str(el)
        ]
    best_x = [
        np.load(el) for el in find_folder(data_directory, "bestX.npy")
        if s in str(el) and "(old)" not in str(el) and "pre" not in str(el)
        ]
    best_xs = [
        np.load(el) for el in find_folder(data_directory, "bestXs.npy")
        if s in str(el) and "(old)" not in str(el) and "pre" not in str(el)
        ]
    evaluations = [
        np.load(el).item() for el in find_folder(data_directory, "numEvaluations.npy")
        if s in str(el) and "(old)" not in str(el) and "pre" not in str(el)
        ]
    folder_name = [
        el.parent for el in find_folder(data_directory, "bestX.npy")
        if s in str(el) and "(old)" not in str(el) and "pre" not in str(el)
        ]
    inputs = [
        np.load(el) for el in find_folder(data_directory, "start_inputs.npy")
        if "(old)" not in str(el)
        ]
    targets = [
        np.load(el) for el in find_folder(data_directory, "start_targets.npy")
        if "(old)" not in str(el)
        ]

    ####################################################################
    # pl = [pp for pp in parameter_paths]
    #
    # ea = [
    #     (el) for el in find_folder(data_directory, "errorArray")
    #     if s in str(el) and "(old)" not in str(el)
    #     ]
    # rt = [
    #     (el) for el in find_folder(data_directory, "runTime")
    #     if s in str(el) and "(old)" not in str(el)
    #     ]
    # be = [
    #     (el) for el in find_folder(data_directory, "bestError")
    #     if s in str(el) and "(old)" not in str(el)
    #     ]
    # bx = [
    #     (el) for el in find_folder(data_directory, "bestX")
    #     if s in str(el) and "(old)" not in str(el)
    #     ]
    # ev = [
    #     (el) for el in find_folder(data_directory, "numEvaluations")
    #     if s in str(el) and "(old)" not in str(el)
    #     ]

    # for p in layers:
    #     print(p)
    # for p in pl:
    #     print(p)
    # for p in ea:
    #     print(p)
    # for p in rt:
    #     print(p)
    # for p in be:
    #     print(p)
    # for p in bx:
    #     print(p)
    # for p in ev:
    #     print(p)
    ####################################################################

    best_S = []
    if build_S:
        build_S_path = find_folder(data_directory, "best_S")
        for i, parameters in enumerate(parameters_list):

            if len(build_S_path) and not build_S:
                best_S.append(np.load(build_S_path[i]))
            else:
                try:
                    lossy = parameters["lossy"]
                except KeyError:
                    lossy = True

                build_system, _ = b.qonn.build_system_function(
                    int(parameters["n_photons"]),
                    int(parameters["m_modes"]),
                    int(parameters["num_of_layers"][0]),
                    phi=None,
                    method=str(parameters["method"]),
                    lossy=lossy,
                    )
                best_S_single = []
                for x in best_x:
                    S = build_system(x)
                    best_S_single.append(S)
                best_S.append(best_S_single)
                np.save(data_directory / f"best_S_{parameter_paths[i].name}",
                        best_S_single)

    data_dict = {
        "parameters": parameters_list,
        "layers": layers,
        "error_arrays": error_arrays,
        "run_times": run_times,
        "best_errors": best_errors,
        "fidelity": fidelity,
        "guess": guess,
        "best_x": best_x,
        "best_xs": best_xs,
        "evaluations": evaluations,
        "best_S": best_S,
        "folder_name": folder_name,
        "inputs": inputs,
        "targets": targets,
        }
    return data_dict


def print_dict(dict_to_print):
    for key, value in dict_to_print.items():
        print(f"{key:20}: {value}")


def find_next_word(long_string, word):
    for i, piece in enumerate(long_string.split()):
        if word in piece:
            return long_string.split()[i + 1]
    return None


# NOTE: Slow, but works with autograd
def array_assignment(mat, mini_mat, pos):
    mat = np.array(mat)
    mini_mat = np.array(mini_mat)

    if mini_mat.ndim == 0:
        mini_mat = np.array([[mini_mat]])
    elif mini_mat.ndim == 1:
        mini_mat = np.array([mini_mat])

    new_mat = []
    for i, row in enumerate(mat):
        new_row = []
        for j, el in enumerate(row):
            if ((i >= pos[0] and i < pos[0] + mini_mat.shape[0])
                    and (j >= pos[1] and j < pos[1] + mini_mat.shape[1])):
                new_row.append(mini_mat[i - pos[0]][j - pos[1]])
            else:
                new_row.append(mat[i][j])
        new_mat.append(new_row)

    return np.array(new_mat)


def generate_orth(n, dim):
    ortho_mat = ortho_group.rvs(dim=dim)
    random_targets = []
    rand_pure = []
    rand_1 = ortho_mat.T[0].reshape(-1, 1)
    rand_2 = ortho_mat.T[1].reshape(-1, 1)
    # For random, non-orthogonal rhos
    # rand_1 = (2 * np.random.random(dim) - 1).reshape(-1, 1)
    # rand_2 = (2 * np.random.random(dim) - 1).reshape(-1, 1)
    rand_pure.append(rand_1)
    rand_pure.append(rand_2)
    rand_pure.append((rand_1 + rand_2) / np.sqrt(2))
    rand_pure.append((rand_1 - rand_2) / np.sqrt(2))
    rand_pure.append((rand_1 + 1j * rand_2) / np.sqrt(2))
    rand_pure.append((rand_1 - 1j * rand_2) / np.sqrt(2))

    for rp in rand_pure:
        rand_mat = rp @ np.conj(rp).T

        # new_vec = (
        #     2 * np.random.random((1, rand_mat.shape[0])) - 1
        #     + (2 * np.random.random((1, rand_mat.shape[0])) - 1) * 1j
        #     )
        # rand_mat = (
        #     rand_mat + (np.conj(new_vec.T) @ new_vec)
        #     * 100
        #     )

        rand_rho = rand_mat / np.trace(rand_mat)

        random_targets.append(rand_rho)
    return random_targets


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f"{method.__name__}  {(te - ts) * 1000:2.2f} ms")
        return result
    return timed
