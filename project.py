import numpy as np
import matplotlib.pyplot as plt
from subprocess import run
import os
import sys

# Retrieving working directories
rootdir = os.getcwd()
src = rootdir + "/src/"


def build_cpp():
    run(["make", "all"], cwd=src)


def read_data_file(datafile):
    with open(datafile) as infile:
        data = np.genfromtxt(infile)
    return data


def stabilization_run(nmax, temp, L, randspin=False):
    build_cpp()
    file = rootdir + "/data/test.dat"
    if randspin:
        run(
            [
                "./main.exe",
                file,
                "single",
                f"{L}",
                f"{nmax}",
                f"{temp}",
                "true"
            ],
            cwd=src
        )
    else:
        run(
            [
                "./main.exe",
                file,
                "single",
                f"{L}",
                f"{nmax}",
                f"{temp}",
            ],
            cwd=src
        )
    data = read_data_file(file)

    E = data[:, 0]
    absM = data[:, 2]

    n_cycles = np.linspace(1, nmax, len(E))
    E = np.cumsum(E)/n_cycles
    absM = np.cumsum(absM)/n_cycles

    plt.figure()
    plt.title(f"Average energy of {L}x{L} lattice with T = {temp}")
    plt.plot(n_cycles, E, label="<E>")
    plt.xlabel("N")
    plt.ylabel("<E>")
    plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/{randspin}-t{temp*10}-{L}x{L}-E.pdf",
        bbox_inches='tight'
    )

    plt.figure()
    plt.title(f"Average magnitude of magnetization of {L}x{L}" +
              f"lattice with T = {temp}")
    plt.plot(n_cycles, absM, label="<|M|>")
    plt.xlabel("N")
    plt.ylabel("<|M|>")
    plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/{randspin}-t{temp*10}-{L}x{L}-|M|.pdf",
        bbox_inches='tight'
    )


if __name__ == "__main__":

    build_cpp()
    temp = 1
    nmax = int(1e6)
    L = 2
    file = rootdir + "/data/test.dat"
    run(["./main.exe", file, "single", f"{L}", f"{nmax}", f"{temp}"], cwd=src)
    data = read_data_file(file)

    E = data[:, 0]
    M = data[:, 1]
    absM = data[:, 2]

    E_exp = -2*np.sinh(8/temp)/(np.cosh(8/temp) + 3)
    absM_exp = (2*np.exp(8/temp) + 4)/(np.cosh(8/temp) + 3)/4

    print(f"{np.mean(E)-E_exp:e}")
    print(f"{np.mean(M):e}")
    print(f"{np.mean(absM)-absM_exp:e}")

    n_cycles = np.linspace(1, nmax, len(E))
    E = np.cumsum(E)/n_cycles
    # M = np.cumsum(M)/n_cycles
    absM = np.cumsum(absM)/n_cycles

    plt.figure()
    plt.title(f"Average energy of {L}x{L} lattice with T = {temp}")
    plt.hlines(E_exp, 0, nmax, 'r', label="Analytic")
    plt.plot(n_cycles, E, label="<E>")
    plt.xlabel("N")
    plt.ylabel("<E>")
    plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/t{temp*10}-{L}x{L}-E.pdf", bbox_inches='tight')

    plt.figure()
    plt.title(f"Average magnitude of magnetization of {L}x{L}" +
              f"lattice with T = {temp}")
    plt.hlines(absM_exp, 0, nmax, 'r', label="Analytic")
    plt.plot(n_cycles, absM, label="<|M|>")
    plt.xlabel("N")
    plt.ylabel("<|M|>")
    plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/t{temp*10}-{L}x{L}-|M|.pdf", bbox_inches='tight')

    stabilization_run(nmax, 1, 20)
    stabilization_run(nmax, 2.4, 20)
    stabilization_run(nmax, 1, 20, randspin=True)
    stabilization_run(nmax, 2.4, 20, randspin=True)

    plt.show()
