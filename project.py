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


def read_single_run_file(datafile):
    with open(datafile) as infile:
        data = np.genfromtxt(infile)
    return data


def stabilization_run(maxn, temp, L):
    E = np.empty(maxn//1000)
    M = np.empty(maxn//1000)
    absM = np.empty(maxn//1000)

    for i in range(maxn//1000):
        print(f"Run with {(i+1)*1000} cycles.", end='\r')
        file = rootdir + f"/data/test{i}.dat"
        run(["./main.exe", file, "single", f"{L}", f"{(i+1)*1000}", f"{temp}"], cwd=src)
        data = read_single_run_file(file)

        E[i] = np.mean(data[:, 0])
        M[i] = np.mean(data[:, 1])
        absM[i] = np.mean(data[:, 2])
    return E, M, absM


if __name__ == "__main__":

    build_cpp()
    temp = 1
    nmax = int(1e7)
    L = 2
    file = rootdir + "/data/test.dat"
    run(["./main.exe", file, "single", f"{L}", f"{nmax}", f"{temp}"], cwd=src)
    data = read_single_run_file(file)

    E = data[:, 0]
    M = data[:, 1]
    absM = data[:, 2]

    E_exp = -2*np.sinh(8/temp)/(np.cosh(8/temp) + 3)
    absM_exp = (2*np.exp(8/temp) + 4)/(np.cosh(8/temp) + 3)/4

    print(f"{np.mean(E)-E_exp:e}")
    print(f"{np.mean(M):e}")
    print(f"{np.mean(absM)-absM_exp:e}")

    n_cycles = np.linspace(1, nmax-1, len(E))
    E = np.cumsum(E)/n_cycles
    M = np.cumsum(M)/n_cycles
    absM = np.cumsum(absM)/n_cycles

    plt.figure()
    plt.title(f"Average energy of {L}x{L} lattice")
    plt.hlines(E_exp, 0, nmax, 'r', label="Analytic")
    plt.plot(n_cycles, E, label="<E>")
    plt.xlabel("N")
    plt.ylabel("<E>")
    plt.legend()
    plt.grid()
    plt.savefig(rootdir + f"/data/{L}x{L}-E.pdf", bbox_inches='tight')

    plt.figure()
    plt.title(f"Average magnetization of {L}x{L} lattice")
    plt.hlines(0, 0, nmax, 'r', label="Analytic")
    plt.plot(n_cycles, M, label="<M>")
    plt.xlabel("N")
    plt.ylabel("<M>")
    plt.legend()
    plt.grid()
    plt.savefig(rootdir + f"/data/{L}x{L}-M.pdf", bbox_inches='tight')

    plt.figure()
    plt.title(f"Average magnitude of magnetization of {L}x{L} lattice")
    plt.hlines(absM_exp, 0, nmax, 'r', label="Analytic")
    plt.plot(n_cycles, absM, label="<|M|>")
    plt.xlabel("N")
    plt.ylabel("<|M|>")
    plt.legend()
    plt.grid()
    plt.savefig(rootdir + f"/data/{L}x{L}-|M|.pdf", bbox_inches='tight')

    plt.show()
