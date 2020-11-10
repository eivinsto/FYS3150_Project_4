import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from subprocess import run, Popen
import os
import sys

# Retrieving working directories
rootdir = os.getcwd()
src = rootdir + "/src/"


def Xi(M2mean, Mmean2, T):
    return (M2mean - Mmean2)/T


def Cv(E2mean, Emean2, T):
    return (E2mean - Emean2)/T**2


def build_cpp():
    run(["make", "all"], cwd=src)


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
    data = np.genfromtxt(file)

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


def phase_trans_test(nmax):
    # lists of L-values and filenames:
    Ls = [40, 60, 80, 100]
    files = [rootdir + f"/data/{L}x{L}-multi.dat" for L in Ls]
    Tmin, Tmax = 2, 2.3  # min and max temp.
    n_temps = 7  # number of temps to simulate per process.
    data = {}  # dictionary for data.

    n_thrds = mp.cpu_count()//n_temps  # number of subprocesses to spawn

    def phase_L_sim(i):
        """Function spawning process and returning it."""
        p = Popen(
            [
                "./main.exe",
                files[i],
                "multi",
                f"{Ls[i]}",
                f"{nmax}",
                f"{Tmin}",
                f"{Tmax}",
                f"{n_temps}"
            ],
            cwd=src
        )
        return p

    # sets number of subprocesses to spawn at the same time:
    ranges = [len(Ls[i:i+n_thrds]) for i in range(0, len(Ls), n_thrds)]
    k = 0  # counter for number of subprocesses that have been run

    # iterating through number of subprocesses to spawn at the same time:
    for i in range(len(ranges)):
        print("spawning subprocesses:")
        sims = [phase_L_sim(j) for j in range(k, ranges[i] + k)]
        k += len(sims)  # counting up.

        # waiting for all concurrent subprocesses to finish:
        [p.wait() for p in sims]

    # reading data from files:
    for L, file in zip(Ls, files):
        data[L] = np.genfromtxt(file)


if __name__ == "__main__":

    build_cpp()
    # temp = 1
    nmax = int(1e3)
    # L = 2
    # file = rootdir + "/data/test.dat"
    # run(["./main.exe", file, "single", f"{L}", f"{nmax}", f"{temp}"], cwd=src)
    # data = np.genfromtxt(file)
    #
    # E = data[:, 0]
    # M = data[:, 1]
    # absM = data[:, 2]
    #
    # E_exp = -2*np.sinh(8/temp)/(np.cosh(8/temp) + 3)
    # absM_exp = (2*np.exp(8/temp) + 4)/(np.cosh(8/temp) + 3)/4
    #
    # print(f"{np.mean(E)-E_exp:e}")
    # print(f"{np.mean(M):e}")
    # print(f"{np.mean(absM)-absM_exp:e}")
    #
    # n_cycles = np.linspace(1, nmax, len(E))
    # E = np.cumsum(E)/n_cycles
    # # M = np.cumsum(M)/n_cycles
    # absM = np.cumsum(absM)/n_cycles
    #
    # plt.figure()
    # plt.title(f"Average energy of {L}x{L} lattice with T = {temp}")
    # plt.hlines(E_exp, 0, nmax, 'r', label="Analytic")
    # plt.plot(n_cycles, E, label="<E>")
    # plt.xlabel("N")
    # plt.ylabel("<E>")
    # plt.legend()
    # plt.grid()
    # plt.savefig(
    #     rootdir + f"/data/t{temp*10}-{L}x{L}-E.pdf", bbox_inches='tight')
    #
    # plt.figure()
    # plt.title(f"Average magnitude of magnetization of {L}x{L}" +
    #           f"lattice with T = {temp}")
    # plt.hlines(absM_exp, 0, nmax, 'r', label="Analytic")
    # plt.plot(n_cycles, absM, label="<|M|>")
    # plt.xlabel("N")
    # plt.ylabel("<|M|>")
    # plt.legend()
    # plt.grid()
    # plt.savefig(
    #     rootdir + f"/data/t{temp*10}-{L}x{L}-|M|.pdf", bbox_inches='tight')
    #
    # stabilization_run(nmax, 1, 20)
    # stabilization_run(nmax, 2.4, 20)
    # stabilization_run(nmax, 1, 20, randspin=True)
    # stabilization_run(nmax, 2.4, 20, randspin=True)
    phase_trans_test(nmax)
    plt.show()
