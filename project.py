import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from subprocess import run, Popen
import os
import sys

# Retrieving working directories
rootdir = os.getcwd()
src = rootdir + "/src/"


def Xifunc(M2mean, Mmean2, T):
    return (M2mean - Mmean2)/T


def Cvfunc(E2mean, Emean2, T):
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


def phase_trans_test(nmax, Ls, files, n_temps):
    # lists of L-values and filenames:
    Tmin, Tmax = 2, 2.3  # min and max temp.

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
        print("Spawning subprocesses:")
        sims = [phase_L_sim(j) for j in range(k, ranges[i] + k)]
        k += len(sims)  # counting up.

        # waiting for all concurrent subprocesses to finish:
        [p.wait() for p in sims]


def read_phase_trans(nmax, Ls, files):
    data = {}  # dictionary for data.
    # reading data from files:
    for L, file in zip(Ls, files):
        data[L] = np.genfromtxt(file)
    return data


def get_critical_temperature(Ls,absM,Cv,Xi,T):
    TC = np.zeros(len(Ls))
    for i,L in enumerate(Ls):
        Mi = absM[i,:].flatten()
        Cvi = Cv[i,:].flatten()
        Xii = Xi[i,:].flatten()

        Mi_dder = np.gradient(np.gradient(Mi))
        Midx = np.where(Mi_dder==np.min(Mi_dder))
        TcM = T[Midx]

        TcCv = T[np.where(Cvi==np.max(Cvi))]

        TcXii = T[np.where(Xii==np.max(Xii))]

        TC[i] = (TcM + TcCv + TcXii)/3

    p = np.polyfit(1/np.array(Ls),TC,1)
    return p[1],TC



runflag = "start"
if __name__ == "__main__":
    while runflag != "an" and runflag != "st" and runflag != "ph":
        runflag = input("Analytic vs numeric 2x2 = 'an', " +
                        "stabilization run = 'st', " +
                        "phase transition = ph, " +
                        "quit = 'q'.\n" +
                        "Enter run: ")
        if runflag == "quit" or runflag == "q":
            print("Exiting.")
            sys.exit(0)

    build_cpp()
    nmax = int(1e6)

    if runflag == "an":
        temp = 1
        L = 2
        file = rootdir + "/data/2x2-test.dat"
        run(["./main.exe", file, "single", f"{L}", f"{nmax}", f"{temp}"],
            cwd=src)
        data = np.genfromtxt(file)

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

    if runflag == "st":
        stabilization_run(nmax, 1, 20)
        stabilization_run(nmax, 2.4, 20)
        stabilization_run(nmax, 1, 20, randspin=True)
        stabilization_run(nmax, 2.4, 20, randspin=True)

    if runflag == "ph":
        genflag = input("Generate data? y/n: ")
        Ls = [40, 60, 80, 100]
        files = [rootdir + f"/data/{L}x{L}-multi.dat" for L in Ls]
        n_temps = 8  # number of temps to simulate per process.
        if genflag == "y":
            phase_trans_test(nmax, Ls, files)

        data = read_phase_trans(nmax, Ls, files)

        E = np.zeros((len(Ls), n_temps))
        M = np.zeros((len(Ls), n_temps))
        Cv = np.zeros((len(Ls), n_temps))
        Xi = np.zeros((len(Ls), n_temps))
        absM = np.zeros((len(Ls), n_temps))

        sorted = np.argsort(data[Ls[0]][:, 0])
        T = data[Ls[0]][sorted, 0]

        for i in range(len(Ls)):
            E[i, :] = data[Ls[i]][sorted, 1]
            Cv[i, :] = data[Ls[i]][sorted, 2]
            M[i, :] = data[Ls[i]][sorted, 3]
            Xi[i, :] = data[Ls[i]][sorted, 4]
            absM[i, :] = data[Ls[i]][sorted, 5]

        TCinf,TC = get_critical_temperature(Ls,absM,Cv,Xi,T)
        print(TCinf)

        plt.figure()
        plt.title("<E>")
        for i in range(len(Ls)):
            plt.plot(T, E[i, :], label=f"L = {Ls[i]}")
        plt.xlabel("T")
        plt.ylabel("<E>")
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title("<M>")
        for i in range(len(Ls)):
            plt.plot(T, M[i, :], label=f"L = {Ls[i]}")
        plt.xlabel("T")
        plt.ylabel("<M>")
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title("Cv")
        for i in range(len(Ls)):
            plt.plot(T, Cv[i, :], label=f"L = {Ls[i]}")
        plt.xlabel("T")
        plt.ylabel("Cv")
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title(r"$\chi$")
        for i in range(len(Ls)):
            plt.plot(T, Xi[i, :], label=f"L = {Ls[i]}")
        plt.xlabel("T")
        plt.ylabel(r"$\chi$")
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title("<|M|>")
        for i in range(len(Ls)):
            plt.plot(T, absM[i, :], label=f"L = {Ls[i]}")
        plt.xlabel("T")
        plt.ylabel("<|M|>")
        plt.legend()
        plt.grid()

    plt.show()
