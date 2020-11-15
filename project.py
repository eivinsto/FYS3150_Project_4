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


def stabilization_run(file, nmax, temp, L, randspin=False):
    build_cpp()
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


def read_stabilization_data(file, temp):
    if file.split('-')[-2] == "random":
        randstr = "unordered initial spin."
    else:
        randstr = "ordered initial spin."
    data = np.genfromtxt(file)

    E = data[:, 0]
    absM = data[:, 2]

    n_cycles = np.linspace(1, nmax, len(E))
    E = np.cumsum(E)/n_cycles
    absM = np.cumsum(absM)/n_cycles

    plt.figure()
    plt.title(
        f"Average energy with\n{L = }, T = {temp} and " +
        randstr
    )
    plt.plot(n_cycles, E, label="<E>")
    plt.xlabel("N")
    plt.ylabel("<E>")
    plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/{randstr}-t{temp*10}-{L}x{L}-E.pdf",
        bbox_inches='tight'
    )

    plt.figure()
    plt.title(
        f"Average magnitude of magnetization with\n{L = }, T = {temp} and " +
        randstr
    )
    plt.plot(n_cycles, absM, label="<|M|>")
    plt.xlabel("N")
    plt.ylabel("<|M|>")
    plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/{randstr}-t{temp*10}-{L}x{L}-|M|.pdf",
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


def get_critical_temperature(Ls, absM, Cv, Xi, T):
    # Create array to store critical temperatures
    TC = np.zeros(len(Ls))
    for i, L in enumerate(Ls):
        # Get data arrays
        Mi = absM[i, :].flatten()
        Cvi = Cv[i, :].flatten()
        Xii = Xi[i, :].flatten()

        # Find critical temperature as
        # turning point in (absolute) magnetization
        Mi_dder = np.gradient(np.gradient(Mi))
        Midx = np.where(Mi_dder == np.min(Mi_dder))
        TcM = T[Midx]

        # Find critical temperature as maximum point of heat capacity
        TcCv = T[np.where(Cvi == np.max(Cvi))]

        # Find critical temperature as maximum point in susceptibility
        TcXii = T[np.where(Xii == np.max(Xii))]

        # Average the three values found for final value
        TC[i] = (TcM + TcCv + TcXii)/3

    # Fitting critical temperature as a function of L to estimate critical
    # temperature at L=inf
    p = np.polyfit(1/np.array(Ls), TC, 1)
    return p[1], TC


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

    genflag = input("Generate data? y/n: ")
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
        L = 20
        Ts = [1, 2.4]
        randfiles = [
            rootdir +
            f"/data/{L}-{T}-random-stabilization.dat" for T in Ts
        ]
        ordefiles = [
            rootdir +
            f"/data/{L}-{T}-ordered-stabilization.dat" for T in Ts
        ]

        if genflag == "y":
            for i, (file1, file2) in enumerate(zip(randfiles, ordefiles)):
                stabilization_run(file1, nmax, Ts[i], L, randspin=True)
                stabilization_run(file2, nmax, Ts[i], L)

        for i, (file1, file2) in enumerate(zip(randfiles, ordefiles)):
            read_stabilization_data(file1, Ts[i])
            read_stabilization_data(file2, Ts[i])

    if runflag == "ph":
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

        TCinf, TC = get_critical_temperature(Ls, absM, Cv, Xi, T)
        print("Estimated critical temperature in thermodynamical limit: ",
              TCinf)

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
