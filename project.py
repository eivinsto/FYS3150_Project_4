from subprocess import run, Popen, PIPE, call
import multiprocessing as mp
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Retrieving working directories
rootdir = os.getcwd()
src = rootdir + "/src/"


def build_cpp():
    """Function building cpp program."""
    run(["make", "all"], cwd=src)


def stabilization_run(file, nmax, temp, L, randspin=False):
    """Function running cpp program for LxL lattice with ordered or
    disordered spin. Resulting data is written to files in /data/.

    Arguments:
    file -- str: filename to write data to.
    nmax -- int: number of Monte Carlo cycles to perform.
    temp -- float: temperature of lattice.
    L -- int: dimensionality of lattice.

    Keyword arguments:
    randspin -- bool: Set to True for disordered initial spins
                      (default: False).
    """
    build_cpp()
    spin = str(randspin).lower()
    run(
        ["./main.exe", file, "single", f"{L}",
         f"{nmax}", f"{temp}", f"{spin}"],
        cwd=src
    )


def read_stabilization_data(file):
    """Function reading data from stabilization run,
    and plotting mean energy and magnitude of magnetization
    as function of number of Monte Carlo cycles.

    Arguments:
    file -- str: filename to read data from.
    """
    temp = float(file.split('-')[-3])
    if file.split('-')[-2] == "random":
        randstr = "disordered initial spin."
    else:
        randstr = "ordered initial spin."

    # reading data
    data = np.genfromtxt(file)
    E = data[:, 0]
    absM = data[:, 2]
    accepted_configs = data[:, 3]

    # taking normalized, cumulative sum of data
    n_cycles = np.linspace(1, nmax, len(E))
    Emean = np.cumsum(E)/n_cycles
    absMmean = np.cumsum(absM)/n_cycles

    # plotting data and saving figures to /data/:
    plt.figure()
    plt.title(
        f"Average energy with\nL = {L}, T = {temp} and " +
        randstr
    )
    plt.plot(n_cycles, Emean, label=r"$\langle E \rangle/J$")
    plt.xlabel("N")
    plt.ylabel(r"$\langle E \rangle/J$")
    plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/{randstr}-t{temp}-{L}x{L}-E.pdf",
        bbox_inches='tight'
    )

    plt.figure()
    plt.title(
        f"Average magnitude of magnetization with\nL = {L}, T = {temp} and " +
        randstr
    )
    plt.plot(n_cycles, absMmean, label=r"$\langle | \mathcal{M} | \rangle$")
    plt.xlabel("N")
    plt.ylabel(r"$\langle | \mathcal{M} | \rangle$")
    plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/{randstr}-t{temp}-{L}x{L}-|M|.pdf",
        bbox_inches='tight'
    )

    plt.figure()
    plt.title(
        f"Accepted configurations with\nL = {L}, T = {temp} and " +
        randstr
    )
    plt.semilogy(n_cycles, accepted_configs,
                 label="Accepted spin flips")
    plt.xlabel("N")
    plt.ylabel("Number of flips")
    plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/{randstr}-t{temp}-{L}x{L}-Nconf.pdf",
        bbox_inches='tight'
    )

    slicer = int(4e5)

    Evar = np.var(E[slicer:])
    var = r"$\sigma_{E}^{2} = $" + f"{Evar:.4g}"

    plt.figure()
    plt.title(
        f"Distribution of energies\nL = {L}, T = {temp} and " +
        randstr + "\n" + var
    )
    plt.hist(E[slicer:], bins="auto", density=True, stacked=True)
    #plt.plot(np.sort(E[slicer:]), (2.2/(np.sqrt(2*np.pi*Evar)))*np.exp(-((np.sort(E[slicer:])-np.mean( E[slicer:] ))**2 /Evar  )/2 ) ,'r',label='Normal distribution')
    plt.xlabel("E/J")
    plt.ylabel("P(E/J)")
    # plt.legend()
    plt.grid()
    plt.savefig(
        rootdir + f"/data/{randstr}-t{temp}-{L}x{L}-PE.pdf",
        bbox_inches='tight'
    )


def phase_trans_test(nmax, Ls, files, n_temps):
    """Function running simulations for phase transition.
    Runs the cpp program for each L-value. If enough threads are available,
    multiple instances of the cpp program are run concurrently.

    Arguments:
    nmax -- int: number of Monte Carlo cycles to perform.
    Ls -- iterable: iterable containing the L-values to simulate.
    files -- iterable: iterable containing the filenames to write to.
    n_temps -- int: number of temperatures to simulate in interval
                    [Tmin, Tmax].
    """
    Tmin, Tmax = 2.2, 2.35  # min and max temp.

    # number of concurrent subprocesses to spawn
    n_thrds = np.max([mp.cpu_count()//n_temps, 1])

    def phase_L_sim(i):
        """Function spawning subprocess and returning it."""
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
    # iterating through number of subprocesses to spawn at the same time:
    print("Spawning subprocesses:")
    sims = [phase_L_sim(j) for j in range(ranges[0])]

    # loop checking status of subprocesses,
    # and spawning new subprocesses when a current one finishes.
    i = len(sims)
    while i < len(Ls):
        for p in sims:
            if p.poll() is not None:
                print("Spawning subprocesses:")
                sims.append(phase_L_sim(i))
                i += 1

    # waiting for all subprocesses to finish
    [p.wait() for p in sims]


def read_phase_trans(nmax, Ls, files):
    """Function reading phase transition data from files.

    Returns dictionary containing numpy arrays. Keys of the dictionary are the
    L-values simulated. Each array contains columns containing
    T <E> <M> Cv Xi absM iT
    where iT is the index of temperature in interval [Tmin, Tmax].

    Arguments:
    nmax -- int: number of Monte Carlo cycles to perform.
    Ls -- iterable: iterable containing the simulated L-values.
    files -- iterable: iterable containing the filenames to read from.
    """
    data = {}  # dictionary for data.
    # reading data from files:
    for L, file in zip(Ls, files):
        data[L] = np.genfromtxt(file)
    return data


def get_critical_temperature(Ls, Cv, Xi, T):
    """Function calculating critical temperature from phase transition data.

    Returns tuple where first element is critical temperature at the
    thermodynamical limit (float). Second element is array containing mean of
    critical temperatures for each value of L.

    Arguments:
    Ls -- iterable: iterable containing the simulated L-values.
    Cv -- iterable: iterable containing the values of the specific heat.
    Xi -- iterable: iterable containing the values of the susceptibility.
    T -- iterable: iterable containing the values of temperature.
    """
    TC = np.zeros(len(Ls))
    for i, L in enumerate(Ls):
        # Get data arrays
        Cvi = Cv[i, :].flatten()
        Xii = Xi[i, :].flatten()

        # Find critical temperature as maximum point of heat capacity
        TcCv = T[np.where(Cvi == np.max(Cvi))]

        # Find critical temperature as maximum point in susceptibility
        TcXii = T[np.where(Xii == np.max(Xii))]

        # Average the three values found for final value
        TC[i] = (TcCv + TcXii)*0.5

    # Fitting critical temperature as a function of L to estimate critical
    # temperature at L=inf
    p, V = np.polyfit(1/np.array(Ls), TC, 1, cov=True)
    return p[1], np.sqrt(V[1][1]), TC


def benchmark(N_list, gccflags, archflag, L, n_temps):
    """Function generating benchmark data for -O0, -O1, -O2, -O3
    and -march=native compiler flags. Data is stored in
    benchmarkrun.npy file in /data/.

    Arguments:
    N_list -- Iterable: Iterable containing number of Monte Carlo
                        cycles to run.
    gccflags -- Iterable: Iterable containing optimization-flags as strings.
    archflag -- str: flag for architecture-specific optimization.
    L -- int: dimensionality of test lattice.
    n_temps -- int: number of temperature values to test.
    """
    Tmin, Tmax = 2.2, 2.35  # min and max temp.
    output = {}
    print(N_list)
    for k, N in enumerate(N_list):
        for j in range(2):
            for i in range(len(gccflags)):
                # compiling program with specified flags
                if j == 0:
                    call(["g++", gccflags[i], "ising_metropolis.cpp",
                          "-fopenmp", "-c"], cwd=src)
                    call(["g++", gccflags[i], "main.cpp",
                          "-fopenmp", "-c"], cwd=src)
                    call(["g++", gccflags[i], "-fopenmp", "main.o",
                          "ising_metropolis.o", "-o", "benchmark.exe",
                          "-larmadillo"],
                         cwd=src)
                else:
                    call(["g++", gccflags[i], archflag, "ising_metropolis.cpp",
                          "-fopenmp", "-c"], cwd=src)
                    call(["g++", gccflags[i], archflag, "main.cpp",
                          "-fopenmp", "-c"], cwd=src)
                    call(["g++", gccflags[i], archflag, "-fopenmp", "main.o",
                          "ising_metropolis.o", "-o", "benchmark.exe",
                          "-larmadillo"],
                         cwd=src)

                # running simulation with current compilation.
                p = Popen(
                    [
                        "./benchmark.exe",
                        rootdir + "/data/benchmarkrun.dat",
                        "multi",
                        f"{L}",
                        f"{N}",
                        f"{Tmin}",
                        f"{Tmax}",
                        f"{n_temps}"
                    ],
                    stdout=PIPE,
                    stderr=PIPE,
                    cwd=src
                )

                # capturing standard streams from process
                # and decoding data
                stdout, stderr = p.communicate()
                output[(k, j, i)] = stdout.decode('utf-8')
                run(["rm", "-rf", rootdir + "/data/benchmarkrun.dat"])

    # unpacking data and storing in array
    times = np.empty((len(N_list), 2, 4))
    for key in output:
        string = output[key].split('=')[3].strip()
        time = string.split('\n')[0]
        times[key[0], key[1], key[2]] = float(time)

    # saving array to file
    np.save(rootdir + "/data/benchmarkrun.npy", times, allow_pickle=False)


def read_benchmark(N_list, gccflags, archflag, L, n_temps):
    """Function reading and plotting benchmark data for -O0, -O1, -O2,
    -O3 and -march=native compiler flags.

    Arguments:
    N_list -- Iterable: Iterable containing number of Monte Carlo
                        cycles to run.
    gccflags -- Iterable: Iterable containing optimization-flags as strings.
    archflag -- str: flag for architecture-specific optimization.
    L -- int: dimensionality of test lattice.
    n_temps -- int: number of temperature values to test.
    """
    # loading data from file
    times = np.load(rootdir + "/data/benchmarkrun.npy")

    normtime = times[:, 0, 0]
    print(f"Unoptimized {n_temps} threads, {N_list[2]} MC cycles:" +
          f" {normtime[2]} s")

    # plotting data
    plt.figure()
    for j in range(2):
        for i in range(1, len(gccflags)):
            if j == 0:
                labelstr = f"{gccflags[i]}"
            else:
                labelstr = f"{gccflags[i]}, {archflag}"

            print(labelstr +
                  f" {n_temps} threads, {N_list[2]} MC cycles:" +
                  f" {times[2, j, i]} s")

            plt.semilogx(N_list, 100*(normtime-times[:, j, i])/normtime, 'x--',
                         label=labelstr)

    plt.title(f"Timing of compiler flags for L = {L}")
    plt.xlabel("N")
    plt.ylabel("Time improvement [%]")
    plt.legend()
    plt.grid()
    plt.savefig(rootdir + "/data/benchmark.pdf")


runflag = "start"
nmax = int(1e7)
if __name__ == "__main__":
    while (runflag != "an" and runflag != "st" and runflag != "ph"
           and runflag != "b" and runflag != "test"):

        runflag = input("Analytic vs numeric 2x2 = 'an', " +
                        "stabilization run = 'st', " +
                        "phase transition = 'ph', " +
                        "OpenMP benchmark = 'b', " +
                        "Unit tests = 'test', " +
                        "quit = 'q'.\n" +
                        "Enter run: ").strip().lower()

        if runflag == "quit" or runflag == "q":
            print("Exiting.")
            sys.exit(0)

    if runflag != "test":
        genflag = input("Generate data? y/n: ").strip().lower()

    if runflag == "an":
        """Comparing numerical results for 2x2 lattice
        with analytic results."""
        temp = 1  # temperature of system.
        L = 2  # dimensionality of lattice.
        file = rootdir + "/data/2x2-comparison.dat"
        if genflag == "y":
            build_cpp()
            run(["./main.exe", file, "single", f"{L}", f"{nmax}", f"{temp}"],
                cwd=src)

        # reading data:
        data = np.genfromtxt(file)
        E = data[:, 0]
        M = data[:, 1]
        absM = data[:, 2]

        # Calculating heat capacity and magnetization per spin
        Cv = (4/(temp**2)) * (np.mean(E**2) - np.mean(E)**2)
        Xi = (4/temp) * (np.mean(M**2) - np.mean(M)**2)

        # calculating analytic results:
        E_exp = -2*np.sinh(8/temp)/(np.cosh(8/temp) + 3)
        absM_exp = (2*np.exp(8/temp) + 4)/(np.cosh(8/temp) + 3)/4
        Xi_exp = (4/(2*temp)) * ((np.exp(8/temp) + 1)/(np.cosh(8/temp) + 3))
        Cv_exp = (48/(temp**2)) * (np.cosh(8/temp)/((np.cosh(8/temp) + 3)**2))

        # printing absolute difference to terminal.
        print("Difference between analytic and numeric results:")
        print(f"<E>: {np.mean(E)-E_exp:e}")
        print(f"<M>: {np.mean(M):e}")
        print(f"<|M|>: {np.mean(absM)-absM_exp:e}")
        print(rf"$C_V$: {Cv-Cv_exp:e}")
        print(rf"$\chi$: {Xi-Xi_exp:e}")

        # printing relative error to terminal.
        print("Relative error between analytic and numeric results:")
        print(f"<E>: {abs((np.mean(E)-E_exp)/E_exp):e}")
        print(f"<|M|>: {abs((np.mean(absM)-absM_exp)/absM_exp):e}")
        print(rf"$C_V$: {abs((Cv-Cv_exp)/Cv_exp):e}")
        print(rf"$\chi$: {abs((Xi-Xi_exp)/Xi_exp):e}")

        # taking normalized, cumulative sum of data
        n_cycles = np.linspace(1, nmax, len(E))
        E2 = np.cumsum(E**2)/n_cycles
        M2 = np.cumsum(M**2)/n_cycles
        M = np.cumsum(M)/n_cycles
        E = np.cumsum(E)/n_cycles
        absM = np.cumsum(absM)/n_cycles

        # plotting data, and saving figures to /data/:
        plt.figure()
        plt.title(f"Average energy of {L}x{L} lattice with T = {temp}")
        plt.hlines(
            E_exp, 0, nmax, 'r', label="Analytic " + r"$\langle E \rangle/\, J$"
        )
        plt.plot(n_cycles, E, label=r"$\langle E \rangle/\, J$")
        plt.xlabel("N")
        plt.ylabel(r"$\langle E \rangle/\,J$")
        plt.legend()
        plt.grid()
        plt.savefig(
            rootdir + f"/data/t{temp*10}-{L}x{L}-E.pdf", bbox_inches='tight')

        plt.figure()
        plt.title(f"Average magnitude of magnetization of {L}x{L}" +
                  f" lattice with T = {temp}")
        plt.hlines(
            absM_exp, 0, nmax, 'r',
            label="Analytic " + r"$\langle | \mathcal{M} | \rangle$"
        )
        plt.plot(n_cycles, absM, label=r"$\langle | \mathcal{M} | \rangle$")
        plt.xlabel("N")
        plt.ylabel(r"$\langle | \mathcal{M} | \rangle$")
        plt.legend()
        plt.grid()
        plt.savefig(
            rootdir + f"/data/t{temp*10}-{L}x{L}-|M|.pdf", bbox_inches='tight')

        plt.figure()
        plt.title(f"Heat capacity of {L}x{L} lattice with T = {temp}")
        plt.hlines(Cv_exp, 0, nmax, 'r', label=r"Analytic $C_V$")
        plt.plot(n_cycles, ((L**2)/(temp**2)) * (E2 - E**2), label=r"$C_V$")
        plt.xlabel("N")
        plt.ylabel(r"$C_V/\, J k_B$")
        plt.legend()
        plt.grid()
        plt.savefig(
            rootdir + f"/data/t{temp*10}-{L}x{L}-Cv.pdf", bbox_inches='tight'
        )

        plt.figure()
        plt.title(f"Magnetic susceptibility of {L}x{L} lattice with T = {temp}")
        plt.hlines(Xi_exp, 0, nmax, 'r', label=r"Analytic $\chi$")
        plt.plot(n_cycles, ((L**2)/temp) * (M2 - M**2), label=r"$\chi$")
        plt.xlabel("N")
        plt.ylabel(r"$\chi$")
        plt.legend()
        plt.grid()
        plt.savefig(
            rootdir + f"/data/t{temp*10}-{L}x{L}-Xi.pdf", bbox_inches='tight'
        )

    if runflag == "st":
        """Running simulations for 20x20 lattice with T = 1 and T = 2.4."""
        L = 20  # dimensionality of lattice.
        Ts = [1, 2.4]  # temperatures of lattice.

        # filenames for unordered and ordered spin respectivly:
        randfiles = [
            rootdir +
            f"/data/{L}-{T}-random-stabilization.dat" for T in Ts
        ]
        ordefiles = [
            rootdir +
            f"/data/{L}-{T}-ordered-stabilization.dat" for T in Ts
        ]

        if genflag == "y":  # running simulations
            for i, (file1, file2) in enumerate(zip(randfiles, ordefiles)):
                stabilization_run(file1, nmax, Ts[i], L, randspin=True)
                stabilization_run(file2, nmax, Ts[i], L)

        # reading and plotting data:
        for file1, file2 in zip(randfiles, ordefiles):
            read_stabilization_data(file1)
            read_stabilization_data(file2)

    if runflag == "ph":
        """Simulating phase transitions to estimate critical temperature."""
        Ls = [40, 60, 80, 100]  # dimensionalities of lattices.
        # filenames for each simulation:
        files = [rootdir + f"/data/{L}x{L}-multi.dat" for L in Ls]
        n_temps = 8  # number of temps to simulate per subprocess.

        if genflag == "y":  # running simulations:
            build_cpp()
            phase_trans_test(nmax, Ls, files, n_temps)

        # reading data from files:
        data = read_phase_trans(nmax, Ls, files)

        # creating arrays for data:
        E = np.zeros((len(Ls), n_temps))
        M = np.zeros((len(Ls), n_temps))
        Cv = np.zeros((len(Ls), n_temps))
        Xi = np.zeros((len(Ls), n_temps))
        absM = np.zeros((len(Ls), n_temps))

        # sorting data:
        sorted = np.argsort(data[Ls[0]][:, 0])
        T = data[Ls[0]][sorted, 0]

        # unpacking data:
        for i in range(len(Ls)):
            E[i, :] = data[Ls[i]][sorted, 1]
            Cv[i, :] = data[Ls[i]][sorted, 2]
            M[i, :] = data[Ls[i]][sorted, 3]
            Xi[i, :] = data[Ls[i]][sorted, 4]
            absM[i, :] = data[Ls[i]][sorted, 5]

        # estimating critical temperature:
        TCinf, TCinf_uncertainty, TC = get_critical_temperature(Ls, Cv, Xi, T)
        print("Estimated critical temperature in thermodynamical limit: ",
              TCinf, r" $\pm$ ",TCinf_uncertainty )

        for i, L in enumerate(Ls):
            print("Estimate critical temperature " +
                  f"with {L}x{L} lattice: {TC[i]}")

        # plotting data:
        plt.figure()
        plt.title(r"Mean energy")
        for i in range(len(Ls)):
            plt.plot(T, E[i, :], label=f"{Ls[i]}x{Ls[i]}", linestyle='dashed',
                     marker='+', ms=8)
        plt.xlabel(r"$k_B T/\,J$")
        plt.ylabel(r"$\langle E \rangle/\,J$")
        plt.legend()
        plt.grid()
        plt.savefig(
            rootdir + f"/data/phase-transition-E.pdf", bbox_inches='tight'
        )

        plt.figure()
        plt.title(r"Mean magnetization")
        for i in range(len(Ls)):
            plt.plot(T, M[i, :], label=f"{Ls[i]}x{Ls[i]}",
                     linestyle='dashed', marker='+', ms=8)
        plt.xlabel(r"$k_B T/\,J$")
        plt.ylabel(r"$\langle \mathcal{M} \rangle$")
        plt.legend()
        plt.grid()
        plt.savefig(
            rootdir + "/data/phase-transition-M.pdf", bbox_inches='tight'
        )

        plt.figure()
        plt.title("Heat capacity")
        for i in range(len(Ls)):
            plt.plot(T, Cv[i, :], label=f"{Ls[i]}x{Ls[i]}",
                     linestyle='dashed', marker='+', ms=8)
        plt.xlabel(r"$k_B T/\,J$")
        plt.ylabel(r"$C_{v}/\, Jk_B$")
        plt.legend()
        plt.grid()
        plt.savefig(
            rootdir + "/data/phase-transition-Cv.pdf", bbox_inches='tight'
        )

        plt.figure()
        plt.title("Magnetic susceptibility")
        for i in range(len(Ls)):
            plt.plot(T, Xi[i, :], label=f"{Ls[i]}x{Ls[i]}",
                     linestyle='dashed', marker='+', ms=8)
        plt.xlabel(r"$k_B T/\,J$")
        plt.ylabel(r"$\chi$")
        plt.legend()
        plt.grid()
        plt.savefig(
            rootdir + "/data/phase-transition-Xi.pdf", bbox_inches='tight'
        )

        plt.figure()
        plt.title(r"Mean absolute value of magnetization")
        for i in range(len(Ls)):
            plt.plot(T, absM[i, :], label=f"{Ls[i]}x{Ls[i]}",
                     linestyle='dashed', marker='+', ms=8)
        plt.xlabel(r"$k_B T/\,J$")
        plt.ylabel(r"$\langle | \mathcal{M} | \rangle$")
        plt.legend()
        plt.grid()
        plt.savefig(
            rootdir + "/data/phase-transition-|M|.pdf", bbox_inches='tight'
        )

    if runflag == "b":
        L = 20
        n_temps = 8
        N_list = np.logspace(3, 7, 5)
        gccflags = ["-O0", "-O1", "-O2", "-O3"]
        archflag = "-march=native"
        if genflag == "y":
            benchmark(N_list, gccflags, archflag, L, n_temps)
            run(["make", "clean"], cwd=src)

        read_benchmark(N_list, gccflags, archflag, L, n_temps)

    if runflag == "test":
        run(["make", "test"], cwd=src)
        run(["./test_main.exe"], cwd=src)
        run(["python3", "-m", "pytest", "-v"])
        run(["rm", "-rf", rootdir + "/data/2x2-test.dat"])

    plt.show()
