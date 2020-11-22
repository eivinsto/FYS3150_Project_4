# FYS3150_Project_4
All code for this report was written in python 3.6, and C++.
To generate the data used in the report, run the python-script "project.py" in the root directory of this repo.

## Usage:
When executed, the "project.py" script will ask you which part of the report to run:
*   an   - Run comparisons between numerical simulation and analytic values of 2x2 lattice.
*   st  - Run simulations of 20x20 lattice for ordered and unordered spin, and for T=1 and T=2.4.
*   ph  - Run simulations of phase transition for 40x40, 60x60, 80x80 and 100x100 lattices using 8 evenly spaced temperature values from the interval \[2.2, 2.35\].
*   b  - Run benchmark and timings of optimization flags for parallelization of phase transition simulation.
*   test - Run unit tests for the Metropolis simulation of the Ising model.

```console
$ python project.py
Analytic vs numeric 2x2 = 'an', stabilization run = 'st', phase transition = 'ph', OpenMP benchmark = 'b', Unit tests = 'test', quit = 'q'.
Enter run:
```

For all runs except "test", the script will ask you if you wish to generate data for the selected run.
```console
Generate data? y/n:
```
Selecting 'y' will run the selected simulation(s), and data analysis. Selecting 'n' will not run the selected simulation(s). Instead the script will attempt to load previously generated data from the /data/ directory, and perform the data analysis.

The "test" run will generate data each time.

Example run of benchmark:
```console
$ python project.py
Analytic vs numeric 2x2 = 'an', stabilization run = 'st', phase transition = ph, OpenMP benchmark = b, Unit tests = test, quit = 'q'.
Enter run: b
Generate data? y/n: n
Unoptimized 8 threads, 100000.0 MC cycles: 6.83823 s
-O1 8 threads, 100000.0 MC cycles: 3.27981 s
-O2 8 threads, 100000.0 MC cycles: 3.5443 s
-O3 8 threads, 100000.0 MC cycles: 3.33299 s
-O1, -march=native 8 threads, 100000.0 MC cycles: 3.28577 s
-O2, -march=native 8 threads, 100000.0 MC cycles: 3.54694 s
-O3, -march=native 8 threads, 100000.0 MC cycles: 3.03317 s
```

Example run of unit tests:
```console
$ python project.py
Analytic vs numeric 2x2 = 'an', stabilization run = 'st', phase transition = 'ph', OpenMP benchmark = 'b', Unit tests = 'test', quit = 'q'.
Enter run: test
g++ -Wall -Wextra -O3 -march=native -fopenmp test_functions.cpp -c
g++ -Wall -Wextra -O3 -march=native -fopenmp test_main.o test_functions.o ising_metropolis.o -o test_main.exe -larmadillo
===============================================================================
All tests passed (4 assertions in 2 test cases)

===================================================================================== test session starts =====================================================================================
platform linux -- Python 3.8.5, pytest-6.0.1, py-1.9.0, pluggy-0.13.1 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/anders/Documents/2020H/FYS3150/FYS3150_Project_4
collected 4 items

test_functions.py::test_analytic_energy PASSED                                                                                                                                          [ 25%]
test_functions.py::test_analytic_absM PASSED                                                                                                                                            [ 50%]
test_functions.py::test_analytic_Cv PASSED                                                                                                                                              [ 75%]
test_functions.py::test_analytic_Xi PASSED                                                                                                                                              [100%]

===================================================================================== 4 passed in 32.96s ======================================================================================
```
