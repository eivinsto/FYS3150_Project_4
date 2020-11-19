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
