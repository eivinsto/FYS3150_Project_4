import pytest
import project as p
from subprocess import run
import numpy as np


def test_analytic():
    """Unit test of metropolis ising simulation of 2x2 lattice."""
    tol = 1e-3
    temp = 1  # temperature of system.
    L = 2  # dimensionality of lattice.
    file = p.rootdir + "/data/2x2-test.dat"
    nmax = int(1e6)
    p.build_cpp()
    run(["./main.exe", file, "single", f"{L}", f"{nmax}", f"{temp}"],
        cwd=p.src)

    # reading data:
    data = np.genfromtxt(file)
    E = np.mean(data[:, 0])
    absM = np.mean(data[:, 2])

    # calculating analytic results:
    E_exp = -2*np.sinh(8/temp)/(np.cosh(8/temp) + 3)
    absM_exp = (2*np.exp(8/temp) + 4)/(np.cosh(8/temp) + 3)/4
    errors = []
    if not np.isclose(E, E_exp, rtol=tol):
        errors.append(f"Numeric <E> = {E} != analytic <E> = {E_exp}")
    if not np.isclose(absM, absM_exp, rtol=tol):
        errors.append(f"Numeric <E> = {absM} != analytic <E> = {absM_exp}")

    msg = "Errors: \n{}".format("\n".join(errors))
    assert not errors, msg
