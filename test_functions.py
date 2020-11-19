import pytest
import project as p
from subprocess import run
import numpy as np

temp = 1  # temperature of system.
L = 2  # dimensionality of lattice.
file = p.rootdir + "/data/2x2-test.dat"
nmax = int(3e6)
p.build_cpp()
run(["./main.exe", file, "single", f"{L}", f"{nmax}", f"{temp}"],
    cwd=p.src)

# reading data:
data = np.genfromtxt(file)
E = data[:, 0]
M = data[:, 1]
absM = data[:, 2]


def test_analytic_energy():
    """Unit test of <E> of metropolis ising simulation of 2x2 lattice."""

    # calculating analytic results:
    E_exp = -2*np.sinh(8/temp)/(np.cosh(8/temp) + 3)
    success = np.isclose(np.mean(E), E_exp, rtol=1e-3)

    msg = f"Numeric <E> = {np.mean(E)} != analytic <E> = {E_exp}"
    assert success, msg


def test_analytic_absM():
    """Unit test of <|M|> of metropolis ising simulation of 2x2 lattice."""

    # calculating analytic results:
    absM_exp = ((2*np.exp(8/temp) + 4)/(np.cosh(8/temp) + 3))/4
    success = np.isclose(np.mean(absM), absM_exp, rtol=1e-3)

    msg = f"Numeric <|M|> = {np.mean(absM)} != analytic <|M|> = {absM_exp}"
    assert success, msg


def test_analytic_Cv():
    """Unit test of <Cv> of metropolis ising simulation of 2x2 lattice."""

    # calculating analytic results:
    Cv = (1/(temp**2)) * (np.mean(E**2) - np.mean(E)**2)
    Cv_exp = (12/(temp**2)) * (np.cosh(8/temp)/((np.cosh(8/temp) + 3)**2))
    success = np.isclose(Cv, Cv_exp, rtol=5e-2)

    msg = f"Numeric <Cv> = {Cv} != analytic <Cv> = {Cv_exp}"
    assert success, msg


def test_analytic_Xi():
    """Unit test of <Xi> of metropolis ising simulation of 2x2 lattice."""

    # calculating analytic results:
    Xi = (1/temp) * (np.mean(M**2) - np.mean(M)**2)
    Xi_exp = (1/(2*temp)) * ((np.exp(8/temp) + 1)/(np.cosh(8/temp) + 3))
    success = np.isclose(Xi, Xi_exp, rtol=1e-2)

    msg = f"Numeric <Xi> = {Xi} != analytic <Xi> = {Xi_exp}"
    assert success, msg
