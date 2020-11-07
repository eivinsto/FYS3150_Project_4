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

if __name__=="__main__":


    build_cpp()
    temp = 3
    file = "../data/test.dat"
    run(["./main.exe",file,"single","2","100000",f"{temp}"],cwd=src)
    data = read_single_run_file(src+file)

    E = data[:,0]
    M = data[:,1]
    absM = data[:,2]

    E_exp = -2*np.sinh(8/temp)/(np.cosh(8/temp) + 3)
    absM_exp = (2*np.exp(8/temp) + 4)/(np.cosh(8/temp) + 3)/4

    print(np.mean(E)-E_exp)
    print(np.mean(M))
    print(np.mean(absM)-absM_exp)
