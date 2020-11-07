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
    run(["./main.exe",file,"single","2","100",f"{temp}"],cwd=src)
    data = read_single_run_file(src+file)

    E = data[:,0]
    M = data[:,1]
    absM = data[:,2]

    print(np.mean(E))
