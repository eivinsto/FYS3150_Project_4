#include "metropolis.hpp"
#include <string>
#include <iostream>


/**
* This program runs a Monte Carlo simulation of the Ising model with the
* Metropolis algorithm for particles in a square two-dimensional lattice.
* Command line arguments control the simulation:
*
* 1: Name of file to output data to
*
* 2: Flag that dictates whether simulations should be run for multiple
*    temperatures or just a single temperature. "multi" for multiple and
*    "single" for single temperature(s)
*    The data ouput differs depending on this flag. "multi" outputs averaged
*    quantities including heat capacity and magnetic susceptibility, and the
*    the averaged square energy and magnetization once per every simulation
*    (at a given temperature that is also added to the datafile). "single"
*    outputs energy and magnetization once per Monte Carlo cycle.
*
* 3: Number of spins in in each dimension.
*
* 4: Amount of Monte Carlo cycles to perform.
*
* 5: If the runflag is set to "single" this should be the temperature of the system.
*    This is then the last command line argument.
*    If the runflag is set to "multi" this is the minimum temperature of the system.
*    In this case, two more command line arguments are expected.
*
* (Only if runflag is set to "multi"):
* 6: Maximum temperature of the system.
*
* 7: Amount of temperature values which should be simulated.
*/
int main(int argc, char** argv) {

  // Check whether an appropriate amount of command line arguments are given
  if ( (argc!=5)&&(argc!=7) ) {
    std::cout << "Bad usage: Program must have exactly 5 or 7 command line arguments!" << std::endl;
    std::cout << "Exiting program..." << std::endl;
    return 1;
  }

  // Initialize variables and read in first four command line arguments
  std::string datafile = argv[0];
  std::string runflag = argv[1];
  int num_spins = atoi(argv[2]);
  int max_cycles = atoi(argv[3]);
  double min_temp = 1;
  double max_temp = 1;
  double num_tempsteps = 1;
  double temp = 1;

  // Read in last command line argument(s)
  if (runflag=="multi"){
    // Check if amount of command line arguments is correct pertaining to the runflag
    if (argc!=7) {
      std::cout << "Bad usage: If runflag multi is chosen, there has to be 7 command line arguments." << std::endl;
      std::cout << "Exiting program..." << std::endl;
      return 1;
    }
    // Read in command line arguments
    min_temp = atof(argv[4]);
    max_temp = atof(argv[5]);
    num_tempsteps = atof(argv[6]);
  }
  else{
    // Check if amount of command line arguments is correct pertaining to the runflag
    if (argc!=5) {
      std::cout << "Bad usage: If runflag single is chosen, there has to be 5 command line arguments." << std::endl;
      std::cout << "Exiting program..." << std::endl;
      return 1;
    }
    // Read in command line argument
    temp = atof(argv[4]);
  }

  // Run simulation
  if (runflag=="multi") {
    Metropolis simulation = Metropolis(num_spins,max_cycles,min_temp,max_temp,num_tempsteps,datafile);
    simulation.run();
  }

  if (runflag=="single") {
    Metropolis simulation = Metropolis(num_spins,max_cycles,temp,datafile);
    simulation.run();
  }
}
