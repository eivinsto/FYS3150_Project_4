#include "metropolis.hpp"
#include <string>
#include <iostream>

int main(int argc, char** argv) {

  if ( (argc!=5)&&(argc!=7) ) {
    std::cout << "Bad usage: Program must have exactly 5 or 7 command line arguments!" << std::endl;
    std::cout << "Exiting program..." << std::endl;
    return 1;
  }

  std::string datafile = argv[0];
  std::string runflag = argv[1];
  int num_spins = atoi(argv[2]);
  int max_cycles = atoi(argv[3]);
  double min_temp = 1;
  double max_temp = 1;
  double num_tempsteps = 1;
  double temp = 1;

  if (runflag=="multi"){
    if (argc!=7) {
      std::cout << "Bad usage: If runflag multi is chosen, there has to be 7 command line arguments." << std::endl;
      std::cout << "Exiting program..." << std::endl;
      return 1;
    }
    min_temp = atof(argv[4]);
    max_temp = atof(argv[5]);
    num_tempsteps = atof(argv[6]);
  }
  else{
    if (argc!=5) {
      std::cout << "Bad usage: If runflag single is chosen, there has to be 5 command line arguments." << std::endl;
      std::cout << "Exiting program..." << std::endl;
      return 1;
    }
    temp = atof(argv[4]);
  }

  if (runflag=="multi") {
    Metropolis simulation = Metropolis(num_spins,max_cycles,min_temp,max_temp,num_tempsteps,datafile);
    simulation.run();
  }

  if (runflag=="single") {
    Metropolis simulation = Metropolis(num_spins,max_cycles,temp,datafile);
    simulation.run();
  }
}
