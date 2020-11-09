#ifndef METROPOLIS_HPP
#define METROPOLIS_HPP

#include <armadillo>
#include <string>
#include <iostream>

class Metropolis {
public:
  // Constructors and destructors
  Metropolis (int, int, double, double, int, std::string);
  Metropolis (int, int, double, std::string);
  ~Metropolis();

  // Functions that run simulations
  void run(bool);
  void run();


private:
  // Private functions
  void one_monte_carlo_cycle(arma::Mat<int>&, double&, double&, arma::vec);
  void write_to_file_multi(arma::vec, double);
  void write_to_file_single(double, double);
  void run_multi(bool);
  void run_single(bool);
  void initialize(bool, arma::Mat<int>&, double&, double&);
  double ran1();

  // Inline function that is used to generate periodic boundary conditions
  inline int periodic(int i, int limit, int add) { return (i+limit+add)%limit; }

  // Private variables
  int n_spins;
  arma::vec temperature;
  double temp;
  int n_temps;
  int L;
  int max_cycles;
  std::string output_filename;
  long idum = -1;
  int current_cycle;
  std::string runflag;
  std::ofstream ofile;
  double n_spins2;
  int accepted_configs = 0;
};


#endif
