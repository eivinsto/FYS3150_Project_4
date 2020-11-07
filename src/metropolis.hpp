#ifndef METROPOLIS_HPP
#define METROPOLIS_HPP

#include <armadillo>
#include <string>
#include <iostream>

class Metropolis {
public:
  Metropolis (int, int, double, double, int, std::string);
  Metropolis (int, int, double, std::string);
  void run();


private:
  inline int periodic(int i, int limit, int add) { return (i+limit+add)%limit; }
  void one_monte_carlo_cycle();
  void write_to_file_multi();
  void write_to_file_single();
  void run_multi();
  void run_single();
  void initialize();

  int n_spins;
  arma::Mat<int> spin_matrix;
  arma::vec temperature;
  double temp;
  int n_temps;
  double E;
  double M;
  int L;
  int max_cycles;
  std::string output_filename;
  arma::vec w = arma::zeros(17);
  long idum = -1;
  int current_cycle;
  std::string runflag;
  std::ofstream ofile;
  arma::vec average = arma::zeros(5);
  double n_spins2;
};


#endif
