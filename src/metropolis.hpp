#ifndef METROPOLIS_HPP
#define METROPOLIS_HPP

#include <armadillo>
#include <string>
#include <iostream>

class Metropolis {
public:
  Metropolis (int, int, double, double, int, std::string);
  Metropolis (int, int, double, std::string);
  void run(bool randspin);
  void run();


private:
  inline int periodic(int i, int limit, int add) { return (i+limit+add)%limit; }
  void one_monte_carlo_cycle(double&, double&);
  void write_to_file_multi(arma::vec average);
  void write_to_file_single(double, double);
  void run_multi(bool randspin);
  void run_single(bool randspin);
  void initialize(bool randspin, double&, double&);
  double ran1();

  int n_spins;
  arma::Mat<int> spin_matrix;
  arma::vec temperature;
  double temp;
  int n_temps;
  int L;
  int max_cycles;
  std::string output_filename;
  arma::vec w = arma::zeros(17);
  long idum = -1;
  int current_cycle;
  std::string runflag;
  std::ofstream ofile;
  double n_spins2;
};


#endif
