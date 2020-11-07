#ifndef METROPOLIS_HPP
#define METROPOLIS_HPP

#include <armadillo>
#include <string>

class Metropolis {
public:
  Metropolis (int, int, int, double, double, double, std::string);
  void run();


private:
  inline int periodic(int i, int limit, int add) { return (i+limit+add)%limit; }
  void one_monte_carlo_cycle();
  void write_to_file();

  int n_spins;
  arma::imat spin_matrix;
  arma::vec temperature;
  double E;
  double M;
  int L;
  int mcs;
  std::string output_filename;
  arma::vec w;
  long idum = -1;
};


#endif
