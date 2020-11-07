#include "metropolis.hpp"
#include <armadillo>
#include "lib.h"  // For RNG ran1(int)


Metropolis::Metropolis (int num_spins, int dim, int num_mcs, double min_temp, double max_temp, double temp_step, std::string filename){
    int N = int((max_temp-min_temp)/temp_step);
    temperature = arma::linspace(min_temp,max_temp,N);
    n_spins = num_spins;
    L = dim;
    mcs = num_mcs;
    output_filename = filename;
}

void Metropolis::one_monte_carlo_cycle() {
  // Loop over all spins
  for(int y =0; y < n_spins; y++) {
    for (int x= 0; x < n_spins; x++){

      // Get random coordinates
      int ix = int(ran1(&idum)*(double)n_spins);
      int iy = int(ran1(&idum)*(double)n_spins);

      // Calculate change in energy
      int deltaE =  2*spin_matrix(iy,ix)*
                    (spin_matrix(iy,periodic(ix,n_spins,-1))+
                    spin_matrix(periodic(iy,n_spins,-1),ix) +
                    spin_matrix(iy,periodic(ix,n_spins,1)) +
                    spin_matrix(periodic(iy,n_spins,1),ix));

      // Flip spin if new config is accepted
      if ( ran1(&idum) <= w[deltaE+8] ) {
        spin_matrix(iy,ix) *= -1;

        // Update energy and magnetization if spin is flipped
        M += (double) 2*spin_matrix(iy,ix);
        E += (double) deltaE;
      }
    }
  }
}
