#include "metropolis.hpp"
#include <armadillo>
#include "lib.h"  // For RNG ran1(int)
#include <cmath>
#include <iostream>
#include <iomanip>


Metropolis::Metropolis (int num_spins, int dim, int num_mcs, double min_temp, double max_temp, double temp_step, std::string filename){
    n_temps = int((max_temp-min_temp)/temp_step);
    temperature = arma::linspace(min_temp,max_temp,n_temps);
    n_spins = num_spins;
    L = dim;
    max_cycles = num_mcs;
    output_filename = filename;
    spin_matrix.ones(L,L);
    runflag = "multi";
}

Metropolis::Metropolis (int num_spins, int dim, int num_mcs, double input_temp, std::string filename){
    temp = input_temp;
    n_spins = num_spins;
    L = dim;
    max_cycles = num_mcs;
    output_filename = filename;
    spin_matrix.ones(L,L);
    runflag = "single";
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
        M += double(2*spin_matrix(iy,ix));
        E += double(deltaE);
      }
    }
  }
}


void Metropolis::initialize() {

  // Reset spin_matrix
  spin_matrix.ones(L,L);

  // Initial magnetization
  M = double(L*L);

  // Initial energy
  for(int y =0; y < n_spins; y++) {
    for (int x= 0; x < n_spins; x++){
      E -=  double(spin_matrix(y,x)*
	          (spin_matrix(periodic(y,n_spins,-1),x) +
	          spin_matrix(y,periodic(x,n_spins,-1))));
    }
  }
}


void Metropolis::run() {
  if (runflag=="single") run_single();
  else run_multi();
}


void Metropolis::run_multi() {
  for (int i = 0; i<n_temps; ++i) {
    temp = temperature(i);
    E = 0;
    M = 0;
    w.zeros();
    for (int de = -8; de<= 8; de+=4) {
      w(de+8) = exp(-de/temp);
    }

    for (int current_cycle = 1; current_cycle<=max_cycles; current_cycle++){
      one_monte_carlo_cycle();
      average(0) += E;
      average(1) += E*E;
      average(2) += M;
      average(3) += M*M;
      average(4) += fabs(M);
    }
    write_to_file(max_cycles);
  }
}


void Metropolis::run_single() {
  E = 0;
  M = 0;
  w.zeros();
  for (int de = -8; de<= 8; de+=4) {
    w(de+8) = exp(-de/temp);
  }

  for (int current_cycle = 1; current_cycle<=max_cycles; current_cycle++){
    one_monte_carlo_cycle();

    // Technically these will not provide averages, but we pass them in the same array
    // so that write_to_file() prints them to file.
    average(0) = E;
    average(1) = E*E;
    average(2) = M;
    average(3) = M*M;
    average(4) = fabs(M);
    write_to_file(1);
  }
}


void Metropolis::write_to_file(int mcs) {
  if(!ofile.good()) {
    ofile.open(output_filename.c_str(), std::ofstream::out);
    if(!ofile.good()) {
      std::cout << "Error opening file " << output_filename << ". Aborting!" << std::endl;
      std::terminate();
    }
  }

  double norm = 1/(double(mcs));  // divided by total number of cycles
  double Eaverage = average(0)*norm;
  double E2average = average(1)*norm;
  double Maverage = average(2)*norm;
  double M2average = average(3)*norm;
  double Mabsaverage = average(4)*norm;

  // all expectation values are per spin, divide by 1/n_spins/n_spins
  double Evariance = (E2average- Eaverage*Eaverage)/n_spins/n_spins;
  double Mvariance = (M2average - Mabsaverage*Mabsaverage)/n_spins/n_spins;
  ofile << std::setiosflags(ios::showpoint | ios::uppercase);
  ofile << std::setw(15) << std::setprecision(8) << temperature;
  ofile << std::setw(15) << std::setprecision(8) << Eaverage/n_spins/n_spins;
  ofile << std::setw(15) << std::setprecision(8) << Evariance/temperature/temperature;
  ofile << std::setw(15) << std::setprecision(8) << Maverage/n_spins/n_spins;
  ofile << std::setw(15) << std::setprecision(8) << Mvariance/temperature;
  ofile << std::setw(15) << std::setprecision(8) << Mabsaverage/n_spins/n_spins << std::endl;
}
