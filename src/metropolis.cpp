#include "metropolis.hpp"
#include <armadillo>
#include "lib.h"  // For RNG ran1(int)
#include <cmath>
#include <iostream>
#include <iomanip>


/**
* Constructor that takes a minimum temperature, a max temperature, and amount
* of steps for the temperature and sets up an array that contains temperature
* values. This constructor also initializes variables that store the amount of
* spins, the amount of Monte Carlo cycles to perform and the name of the output
* file.
*/
Metropolis::Metropolis (int num_spins, int num_mcs, double min_temp, double max_temp, int temp_steps, std::string filename){
    n_temps = temp_steps;                                      // Number of temperature steps
    temperature = arma::linspace(min_temp,max_temp,n_temps);   // Setting temperature array
    n_spins = num_spins;                                       // Number of spins to model
    n_spins2 = n_spins*n_spins;                                // Number of spins squared (for use in write_to_file())
    max_cycles = num_mcs;                                      // Number of Monte Carlo cycles to perform
    output_filename = filename;                                // Output file name
    spin_matrix.ones(n_spins,n_spins);                         // Set spin matrix
    runflag = "multi";                                         // Runflag depending on temperature initialization
}

/**
* Constructor that initializes variables that store the amount of
* spins, the amount of Monte Carlo cycles to perform, the name of the output
* file and the temperature to simulate at.
*/
Metropolis::Metropolis (int num_spins, int num_mcs, double input_temp, std::string filename){
    temp = input_temp;                    // Setting temperature
    n_spins = num_spins;                  // Number of spins
    n_spins2 = n_spins*n_spins;           // Number of spins squared (for use in write_to_file())
    max_cycles = num_mcs;                 // Number of Monte Carlo cycles to perform
    output_filename = filename;           // Output file name
    spin_matrix.ones(n_spins,n_spins);    // Set spin matrix
    runflag = "single";                   // Runflag depending on temperature initialization
}

/**
* Member function that performs one Monte Carlo cycle on the system, using
* the Metropolis algorithm.
*/
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


/**
* Member function that resets the spin matrix, magnetization and energy.
*/
void Metropolis::initialize() {

  // Reset spin_matrix
  spin_matrix.ones();

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

/**
* Member function that calls the correct function to run the simulation.
* This calls either run_multi() or run_single() depending on the runflag.
*/
void Metropolis::run() {
  if (runflag=="single") run_single();
  else run_multi();
}


/**
* Member function that runs the simulation for multiple values of the
* temperature. In these kinds of simulations the average energy, magnetization
* and other values derived from these are written to file after each simulation
* is completed.
*/
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


/**
* Member function that runs the simulation for a single temperature value
* specified upon instantiating the class. The energy, magnetization, and values
* derived from these are in this case written to file once every Monte Carlo cycle.
*/
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

/**
* Member function used to write data to file.
*/
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

  // all expectation values are per spin, divide by 1/(n_spins^2)
  double Evariance = (E2average- Eaverage*Eaverage)/n_spins2;
  double Mvariance = (M2average - Mabsaverage*Mabsaverage)/n_spins2;
  ofile << std::setiosflags(ios::showpoint | ios::uppercase);
  // Writing temperature
  ofile << std::setw(15) << std::setprecision(8) << temperature;
  // Writing average energy per spin
  ofile << std::setw(15) << std::setprecision(8) << Eaverage/n_spins2;
  // Writing heat capacity
  ofile << std::setw(15) << std::setprecision(8) << Evariance/(temperature*temperature);
  // Writing average magnetization per spin
  ofile << std::setw(15) << std::setprecision(8) << Maverage/n_spins2;
  // Writing susceptibility
  ofile << std::setw(15) << std::setprecision(8) << Mvariance/temperature;
  // Writing average absolute of magnetization per spin
  ofile << std::setw(15) << std::setprecision(8) << Mabsaverage/n_spins2 << std::endl;
}
