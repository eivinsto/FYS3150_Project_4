#include "metropolis.hpp"
#include <armadillo>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <ctime>


/**
* Constructor that takes a minimum temperature, a max temperature, and amount
* of steps for the temperature and sets up an array that contains temperature
* values. This constructor also initializes variables that store the amount of
* spins, the amount of Monte Carlo cycles to perform and the name of the output
* file.
* @num_spins -- number of spins in each dimension
* @num_mcs -- number of Monte Carlo cycles to perform
* @min_temp -- lowest temperature to simulate at
* @max_temp -- largest temperature to simulate at
* @temp_steps -- amount of steps in temperature
* @filename -- name of file to write resulting data from simulation to
*/
Metropolis::Metropolis (int num_spins, int num_mcs, double min_temp, double max_temp, int temp_steps, std::string filename){
    n_temps = temp_steps;                                      // Number of temperature steps
    temperature = arma::linspace(min_temp,max_temp,n_temps);   // Setting temperature array
    n_spins = num_spins;                                       // Number of spins to model
    n_spins2 = n_spins*n_spins;                                // Number of spins squared (for use in write_to_file())
    max_cycles = num_mcs;                                      // Number of Monte Carlo cycles to perform
    output_filename = filename;                                // Output file name
    runflag = "multi";                                         // Runflag depending on temperature initialization
}

/**
* Constructor that initializes variables that store the amount of
* spins, the amount of Monte Carlo cycles to perform, the name of the output
* file and the temperature to simulate at.
* @num_spins -- number of spins in each dimension
* @num_mcs -- number of Monte Carlo cycles to perform
* @input_temp -- temperature to run simulation at
*/
Metropolis::Metropolis (int num_spins, int num_mcs, double input_temp, std::string filename){
    temp = input_temp;                    // Setting temperature
    n_spins = num_spins;                  // Number of spins
    n_spins2 = n_spins*n_spins;           // Number of spins squared (for use in write_to_file())
    max_cycles = num_mcs;                 // Number of Monte Carlo cycles to perform
    output_filename = filename;           // Output file name
    runflag = "single";                   // Runflag depending on temperature initialization
}

/**
* Member function that performs one Monte Carlo cycle on the system, using
* the Metropolis algorithm.
* @spin_matrix -- matrix storing spin values
* @E -- double containing energy of system
* @M -- double containing magnetization of system
* @w -- vector containing "probabilities" for energy changes
*/
void Metropolis::one_monte_carlo_cycle(arma::Mat<int> &spin_matrix, double &E, double &M, arma::vec w) {
  // Loop over all spins
  for(int y =0; y < n_spins; y++) {
    for (int x= 0; x < n_spins; x++){

      // Get random indices
      int ix = int(ran1()*(double)n_spins);
      int iy = int(ran1()*(double)n_spins);

      // Calculate change in energy
      int deltaE =  2*spin_matrix(iy,ix)*
                    (spin_matrix(iy,periodic(ix,n_spins,-1))+
                    spin_matrix(periodic(iy,n_spins,-1),ix) +
                    spin_matrix(iy,periodic(ix,n_spins,1)) +
                    spin_matrix(periodic(iy,n_spins,1),ix));


      // Flip spin if new config is accepted
      if ( ran1() <= w(deltaE+8) ) {
        spin_matrix(iy,ix) *= -1;


        // Update energy and magnetization if spin is flipped
        M += double(2*spin_matrix(iy,ix));
        E += double(deltaE);

        // Count accepted configs
        accepted_configs++;
      }
    }
  }
}

/**
* Member function that resets the spin matrix, magnetization and energy.
* @randspin -- bool, generates random spin_matrix if true.
* @spin_matrix -- matrix storing spin values
* @E -- double containing energy
* @M -- double containing magnetization
*/
void Metropolis::initialize(bool randspin, arma::Mat<int> &spin_matrix, double &E, double &M) {

  // Reset spin_matrix
  if (randspin) {
    srand(time(NULL));
    for (int x = 0; x < n_spins; x++) {
      for (int y = 0; y < n_spins; y++) {
        spin_matrix(x,y) = (rand() > RAND_MAX/2) ? -1 : 1;
      }
    }
  } else {
    spin_matrix.ones();
  }

  // Initial magnetization
  M = n_spins2;

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
  if (runflag=="single") run_single(false);
  else run_multi(false);
}

/**
* Member function that calls the correct function to run the simulation.
* This calls either run_multi() or run_single() depending on the runflag.
* @randspin -- specifies whether or not the spin_matrix should be randomly generated
*/
void Metropolis::run(bool randspin) {
  if (runflag=="single") run_single(randspin);
  else run_multi(randspin);
}


/**
* Member function that runs the simulation for multiple values of the
* temperature. In these kinds of simulations the average energy, magnetization
* and other values derived from these are written to file after each simulation
* is completed.
* @randspin -- specifies whether or not the spin_matrix should be randomly generated
*/
void Metropolis::run_multi(bool randspin) {
  // Parallelized for loop
  #pragma omp parallel for
  for (int i = 0; i<n_temps; ++i) {
    // Define local variables so the different cores do not update the same
    // variables at the same time.
    arma::vec average = arma::zeros(5);
    double E = 0;
    double M = 0;
    arma::Mat<int> spin_matrix(n_spins,n_spins,arma::fill::ones);
    arma::vec w = arma::zeros(17);
    for (int de = -8; de<= 8; de+=4) {
      w(de+8) = exp(-de/temperature(i));
    }

    // Initialize spin matrix
    initialize(randspin,spin_matrix,E,M);

    for (int current_cycle = 1; current_cycle<=max_cycles; current_cycle++){
      one_monte_carlo_cycle(spin_matrix,E,M,w);
      average(0) += E;
      average(1) += E*E;
      average(2) += M;
      average(3) += M*M;
      average(4) += fabs(M);
    }
    // Only one core should call this function at a time
    #pragma omp critical
    write_to_file_multi(average);
  }
}


/**
* Member function that runs the simulation for a single temperature value
* specified upon instantiating the class. The energy, magnetization, and values
* derived from these are in this case written to file once every Monte Carlo cycle.
* @randspin -- specifies whether or not the spin_matrix should be randomly generated
*/
void Metropolis::run_single(bool randspin) {
  double E = 0;
  double M = 0;
  arma::Mat<int> spin_matrix(n_spins,n_spins,arma::fill::ones);
  arma::vec w = arma::zeros(17);
  for (int de = -8; de<= 8; de+=4) {
    w(de+8) = exp(-de/temp);
  }

  initialize(randspin,spin_matrix,E,M);

  for (int current_cycle = 1; current_cycle<=max_cycles; current_cycle++){
    one_monte_carlo_cycle(spin_matrix,E,M,w);
    write_to_file_single(E,M);
  }
}

/**
* Member function used to write data to file when runflag is multi.
* @average -- vector containing averaged values of:
*             -Energy
*             -Energy squared
*             -Magnetization
*             -Magnetization squared
*             -Absolute of magnetization
*             ... in that order.
*/
void Metropolis::write_to_file_multi(arma::vec average) {
  if(!ofile.good()) {
    ofile.open(output_filename.c_str(), std::ofstream::out);
    if(!ofile.good()) {
      std::cout << "Error opening file " << output_filename << ". Aborting!" << std::endl;
      std::terminate();
    }
  }

  double norm = 1/(double(max_cycles));  // divided by total number of cycles
  double Eaverage = average(0)*norm;
  double E2average = average(1)*norm;
  double Maverage = average(2)*norm;
  double M2average = average(3)*norm;
  double Mabsaverage = average(4)*norm;

  // all expectation values are per spin, divide by 1/(n_spins^2)
  double Evariance = (E2average- Eaverage*Eaverage)/n_spins2;
  double Mvariance = (M2average - Mabsaverage*Mabsaverage)/n_spins2;
  ofile << std::setiosflags(std::ios::showpoint | std::ios::uppercase);
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


/**
* Member function used to write to file when runflag is single.
* @E -- double containing energy of system
* @M -- doouble containing magnetization of system
*/
void Metropolis::write_to_file_single(double E, double M) {
  if(!ofile.good()) {
    ofile.open(output_filename.c_str(), std::ofstream::out);
    if(!ofile.good()) {
      std::cout << "Error opening file " << output_filename << ". Aborting!" << std::endl;
      std::terminate();
    }
  }

  ofile << std::setiosflags(std::ios::showpoint | std::ios::uppercase);
  // Writing average energy per spin
  ofile << std::setw(15) << std::setprecision(8) << E/n_spins2;
  // Writing average magnetization per spin
  ofile << std::setw(15) << std::setprecision(8) << M/n_spins2;
  // Writing average absolute of magnetization per spin
  ofile << std::setw(15) << std::setprecision(8) << fabs(M)/n_spins2;
  // Write amount of accepted configs
  ofile << accepted_configs << std::endl;
}


/*
** This function was "borrowed" from:
** https://github.com/CompPhysics/ComputationalPhysics/blob/master/doc/Programs/LecturePrograms/programs/cppLibrary/lib.cpp
** The function
**           ran1()
** is an "Minimal" random number generator of Park and Miller
** (see Numerical recipe page 280) with Bays-Durham shuffle and
** added safeguards. Call with idum a negative integer to initialize (NB!: this
** was placed as a member variable of the class instead);
** thereafter, do not alter idum between sucessive deviates in a
** sequence. RNMX should approximate the largest floating point value
** that is less than 1.
** The function returns a uniform deviate between 0.0 and 1.0
** (exclusive of end-point values).
*/

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double Metropolis::ran1() {
  int             j;
  long            k;
  static long     iy=0;
  static long     iv[NTAB];
  double          temp;

  if (idum <= 0 || !iy) {
    if (-idum < 1) idum=1;
    else idum = -idum;
    for(j = NTAB + 7; j >= 0; j--) {
      k     = idum/IQ;
      idum = IA*(idum - k*IQ) - IR*k;
      if(idum < 0) idum += IM;
      if(j < NTAB) iv[j] = idum;
    }
    iy = iv[0];
  }
  k     = (idum)/IQ;
  idum = IA*(idum - k*IQ) - IR*k;
  if(idum < 0) idum += IM;
  j     = iy/NDIV;
  iy    = iv[j];
  iv[j] = idum;
  if((temp=AM*iy) > RNMX) return RNMX;
  else return temp;
}
#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
