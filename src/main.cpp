#include "metropolis.hpp"
#include <string>

int main() {
  Metropolis sim = Metropolis(2,100,20,"../data/test.dat"); // Filename included in .gitignore
  sim.run();
}
