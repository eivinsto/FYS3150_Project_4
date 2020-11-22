#include "catch.hpp"
#include "ising_metropolis.hpp"
#include <string>
#include <fstream>

bool file_exists(std::string filename) {
  std::ifstream infile(filename);
  return infile.good();
}

TEST_CASE("Test file creation on IsingMetropolis multi-constructor call.") {
  std::string filename = "../data/multi-filetest.dat";

  REQUIRE(!file_exists(filename));
  {
    IsingMetropolis IsingMetropolis(3, 100, 1, 3, 4, filename);
  }
  REQUIRE(file_exists(filename));
  std::remove(filename.c_str());
}


TEST_CASE("Test file creation on IsingMetropolis single-constructor call.") {
  std::string filename = "../data/single-filetest.dat";

  REQUIRE(!file_exists(filename));
  {
    IsingMetropolis IsingMetropolis(3, 100, 1, filename);
  }
  REQUIRE(file_exists(filename));
  std::remove(filename.c_str());
}
