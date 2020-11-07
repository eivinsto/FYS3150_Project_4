CC := g++
FLAGS := -Wall -Wextra -O2 -march=native
DEBUG := -g
LIBS := -larmadillo
OFILES := metropolis.o lib.o

.PHONY: all clean benchmark test

# compile code:
all: main.exe

# compile program:
main.exe: main.o $(OFILES)
	$(CC) $(FLAGS) $(DEBUG) main.o $(OFILES) -o main.exe $(LIBS)

main.o: main.cpp
	$(CC) $(FLAGS) $(DEBUG) main.cpp -c

metropolis.o: metropolis.cpp
	$(CC) $(FLAGS) $(DEBUG) metropolis.cpp -c

lib.o: lib.cpp
	$(CC) $(FLAGS) $(DEBUG) lib.cpp -c


# unittest program:
test: test_main.exe

test_main.exe: test_main.o test_functions.o $(OFILES)
	$(CC) $(FLAGS) test_main.o test_functions.o $(OFILES) -o test_main.exe $(LIBS)

test_main.o: test_main.cpp
	$(CC) $(FLAGS) test_main.cpp -c

test_functions.o: test_functions.cpp
	$(CC) $(FLAGS) test_functions.cpp -c

# benchmark
benchmark: benchmark.exe

benchmark.exe: benchmark.o $(OFILES)
	$(CC) $(FLAGS) benchmark.o $(OFILES) -o benchmark.exe $(LIBS)

benchmark.o: benchmark.cpp
	$(CC) $(FLAGS) benchmark.cpp -c

# project cleaning:
clean:
	rm -f *.o
	rm -f *.exe
	rm -f *.dat

print-% : ; @echo $* = $($*)