gcc -I. -c piecewise_constant_approximation.c 
gcc -I. -O0 -c time_piecewise_constant_approximation.c
gcc -I. -o time_piecewise_constant_approximation time_piecewise_constant_approximation.o piecewise_constant_approximation.o -lgsl
