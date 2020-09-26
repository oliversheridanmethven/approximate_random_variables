gcc -I. -c piecewise_constant_approximation.c 
gcc -I. -O0 -c time_gaussian_approximations.c
gcc -I. -o time_approximations time_gaussian_approximations.o piecewise_constant_approximation.o -lgsl
