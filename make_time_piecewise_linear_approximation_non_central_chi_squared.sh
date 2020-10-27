# The OpenMP SIMD approximation.
gcc -I. -fopenmp -Ofast -DPOLYNOMIAL_ORDER=1 -c piecewise_linear_approximation_non_central_chi_squared.c
gcc -I. -O0 -c time_piecewise_linear_approximation_non_central_chi_squared.c
gcc -I. -o time_piecewise_linear_approximation_non_central_chi_squared time_piecewise_linear_approximation_non_central_chi_squared.o piecewise_linear_approximation_non_central_chi_squared.o -lgsl -lcdflib
