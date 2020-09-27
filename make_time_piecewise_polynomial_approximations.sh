# The OpenMP SIMD approximation using a cubic. 
gcc -I. -fopenmp -Ofast -DUSE_OPENMP_SIMD_APPROX -DPOLYNOMIAL_ORDER=3 -c piecewise_polynomial_approximation.c
gcc -I. -O0 -c time_piecewise_polynomial_approximation.c
gcc -I. -o time_piecewise_polynomial_approximation time_piecewise_polynomial_approximation.o piecewise_polynomial_approximation.o -lgsl
