# The OpenMP SIMD approximation using a cubic. 
gcc -I. -fopenmp -Ofast -DUSE_OPENMP_SIMD_APPROX -DPOLYNOMIAL_ORDER=3 -c piecewise_polynomial_approximation.c
gcc -I. -O0 -c time_piecewise_polynomial_approximation.c
gcc -I. -o time_piecewise_polynomial_approximation time_piecewise_polynomial_approximation.o piecewise_polynomial_approximation.o -lgsl

# Using Intel intrinsics.
icc -I. -DUSE_INTEL_INTRINSICS_APPROX_LINEAR -DPOLYNOMIAL_ORDER=1 -std=c11 -O3 -simd -p -mkl -qopenmp -xCORE-AVX512 -march=skylake-avx512 -qopt-zmm-usage=high -c piecewise_polynomial_approximation.c
icc -I. -DCOMPARE_AGAINST_MKL -mkl -O0 -c time_piecewise_polynomial_approximation.c  -D__PURE_INTEL_C99_HEADERS__
icc -I. -o time_piecewise_polynomial_approximation time_piecewise_polynomial_approximation.o piecewise_polynomial_approximation.o -lgsl -lgslcblas -mkl
