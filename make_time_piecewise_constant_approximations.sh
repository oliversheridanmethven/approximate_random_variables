# Hardware agnostic, e.g.
gcc -I. -c piecewise_constant_approximation.c 
gcc -I. -O0 -c time_piecewise_constant_approximation.c
gcc -I. -o time_piecewise_constant_approximation time_piecewise_constant_approximation.o piecewise_constant_approximation.o -lgsl

# On Intel hardware, e.g.
icc -I. -std=c11 -O3 -simd -p -mkl -qopenmp -xCORE-AVX512 -march=skylake-avx512 -qopt-zmm-usage=high -c piecewise_constant_approximation.c
icc -I. -mkl -O0 -c time_piecewise_constant_approximation.c -D__PURE_INTEL_C99_HEADERS__
icc -I. -o time_piecewise_constant_approximation time_piecewise_constant_approximation.o piecewise_constant_approximation.o -lgsl -lgslcblas
