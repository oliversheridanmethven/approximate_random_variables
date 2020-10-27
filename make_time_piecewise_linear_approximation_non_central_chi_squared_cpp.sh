# The OpenMP SIMD approximation.
c++ -I. -fopenmp -Ofast -DPOLYNOMIAL_ORDER=1 -c piecewise_linear_approximation_non_central_chi_squared.cpp -o piecewise_linear_approximation_non_central_chi_squared_cpp.o
c++ -I. -O0 -c time_piecewise_linear_approximation_non_central_chi_squared.cpp -o time_piecewise_linear_approximation_non_central_chi_squared_cpp.o
c++ -I. -o time_piecewise_linear_approximation_non_central_chi_squared_cpp time_piecewise_linear_approximation_non_central_chi_squared_cpp.o piecewise_linear_approximation_non_central_chi_squared_cpp.o
