# The OpenMP SIMD approximation.
g++ -I. -D_GLIBCXX_USE_CXX11_ABI=0 -fopenmp -Ofast -DPOLYNOMIAL_ORDER=1 -c piecewise_linear_approximation_non_central_chi_squared.cpp -o piecewise_linear_approximation_non_central_chi_squared_cpp.o
g++ -I. -D_GLIBCXX_USE_CXX11_ABI=0 -O0 -c time_piecewise_linear_approximation_non_central_chi_squared.cpp -o time_piecewise_linear_approximation_non_central_chi_squared_cpp.o -I/usr/local/Cellar/boost/1.74.0/include/
g++ -I. -D_GLIBCXX_USE_CXX11_ABI=0 -o time_piecewise_linear_approximation_non_central_chi_squared time_piecewise_linear_approximation_non_central_chi_squared_cpp.o piecewise_linear_approximation_non_central_chi_squared_cpp.o

icpc -I. -D_GLIBCXX_USE_CXX11_ABI=0 -fopenmp -Ofast -DPOLYNOMIAL_ORDER=1 -c piecewise_linear_approximation_non_central_chi_squared.cpp -o piecewise_linear_approximation_non_central_chi_squared_cpp.o
icpc -I. -D_GLIBCXX_USE_CXX11_ABI=0 -O0 -c time_piecewise_linear_approximation_non_central_chi_squared.cpp -o time_piecewise_linear_approximation_non_central_chi_squared_cpp.o -I/usr/local/Cellar/boost/1.74.0/include/
icpc -I. -D_GLIBCXX_USE_CXX11_ABI=0 -o time_piecewise_linear_approximation_non_central_chi_squared time_piecewise_linear_approximation_non_central_chi_squared_cpp.o piecewise_linear_approximation_non_central_chi_squared_cpp.o