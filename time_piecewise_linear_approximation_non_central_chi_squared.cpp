// Author:
//
//      Oliver Sheridan-Methven, October 2020.
//
// Description:
//
//      Timing the performance for approximations to the
//      inverse cumulative distribution function of the
//      non-central chi-squared distribution.

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include "piecewise_linear_approximation_non_central_chi_squared.hpp"
#include <boost/math/distributions/non_central_chi_squared.hpp>

int main(int argc, char **argv)
{
    // We don't want to hold to many numbers in memory at once, so we break them into batches.

#define N_LAMBDAS 8
#define N_NUS 5
    float32 lambdas[N_LAMBDAS] = {1, 5, 10, 50, 100, 200, 500, 1000};
    float32 nus[N_LAMBDAS] = {1, 5, 10, 50, 100};
    unsigned int samples_in_batch = 512 * 10;
    unsigned int n_batches = 10;
    unsigned int total_number_of_samples = samples_in_batch * n_batches;
    float32 input[samples_in_batch];
    float32 nc[samples_in_batch];
    float32 output[samples_in_batch];
    for (unsigned int i = 0; i < samples_in_batch; i++)
    {    // Random numbers in the range (0, 1), which are non-inclusive.
        input[i] = (float32) ((unsigned long int) rand() + 1) / (float32) ((unsigned long int) RAND_MAX + 2);
    }
    clock_t run_time;
    double elapsed_time;

//    printf("nu, lambda, approximate, exact");
    for (unsigned int lambda = 0; lambda < N_LAMBDAS; lambda++)
    {
        for (unsigned int nu = 0; nu < N_NUS; nu++)
        {
            for (unsigned int i = 0; i < samples_in_batch; i++)
            {
                nc[i] = lambdas[lambda];
            }
//            printf("\n%f, %f, ", nus[nu], lambdas[lambda]);
            run_time = clock();
            for (unsigned int batch = 0; batch < n_batches; batch++)
            {
                piecewise_polynomial_approximation_non_central_chi_squared(samples_in_batch, input, nc, output);
            }
            elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
//            printf("Average time for the approximate function: %g s.\n", elapsed_time / total_number_of_samples);
//            printf("%g, ", elapsed_time / total_number_of_samples);

            boost::math::non_central_chi_squared dist(nus[nu], lambdas[lambda]);
            run_time = clock();
            for (unsigned int batch = 0; batch < n_batches; batch++)
            {
                for (unsigned int sample = 0; sample < samples_in_batch; sample++)
                {
                    output[sample] = quantile(dist, input[sample]);
                }
            }
            elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
//            printf("Average time for the exact (Boost) function: %g s.\n", elapsed_time / total_number_of_samples);
//            printf("%g", elapsed_time / total_number_of_samples);
        }
    }
}