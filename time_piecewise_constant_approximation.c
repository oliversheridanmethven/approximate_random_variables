// Author:
//
//      Oliver Sheridan-Methven, September 2020.
//
// Description:
//
//      Timing the performance for approximations to the
//      Gaussian's inverse cumulative distribution function.

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_cdf.h>
#include "piecewise_constant_approximation.h"

int main(int argc, char **argv)
{
    /* We don't want to hold to many numbers in memory at once, so we break them into batches. */
    unsigned int samples_in_batch = 512 * 1000;
    unsigned int n_batches = 1000;
    unsigned int total_number_of_samples = samples_in_batch * n_batches;
    double input[samples_in_batch];
    double output[samples_in_batch];
    for (unsigned int i = 0; i < samples_in_batch; i++)
    {   /* Random numbers in the range (0, 1), which are non-inclusive. */
        input[i] = (double) ((unsigned long int) rand() + 1) / (double) ((unsigned long int) RAND_MAX + 2);
    }
    clock_t run_time;
    double elapsed_time;


    run_time = clock();
    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
        piecewise_constant_approximation(samples_in_batch, input, output);
    }
    elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
    printf("Average time for the approximate function: %g s.\n", elapsed_time / total_number_of_samples);

    run_time = clock();
    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
        for (unsigned int sample = 0; sample < samples_in_batch; sample++)
        {
            output[sample] = gsl_cdf_ugaussian_Pinv(input[sample]);
        }
    }
    elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
    printf("Average time for the exact (GSL) function: %g s.\n", elapsed_time / total_number_of_samples);

}
