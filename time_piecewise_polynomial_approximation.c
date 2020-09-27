// Author:
//
//      Oliver Sheridan-Methven, September 2020.
//
// Description:
//
//      Timing the performance for approximations to the
//      Gaussian's inverse cumulative distribution function.

#ifdef COMPARE_AGAINST_MKL
#include <mkl.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_cdf.h>
#include "piecewise_polynomial_approximation.h"

int main(int argc, char **argv)
{
    /* We don't want to hold to many numbers in memory at once, so we break them into batches. */
    unsigned int samples_in_batch = 512 * 100;
    unsigned int n_batches = 1000;
    unsigned int total_number_of_samples = samples_in_batch * n_batches;
    float32 input[samples_in_batch];
    float32 output[samples_in_batch];
    for (unsigned int i = 0; i < samples_in_batch; i++)
    {   /* Random numbers in the range (0, 1), which are non-inclusive. */
        input[i] = (float32) ((unsigned long int) rand() + 1) / (float32) ((unsigned long int) RAND_MAX + 2);
    }
    clock_t run_time;
    double elapsed_time;


    run_time = clock();
    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
        piecewise_polynomial_approximation(samples_in_batch, input, output);
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

    #ifdef COMPARE_AGAINST_MKL
    run_time = clock();  // Intel high accuracy (HA)
    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
        vsCdfNormInv(samples_in_batch, input, output);
    }
    elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
    printf("Average time for the exact (Intel HA) function: %g s.\n", elapsed_time / total_number_of_samples);


    const MKL_UINT64 mode_lower_accuracy = (VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    run_time = clock();  // Intel low accuracy (LA)
    for (unsigned int batch = 0; batch < n_batches; batch++)
    {
        vmsCdfNormInv(samples_in_batch, input, output, mode_lower_accuracy);
    }
    elapsed_time = difftime(clock(), run_time) / CLOCKS_PER_SEC;
    printf("Average time for the exact (Intel LA) function: %g s.\n", elapsed_time / total_number_of_samples);

    #endif

}
