// Author:
//
//      Oliver Sheridan-Methven, September 2020.
//
// Description:
//
//      An approximation of the Gaussian's inverse cumulative
//      distribution function using a piecewise constant function.

#include "piecewise_constant_approximation.h"
#include "piecewise_constant_approximation_coefficients.h"
#include <omp.h>

#ifndef USE_VECTOR_BATCHES
void piecewise_constant_approximation(unsigned int n_samples, const double *restrict input, double *restrict output)
{
    #pragma omp simd
    for (unsigned int n = 0; n < n_samples; n++)
    {
        output[n] = lookup_table[(unsigned int) (LOOKUP_TABLE_SIZE * input[n])];
    }
}
#else  // Assuming we want to do this using larger AVX512 batches.
#define VECTOR_LENGTH 8  // 8 x 64 bit doubles fit in a AVX512 vector register.
void piecewise_constant_approximation(unsigned int n_samples, const double *restrict input, double *restrict output)
{
    unsigned int n_batches = n_samples / VECTOR_LENGTH;
    for (unsigned int batch = 0; batch < n_batches; batch++, input += VECTOR_LENGTH, output += VECTOR_LENGTH)
    {
        #pragma omp simd simdlen(VECTOR_LENGTH)
        for (unsigned int n = 0; n < VECTOR_LENGTH; n++)
        {
            output[n] = lookup_table[(unsigned int) (LOOKUP_TABLE_SIZE * input[n])];
        }
    }
}
#endif
