// Author:
//
//      Oliver Sheridan-Methven, September 2020.
//
// Description:
//
//      An approximation of the Gaussian's inverse cumulative
//      distribution function using a piecewise constant function.

#include "piecewise_constant_approximation.h"
#include "piecewise_constant_lookup_table.h"
#include <omp.h>

void piecewise_constant_approximation(unsigned int n_samples, const double *restrict input, double *restrict output)
{
    #pragma omp simd
    for (unsigned int n = 0; n < n_samples; n++)
    {
        output[n] = lookup_table[(unsigned int) (LOOKUP_TABLE_SIZE * input[n])];
    }
}

