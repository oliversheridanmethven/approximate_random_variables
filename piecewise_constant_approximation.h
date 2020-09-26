// Author:
//
//      Oliver Sheridan-Methven, September 2020.
//
// Description:
//
//      An approximation of the Gaussian's inverse cumulative
//      distribution function using a piecewise constant function.


#ifndef APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_CONSTANT_APPROXIMATION_H

void piecewise_constant_approximation(unsigned int n_samples, const double *restrict input, double *restrict output);
// n_samples: Number of samples.
// input: Input values.
// output: Output values.

#define APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_CONSTANT_APPROXIMATION_H

#endif //APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_CONSTANT_APPROXIMATION_H
