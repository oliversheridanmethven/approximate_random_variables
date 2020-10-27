//
// Created by Oliver Sheridan-Methven on 26/10/2020.
//

#ifndef APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_LINEAR_APPROXIMATION_NON_CENTRAL_CHI_SQUARED_H

// Assuming IEEE 754 that integers and floats are 32 bits.
typedef unsigned int uint32;
typedef float float32;

void piecewise_polynomial_approximation_non_central_chi_squared(unsigned int n_samples, const float32 *__restrict__ input, const float32 *__restrict__ non_centrality, float32 *__restrict__ output);

#define APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_LINEAR_APPROXIMATION_NON_CENTRAL_CHI_SQUARED_H

#endif //APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_LINEAR_APPROXIMATION_NON_CENTRAL_CHI_SQUARED_H
