/*******************************************************************************
 * This file is part of the "high performance low precision vectorised         *
 * arithmetic" project, created by Oliver Sheridan-Methven 2017.               *
 *                                                                             *
 * Copyright (C) 2017 Oliver Sheridan-Methven, University of Oxford.           *
 *                                                                             *
 * Commercial users wanting to use this software should contact the author:    *
 * oliver.sheridan-methven@maths.ox.ac.uk                                      *
 *                                                                             *
 ******************************************************************************/
#ifndef APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_POLYNOMIAL_APPROXIMATION_H

// Assuming IEEE 754 that integers and floats are 32 bits.
typedef unsigned int uint32;
typedef float float32;

void piecewise_polynomial_approximation(unsigned int n_samples, const float32 *restrict input, float32 *restrict output);

#define APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_POLYNOMIAL_APPROXIMATION_H

#endif //APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_POLYNOMIAL_APPROXIMATION_H
