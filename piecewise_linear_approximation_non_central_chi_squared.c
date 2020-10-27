//
// Created by Oliver Sheridan-Methven on 26/10/2020.
//


#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "piecewise_linear_approximation_non_central_chi_squared.h"
#include "piecewise_linear_approximation_non_central_chi_squared_coefficients.h"

// For IEEE 754
#define N_MANTISSA_32 23
#define FLOAT32_EXPONENT_BIAS 127
#define FLOAT32_EXPONENT_BIAS_TABLE_OFFSET (FLOAT32_EXPONENT_BIAS - 1)

#define TABLE_MAX_INDEX (TABLE_SIZE - 1) // Zero indexing...

#ifdef POLYNOMIAL_ORDER
#pragma omp declare simd
static inline float32 polynomial_approximation(float32 u, uint32 b, uint32 h, uint32 i) {
    /*
     * Polynomial approximation of a function.
     *
     * Input:
     *      u - Input position.
     *      b - Index of polynomial coefficient to use.
     *      h - Index of which half to use.
     *      i - Index of which interpolating function to use.
     *
     */
#if (POLYNOMIAL_ORDER == 1)
    float32 poly_coef_0 = polynomial_coefficients[h][i][0][b];
    float32 poly_coef_1 = polynomial_coefficients[h][i][1][b];
    float32 z =  poly_coef_0 + poly_coef_1 * u;
    return z;
#endif
}
#endif


#pragma omp declare simd
static inline uint32 get_table_index_from_float_format(float32 u)
{
    /*
     * Takes the approximate logarithm of a floating point number and maps this to
     * an array index, where we have the following mappings:
     *
     *      Input value/range           Output Index
     *      0.5                 ->          0
     *      [0.25,      0.5)    ->          1
     *      [0.125,     0.25)   ->          2
     *      ...
     *                                      14
     *                                      15
     *                                      15  <<  Table is capped at 16 entries
     *                                      15  <<
     *                                      15  <<
     *                                      ...
     *                                      15  <<
     *
     * Assumes input has a zero in its sign bit.
     *
     */

    union
    {
        uint32 as_integer;
        float32 as_float;
    } u_pun;
    uint32 b;

    u_pun.as_float = u;
    b = u_pun.as_integer >> N_MANTISSA_32; // Keeping only the exponent, and removing the mantissa.
    b = FLOAT32_EXPONENT_BIAS_TABLE_OFFSET - b; // Getting the table index.
    b = b > TABLE_MAX_INDEX ? TABLE_MAX_INDEX : b;  // Ensuring we don't overflow out of the table.
    return b;
}

#pragma omp declare simd
static inline void interpolation_indices(const float32 y, uint32 * restrict interpolation_index_lower, uint32 * restrict interpolation_index_upper, float32 * restrict weight_lower, float32 * restrict weight_upper)
{
    float32 x = sqrtf(y) * (INTERPOLATION_FUNCTIONS - 1);
    *interpolation_index_lower = (uint32) x;
    *interpolation_index_upper = *interpolation_index_lower + 1;
    *weight_upper = x - ((uint32) x);
    *weight_lower = 1.0f - *weight_upper;
}


void piecewise_polynomial_approximation_non_central_chi_squared(unsigned int n_samples, const float32 *restrict input, const float32 *restrict non_centrality, float32 *restrict output)
{
#pragma omp simd
    for (unsigned int i = 0; i < n_samples; i++)
    {
        float32 u, p, z, lambda;
        u = input[i];
        lambda = non_centrality[i];
        uint32 upper_half = u > 0.5f;
        u = upper_half ? 1.0f - u : u;
        float32 weight_lower, weight_upper;
        uint32 interpolation_index_lower, interpolation_index_upper;
        float32 y = DOF / (lambda + DOF); // interpolation_value
        interpolation_indices(y, &interpolation_index_lower, &interpolation_index_upper, &weight_lower, &weight_upper);
        uint32 b = get_table_index_from_float_format(u);
        float32 p_lower = polynomial_approximation(u, b, upper_half, interpolation_index_lower);
        float32 p_upper = polynomial_approximation(u, b, upper_half, interpolation_index_upper);
        p = weight_lower * p_lower + weight_upper * p_upper;
        z = lambda + DOF + 2.0f * sqrtf(lambda + DOF) * p;
        output[i] = z;
    }
}
