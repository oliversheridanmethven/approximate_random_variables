// Author:
//
//      Oliver Sheridan-Methven, September 2020.
//
// Description:
//
//      A piecewise polynomial approximation to
//      the Gaussian's inverse cumulative distribution function.
// 
// Possible flags:
// 
//      The following flags should be mutually exclusive:
// 
//          USE_OPENMP_SIMD_APPROX 
//          USE_INTEL_INTRINSICS_APPROX_CUBIC 
//          USE_INTEL_INTRINSICS_APPROX_LINEAR
//          USE_ARM_SVE_INLINE_APPROX_CUBIC
//
// Assumptions:
//
//      The code is implemented assuming vectors of exactly 512 bits,
//      and that the floats are in 32 bit single precision, and that
//      the number of samples being processed is an exact multiple
//      of the vector length, and hence there is no loop tail.


// The Intel approximations require the MKL and assume a certain vector length. 
#if (defined(USE_INTEL_INTRINSICS_APPROX_CUBIC) || defined(USE_INTEL_INTRINSICS_APPROX_LINEAR))
#define VECTOR_LENGTH 16  // We can fit 16 x 32 bit floats in an AVX-512 vector register.
#include <mkl.h>
#include <immintrin.h>
#endif


#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "piecewise_polynomial_approximation.h"
#include "piecewise_polynomial_approximation_coefficients.h"

// For IEEE 754
#define N_MANTISSA_32 23
#define FLOAT32_EXPONENT_BIAS 127
#define FLOAT32_EXPONENT_BIAS_TABLE_OFFSET (FLOAT32_EXPONENT_BIAS - 1)

#define TABLE_MAX_INDEX (TABLE_SIZE - 1) // Zero indexing...
#define FLOAT32_AS_UINT32(x) (*((uint32 *) &x))  // Type-punning is performed, so use at your own risk.

#pragma omp declare simd

static inline float32 polynomial_approximation(float32 u, uint32 b);

#pragma omp declare simd

static inline uint32 get_table_index_from_float_format(float32 u);


#ifdef POLYNOMIAL_ORDER

static inline float32 polynomial_approximation(float32 u, uint32 b) {
    /*
     * Polynomial approximation of a function.
     *
     * This assumes a very small polynomial and will exploit Horner's rule
     * and splits the polynomial into the even and odd terms. This could
     * (and should) be generalised to bigger polynomials, but more ideally
     * tailored to constants, piecewise linear, and quadratic. Beyond cubic
     * and this might be redundant and then a very generalised implementation
     * might be preferable...
     *
     * Input:
     *      u - Input position.
     *      b - Index of polynomial coefficient to use.
     *
     */

    // This would ideally be implemented using vector registers and scatter/gather intrinsics,
    // but for now we let the compiler just do its best.
#if (POLYNOMIAL_ORDER == 3)
    float32 z, z_even, z_odd;
    float32 x = u * u;
    z_even = poly_coef_0[b] + poly_coef_2[b] * x;
    z_odd = poly_coef_1[b] + poly_coef_3[b] * x;
    z = z_even + z_odd * u;
    return z;
#elif (POLYNOMIAL_ORDER == 1)
    float32 z =  poly_coef_0[b] + poly_coef_1[b] * u;
    return z;
#else
#endif
}

#endif

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

    uint32 b;
    b = FLOAT32_AS_UINT32(u) >> N_MANTISSA_32; // Keeping only the exponent, and removing the mantissa.
    b = FLOAT32_EXPONENT_BIAS_TABLE_OFFSET - b; // Getting the table index.
    b = b > TABLE_MAX_INDEX ? TABLE_MAX_INDEX : b;  // Ensuring we don't overflow out of the table.
    return b;
}

#ifdef USE_OPENMP_SIMD_APPROX
void piecewise_polynomial_approximation(unsigned int n_samples, const float32 *restrict input, float32 *restrict output)
{
#pragma omp simd 
    for (unsigned int i = 0; i < n_samples; i++)
    {
        float32 u, z;
        u = input[i];
        bool predicate = u < 0.5f;
        u = predicate ? u : 1.0f - u;
        uint32 b = get_table_index_from_float_format(u);
        z = polynomial_approximation(u, b);
        z = predicate ? z : -z;
        output[i] = z;
    }
}
#elif defined(USE_ARM_SVE_INLINE_APPROX_CUBIC)
void piecewise_polynomial_approximation(unsigned int n_samples, const float32 *restrict input, float32 *restrict output)
{

    // This is not VLA, but assumes vector lengths sufficient to hold the arrays of coefficients in vector registers.
    // Based on the example which can be found at
    //      https://developer.arm.com/tools-and-software/server-and-hpc/arm-architecture-tools/documentation/writing-inline-sve-assembly
    unsigned int i;  // A dummy variable!
    asm (""
        // Pre Loop check if 0 entries to be iterated over.
        "\tcbz %[n_samples], 2f                                     "
        // Check if n_samples > 0, else skip to function return.

        // Setting up some coefficients and runtime constants
        "\n\tmov %[i], xzr                                          "
        // Move the XZR (Zero register, aka 0) to X8 (loop counter).
        "\n\tfmov    z0.s, #5.000000000000000000e-01                "
        // Constant.
        "\n\tptrue   p0.s                                           "
        // All true predicate register
        "\n\tfmov    z1.s, #1.000000000000000000e+00                "
        // Constant.

        // Insert the coefficients into vector registers.
        "\n\tld1w    {z2.s}, p0/z, [%[poly_coef_0], %[i], lsl #2]   "
        // Load the coefficients from memory
        "\n\tld1w    {z3.s}, p0/z, [%[poly_coef_1], %[i], lsl #2]   "
        // ""                  ""
        "\n\tld1w    {z4.s}, p0/z, [%[poly_coef_2], %[i], lsl #2]   "
        // Load remaining coefficients.
        "\n\tld1w    {z5.s}, p0/z, [%[poly_coef_3], %[i], lsl #2]   "
        // ""          ""          ""

        // A few remaining constants.
        "\n\tmov z6.s, #126                                         "
        // Constant. (exponent bias)
        "\n\tmov z7.s, %[table_max_index]                           "
        // Constant. (max table index)

        // The main while loop.
        "\n\twhilelo p1.s, xzr, %[n_samples]                        "
        // While the output location valid
        "\n1:                                                       "
        "\n\tld1w    {z8.s}, p1/z, [%[input], %[i], lsl #2]         "
        // Load x1 (the data) into a vector register.
        "\n\tfsub    z9.s, z1.s, z8.s                               "
        // Compute '1.0 - x'
        "\n\tfcmgt   p2.s, p0/z, z0.s, z8.s                         "
        // Evaluate 'x < 0.5'
        "\n\tsel z8.s, p2, z8.s, z9.s                               "
        // Select bytes from '1.0 - x'.
        "\n\tlsr z9.s, z8.s, #23                                    "
        // Shift right 23 bits removing the mantissa.
        "\n\tsub z9.s, z6.s, z9.s                                   "
        // Correct exponent bias.
        "\n\tcmplo   p3.s, p0/z, z9.s, %[table_max_index]           "
        // Compare with 15 using LO (unsigned lower)
        "\n\tsel z9.s, p3, z9.s, z7.s                               "
        // Select either 15 or index value for the index.
        "\n\tfmul    z10.s, z8.s, z8.s                              "
        // Compute 'x*x' for use with Horner's rule.

        // NB - Table lookups don't require a Byte offset.
        "\n\ttbl z11.s, {z2.s}, z9.s                                "
        // Table lookup from register for coefficient.
        "\n\ttbl z12.s, {z4.s}, z9.s                                "
        // Table lookup from register for coefficient.
        "\n\tfmad    z12.s, p0/m, z10.s, z11.s                      "
        // FMA for Horner's rule.
        "\n\ttbl z11.s, {z3.s}, z9.s                                "
        // Table lookup from register for coefficient.
        "\n\ttbl z13.s, {z5.s}, z9.s                                "
        // Table lookup from register for coefficient.
        "\n\tfmad    z13.s, p0/m, z10.s, z11.s                      "
        // FMA for Horner's rule.
        "\n\tfmad    z13.s, p0/m, z8.s, z12.s                       "
        // FMA combining even and odd polynomials.
        "\n\tfneg    z10.s, p0/m, z13.s                             "
        // Negation (predicated) to get sign of function correct.
        "\n\tsel z10.s, p2, z13.s, z10.s                            "
        // Compute the correct sign.
        "\n\tst1w    {z10.s}, p1, [%[output], %[i], lsl #2]         "
        // Store/write the answer.
        "\n\tincw    %[i]                                           "
        // Increment loop by number of words in vector.

        // Increment the while loop
        "\n\twhilelo p1.s, %[i], %[n_samples]                       "
        // While i < n_samples.
        "\n\tb.any 1b                                               "
        // If the first predicate is true branch to the while loop
        // (there are still items to complete).

        // End of the while loop.
        "\n2:                                                       "
        "\n\t                                                       "
        // Return. No need to explicit 'ret' as this ends the program
        // (and not the function!), so would cause seg faults. 
        : 
        [i] "=&r" (i)
        : 
        "[i]" (0),
        [input] "r" (input),
        [output] "r" (output),
        [n_samples] "r" (n_samples),
        [poly_coef_0] "r" (poly_coef_0),
        [poly_coef_1] "r" (poly_coef_1),
        [poly_coef_2] "r" (poly_coef_2),
        [poly_coef_3] "r" (poly_coef_3),
        [table_max_index] "i" (TABLE_MAX_INDEX)
        :
        "memory",
        "cc",
        "p0", "p1", "p2", "p3",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13"
        );
}
#elif defined(USE_INTEL_INTRINSICS_APPROX_CUBIC)
void piecewise_polynomial_approximation(unsigned int n_samples, const float32 *restrict input, float32 *restrict output)
{
    unsigned int n_iterations = n_samples / VECTOR_LENGTH;  // For now we assume a perfect vector amount is used.

    /* Loading in the constants. */
    __m512 half_float = _mm512_set1_ps(0.5f);
    __m512 one_float = _mm512_set1_ps(1.0f);
    __m512 minus_zero_float = _mm512_set1_ps(-0.0f);  // For negation.
    __m512i exponent_bias_table_offset = _mm512_set1_epi32(FLOAT32_EXPONENT_BIAS_TABLE_OFFSET);
    __m512i table_max_index = _mm512_set1_epi32(TABLE_MAX_INDEX);
    __m512 coef_0 = _mm512_load_ps(poly_coef_0);  // These should only be loaded in once.
    __m512 coef_1 = _mm512_load_ps(poly_coef_1);
    __m512 coef_2 = _mm512_load_ps(poly_coef_2);
    __m512 coef_3 = _mm512_load_ps(poly_coef_3);
    __m512 c_0, c_1, c_2, c_3;
    __m512i c;  // A useful temporary variable.

    __m512 u, z, z_even, z_odd, u_squared;
    __m512i b;
    for (unsigned int i = 0, offset = 0; i < n_iterations; i++, offset += VECTOR_LENGTH)
    {

        // Loading in the data.
        u = _mm512_loadu_ps(input + offset);

        // Forming the predicate.
        __mmask16 predicate = _mm512_cmplt_ps_mask(half_float, u);

        // Transforming the input to [0, 0.5].
        u = _mm512_mask_sub_ps(u, predicate, one_float, u);
        b = _mm512_castps_si512(u);  // NB - These intrinsics are used for compilation and do not generate any instructions.

        // Getting the indices.
        b = _mm512_srli_epi32(b, N_MANTISSA_32);
        b = _mm512_sub_epi32(exponent_bias_table_offset, b);
        b = _mm512_min_epi32(b, table_max_index);

        // Obtaining the relevant coefficients.
        /* Shuffle operations will require aliasing as 32-bit integers. */
        c = _mm512_castps_si512(coef_0);
        c = _mm512_permutexvar_epi32(b, c);
        c_0 = _mm512_castsi512_ps(c);

        c = _mm512_castps_si512(coef_1);
        c = _mm512_permutexvar_epi32(b, c);
        c_1 = _mm512_castsi512_ps(c);

        c = _mm512_castps_si512(coef_2);
        c = _mm512_permutexvar_epi32(b, c);
        c_2 = _mm512_castsi512_ps(c);

        c = _mm512_castps_si512(coef_3);
        c = _mm512_permutexvar_epi32(b, c);
        c_3 = _mm512_castsi512_ps(c);

        // Building the polynomial.
        u_squared = _mm512_mul_ps(u, u);
        z_even = _mm512_fmadd_ps(c_2, u_squared, c_0);
        z_odd = _mm512_fmadd_ps(c_3, u_squared, c_1);
        z = _mm512_fmadd_ps(z_odd, u, z_even);
        z = _mm512_mask_xor_ps(z, predicate, minus_zero_float, z); // Negation.

        // Writing the result.
        _mm512_storeu_ps(output + offset, z);

    }
}
#elif defined(USE_INTEL_INTRINSICS_APPROX_LINEAR)
void piecewise_polynomial_approximation(unsigned int n_samples, const float32 *restrict input, float32 *restrict output)
{
    unsigned int n_iterations = n_samples / VECTOR_LENGTH;  // For now we assume a perfect vector amount is used.

    /* Loading in the constants. */
    __m512 half_float = _mm512_set1_ps(0.5f);
    __m512 one_float = _mm512_set1_ps(1.0f);
    __m512 minus_zero_float = _mm512_set1_ps(-0.0f);  // For negation.
    __m512i exponent_bias_table_offset = _mm512_set1_epi32(FLOAT32_EXPONENT_BIAS_TABLE_OFFSET);
    __m512i table_max_index = _mm512_set1_epi32(TABLE_MAX_INDEX);
    __m512 coef_0 = _mm512_load_ps(poly_coef_0);  // These should only be loaded in once.
    __m512 coef_1 = _mm512_load_ps(poly_coef_1);
    __m512 c_0, c_1;
    __m512i c;  // A useful temporary variable.

    __m512 u, z;
    __m512i b;
    for (unsigned int i = 0, offset = 0; i < n_iterations; i++, offset += VECTOR_LENGTH)
    {

        // Loading in the data.
        u = _mm512_loadu_ps(input + offset);

        // Forming the predicate.
        __mmask16 predicate = _mm512_cmplt_ps_mask(half_float, u);

        // Transforming the input to [0, 0.5].
        u = _mm512_mask_sub_ps(u, predicate, one_float, u);
        b = _mm512_castps_si512(u);  // NB - These intrinsics are used for compilation and do not generate any instructions.

        // Getting the indices.
        b = _mm512_srli_epi32(b, N_MANTISSA_32);
        b = _mm512_sub_epi32(exponent_bias_table_offset, b);
        b = _mm512_min_epi32(b, table_max_index);

        // Obtaining the relevant coefficients.
        /* Shuffle operations will require aliasing as 32-bit integers. */
        c = _mm512_castps_si512(coef_0);
        c = _mm512_permutexvar_epi32(b, c);
        c_0 = _mm512_castsi512_ps(c);

        c = _mm512_castps_si512(coef_1);
        c = _mm512_permutexvar_epi32(b, c);
        c_1 = _mm512_castsi512_ps(c);
        // Building the polynomial.
        z = _mm512_fmadd_ps(c_1, u, c_0);
        z = _mm512_mask_xor_ps(z, predicate, minus_zero_float, z); // Negation.

        // Writing the result.
        _mm512_storeu_ps(output + offset, z);

    }
}
#endif

