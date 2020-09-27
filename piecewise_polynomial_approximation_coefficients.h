// Author:
//
//      Oliver Sheridan-Methven, September 2020.
//
// Description:
//
//      Coefficients for a piecewise polynomial approximation to
//      the Gaussian's inverse cumulative distribution function.

#ifndef APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_POLYNOMIAL_APPROXIMATION_COEFFICIENTS_H

#define TABLE_SIZE 16

#if (TABLE_SIZE == 16)
#ifdef POLYNOMIAL_ORDER
#if (POLYNOMIAL_ORDER == 3)
const float32 poly_coef_0[16] = {0.0, -1.679197345, -1.9645626943, -2.23486034388, -2.48492707919, -2.71791046795, -2.93649277839, -3.14280712882, -3.3385534554, -3.52509919412, -3.70355669882, -3.87484129665, -4.03971461576, -4.19881704755, -4.35269227065, -4.85359308578};
const float32 poly_coef_1[16] = {0.0, 5.17091582873, 8.60962819975, 15.0915055833, 27.0888159965, 49.4532234987, 91.4321502911, 170.701989576, 321.160008222, 607.996338237, 1156.90159287, 2210.76316498, 4239.88975409, 8156.59986514, 15733.5204903, 101577.673487};
const float32 poly_coef_2[16] = {0.0, -5.58087455879, -19.6939989371, -72.6657796486, -268.818407116, -1000.40299455, -3747.76238832, -14126.5912329, -53535.4373973, -203827.20389, -779145.668087, -2988644.73987, -11498260.8031, -44353567.0632, -171485300.841, -4882624931.81};
const float32 poly_coef_3[16] = {0.0, 3.91370441691, 23.6043529771, 170.870534719, 1261.83392041, 9402.5615004, 70564.0941404, 532792.279196, 4043788.19301, 30828458.0405, 235926861.25, 1811508864.41, 13949376118.3, 1.07687553657e+11, 8.33188425042e+11, 8.23928040543e+13};
#elif (POLYNOMIAL_ORDER == 1)
const float32 poly_coef_0[16] = {0.0, -1.32705468316, -1.60211363454, -1.89517898841, -2.17029169248, -2.42545425048, -2.66295208398, -2.8853675019, -3.09492212866, -3.29342637073, -3.48234293521, -3.66285965711, -3.83595036528, -4.00242198071, -4.16295020439, -4.56405881592};
const float32 poly_coef_1[16] = {0.0, 2.67304493943, 3.76922290369, 6.07216678027, 10.3896821759, 18.3986090944, 33.3115138442, 61.2513770816, 113.914016285, 213.708456508, 403.695425144, 766.837343251, 1463.3480102, 2803.27422972, 5387.74589832, 21632.6362366};
#else
// Some more coefficients are required for other polynomials. 
#endif // (POLYNOMIAL_ORDER == ...)
#endif // POLYNOMIAL_ORDER
#endif // TABLE_SIZE

#define APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_POLYNOMIAL_APPROXIMATION_COEFFICIENTS_H
#endif // APPROXIMATE_RANDOM_VARIABLES_PIECEWISE_POLYNOMIAL_APPROXIMATION_COEFFICIENTS_H
