"""
Author:

    Oliver Sheridan-Methven, September 2020.

Description:

    Some demonstrations of approximating the
    Gaussian distribution's inverse cumulative
    distribution function to produce approximate random
    variables by the inverse transform method.
"""
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.pylab import plot, legend, savefig, xscale, yscale, ylabel, xlabel
from matplotlib.pylab import clf as clear_plot
from numpy import nan, linspace
from scipy.stats import norm as gaussian
from scipy.stats import uniform
from timeit import default_timer as timer

uniform_numbers = lambda n_samples: uniform.rvs(size=n_samples)
inverse_gaussian_cdf = gaussian.ppf
import approximate_gaussian_distribution as approximations


def plot_piecewise_constant_approximation_of_gaussian(save_figure=False):
    """ Plot a piecewise constant approximation to the Gaussian. """
    uniform_input = linspace(0, 1, 1000)[1:-1]  # We exclude the end points.
    approximate_inverse_gaussian_cdf = approximations.construct_piecewise_constant_approximation(inverse_gaussian_cdf, n_intervals=8)
    clear_plot()
    plot(uniform_input, inverse_gaussian_cdf(uniform_input), 'k--')
    plot(uniform_input, approximate_inverse_gaussian_cdf(uniform_input), 'k,')
    if save_figure:
        savefig('piecewise_constant_approximation_of_gaussian.pdf', format='pdf', bbox_inches='tight', transparent=True)


def piecewise_constant_approximation_of_gaussian_timing():
    """ We assess the speed of a piecewise constant approximation to the Gaussian. """
    n_intervals = 1024
    # We don't want to hold to many numbers in memory at once, so we break them into batches.
    samples_in_batch = 10000
    number_of_batches = 1000
    total_number_of_samples = samples_in_batch * number_of_batches
    input_values = uniform_numbers(samples_in_batch)
    functions = {'exact': inverse_gaussian_cdf, 'approximate': approximations.construct_piecewise_constant_approximation(inverse_gaussian_cdf, n_intervals)}
    for func_name, func in functions.iteritems():
        start_time = timer()
        for batch in range(number_of_batches):
            func(input_values)
        elapsed_time = timer() - start_time
        print("Average time for the {} function: {:g} s.".format(func_name, elapsed_time / total_number_of_samples))


def plot_piecewise_polynomial_approximation_of_gaussian(save_figure=False):
    """ Plot a piecewise polynomial approximation to the Gaussian. """
    uniform_input = linspace(0, 1, 1000)[1:-1]  # We exclude the end points.
    approximate_inverse_gaussian_cdf = approximations.construct_symmetric_piecewise_polynomial_approximation(inverse_gaussian_cdf, n_intervals=4, polynomial_order=1)
    clear_plot()
    plot(uniform_input, inverse_gaussian_cdf(uniform_input), 'k--')
    plot(uniform_input, approximate_inverse_gaussian_cdf(uniform_input), 'k,')
    if save_figure:
        savefig('piecewise_polynomial_approximation_of_gaussian.pdf', format='pdf', bbox_inches='tight', transparent=True)


def piecewise_constant_polynomial_of_gaussian_timing():
    """ We assess the speed of a piecewise polynomial approximation to the Gaussian. """
    n_intervals = 16
    polynomial_order = 1
    # We don't want to hold to many numbers in memory at once, so we break them into batches.
    samples_in_batch = 10000
    number_of_batches = 100
    total_number_of_samples = samples_in_batch * number_of_batches
    input_values = uniform_numbers(samples_in_batch)
    functions = {'exact': inverse_gaussian_cdf, 'approximate': approximations.construct_symmetric_piecewise_polynomial_approximation(inverse_gaussian_cdf, n_intervals, polynomial_order)}
    for func_name, func in functions.iteritems():
        start_time = timer()
        for batch in range(number_of_batches):
            func(input_values)
        elapsed_time = timer() - start_time
        print("Average time for the {} function: {:g} s.".format(func_name, elapsed_time / total_number_of_samples))

def plot_error_of_piecewise_polynomial_approximation_of_gaussian(save_figure=False):
    """ Plots the RMSE of various polynomial approximations. """
    polynomial_orders = range(6)
    interval_sizes = [2, 4, 8, 16]
    clear_plot()
    for n_intervals in interval_sizes:
        rmse = [None] * len(polynomial_orders)
        for polynomial_order in polynomial_orders:
            approximate_inverse_gaussian_cdf = approximations.construct_symmetric_piecewise_polynomial_approximation(inverse_gaussian_cdf, n_intervals, polynomial_order)
            discontinuities = [0.5 ** (i+2) for i in range(n_intervals - 1)]  # Makes the numerical integration involved in the RMSE easier.
            rmse[polynomial_order] = approximations.expected_value_in_interval(lambda u: (inverse_gaussian_cdf(u) - approximate_inverse_gaussian_cdf(u)) ** 2, 0, 0.5, points=discontinuities) ** 0.5
        plot(polynomial_orders, rmse, 'o--', label=n_intervals)
    legend(title="Intervals")
    yscale('log')
    ylabel('RMSE')
    xlabel('Polynomial order')
    if save_figure:
        savefig('piecewise_polynomial_approximation_of_gaussian_rmse.pdf', format='pdf', bbox_inches='tight', transparent=True)

coeffs = approximations.piecewise_polynomial_coefficients_in_half_interval(inverse_gaussian_cdf, 16, 3)

for j in range(coeffs.shape[0]):
    c = coeffs[j]
    print("const float32 poly_coef_{}[16] = {{{}}};".format(j, ', '.join(['{}'.format(i) for i in c])))

