"""
Author:

    Oliver Sheridan-Methven September 2020.

Description:

    Comparison of the non-central chi^2 to the Gaussian.
"""

import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.pylab import plot, legend, savefig, xscale, yscale, ylabel, xlabel
from matplotlib.pylab import clf as clear_plot
from numpy import linspace
import pandas as pd
from scipy.stats import uniform, ncx2
from timeit import default_timer as timer
from approximate_non_central_chi_squared import construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation

uniform_numbers = lambda n_samples: uniform.rvs(size=n_samples)
ncx2_exact = ncx2.ppf


def plot_non_central_chi_squared_polynomial_approximation(save_figure=False):
    """ Plots a polynomial approximation to the non-central chi-squared. """
    u = linspace(0.0, 1.0, 10000)[:-1]  # Excluding the end points.
    dof = 1.0
    non_centralities = [1.0, 10.0, 20.0]
    clear_plot()
    for non_centrality in non_centralities:
        ncx2_approx = construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(dof, n_intervals=4)
        plot(u, ncx2.ppf(u, df=dof, nc=non_centrality), 'k--')
        plot(u, ncx2_approx(u, non_centrality=non_centrality), 'k,')
    savefig('piecewise_polynomial_approximation_of_non_central_chi_squared.pdf', format='pdf', bbox_inches='tight', transparent=True)


def non_central_chi_squared_polynomial_approximation_timing():
    """ We assess the speed of a piecewise polynomial approximation to the non-central chi-squared distribution. """
    nus = [1, 5, 10, 50]
    lambdas = [1, 5, 10, 50]

    n = 10000
    res = {}
    for nu in nus:
        res[nu] = {}
        ncx2_approx = construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(nu)
        for l in lambdas:
            u = uniform.rvs(size=n)
            start = timer()
            ncx2.ppf(u, df=nu, nc=l)
            elapsed_ncx2 = (timer() - start) / n
            start = timer()
            ncx2_approx(u, non_centrality=l)
            elapsed_norm = (timer() - start) / n
            res[nu][l] = round((elapsed_ncx2 / elapsed_norm), 1)

    df = pd.DataFrame(res)
    df.index = df.index.rename('lambda')
    df.columns = df.columns.rename('nu')
    print df