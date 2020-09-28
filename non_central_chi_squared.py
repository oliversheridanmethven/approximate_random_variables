"""
Author:

    Oliver Sheridan-Methven September 2020.

Description:

    Comparison of the non-central chi-squared to the Gaussian.
"""

import pandas as pd
from scipy.stats import uniform, ncx2, norm
from timeit import default_timer as timer

if __name__ == '__main__':
    nus = [1, 5, 10, 50]
    lambdas = [1, 5, 10, 50, 100, 200]

    n = 10000
    res = {}
    for nu in nus:
        res[nu] = {}
        for l in lambdas:
            u = uniform.rvs(size=n)
            start = timer()
            ncx2.ppf(u, df=nu, nc=l)
            elapsed_ncx2 = (timer() - start) / n
            start = timer()
            norm.ppf(u)
            elapsed_norm = (timer() - start) / n
            res[nu][l] = int(elapsed_ncx2 / elapsed_norm)

    df = pd.DataFrame(res)
    df.index = df.index.rename('lambda')
    df.columns = df.columns.rename('nu')
    print df
