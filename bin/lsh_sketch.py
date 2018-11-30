import numpy as np
import sys


import random_proj as rp

def lshSketch(X, N, numProj=100, numBands=10, bandSize=10, seed=None, replace=False):
    n_samples, n_features = X.shape

    if not seed is None:
        np.random.seed(seed)
    if not replace and N > n_samples:
        raise ValueError('Cannot sample {} elements from {} elements '
                         'without replacement'.format(N, n_samples))
    if not replace and N == n_samples:
        return range(N)

    if numProj == None:
        raise ValueError('Please provide number of random projections')

    sketcher = rp.lsh(numProj=numProj, data=X, numBands=numBands, bandSize=bandSize)

    return sorted(sketcher.downSample(sampleSize=N, replace=replace))
