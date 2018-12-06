# for testing stuff
import numpy
import timeit
import time
import cProfile
import matplotlib.pyplot as mpl
from cosineLSH import *
from lsh_sketch import *


def gauss_test(n=[100], d=1, m=1, stdev=[1]):
    'n points, m gaussias, d dimensions, specified sds'
    result = numpy.random.randn(1, d)
    centers = numpy.random.normal(0*d, 10, [m, d])
    for i in range(len(centers)):
        print(n[i])
        print(d)
        c = centers[i]
        result = numpy.concatenate((result,
                                    numpy.random.normal(c, stdev[i], (n[i], d))),
                                    axis = 0)
    return result


if __name__ == '__main__':
    # gauss = numpy.random.randn(100, 99)
    # print(gauss.ndim)
    # scheme = rp.lsh(numProj=100)  # make 20 hash functions
    # print scheme.project(gauss)
    # scheme.makeFinder(5, 3)
    # print scheme.findCandidates(1)

    gauss2D = gauss_test([10,20,100,200], 2, 4, [0.1, 1, 0.01, 2])
    mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])

    subInds = lshSketch(X=gauss2D, N=100, numHashes=3000, numBands=2, bandSize=500)

    # scheme = rp.lsh(200)
    # scheme.makeFinder(1, 100, data=gauss2D)
    # subInds = scheme.downSample(sampleSize=20)
    print(subInds)
    mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')
    mpl.show()

    # counts = [10000, 20000, 30000, 40000]
    # for n in counts:
    #     start = time.time()
    #     scheme = rp.lsh(20)
    #     scheme.makeFinder(5, 3, data=gauss_test(n, 2))
    #     print(scheme.downSample())
    #     # for i in range(n):
    #     #     scheme.findCandidates(i)
    #     print('points: ' + str(n))
    #     print('time: ' + str(time.time()-start))
