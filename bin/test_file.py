# for testing stuff
import numpy
import timeit
import math
from time import time
import cProfile
import matplotlib.pyplot as mpl
from hashers import *
from experiments import *
from copy import deepcopy
#from lsh_experiments import start_experiment
#from lsh_sketch import *




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

    #start_experiment('pug', ['height','weight'],['fat','size'])

    sizes=[100,200,500,1000]
    N=sum(sizes)
    gauss2D = gauss_test(sizes, 2, 4, [1, 0.5, 0.4, 0.2])
    #mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])


    downsampler = gsLSH(gauss2D, target = int(math.sqrt(N)))
    # downsampler_2 = deepcopy(downsampler)
    #
    # downsampler = times(downsampler, downsampler_2)
    # downsampler_2  = randomGridLSH(gauss2D, 0.1, 1,1,1)
    # #
    # # downsampler = plus(downsampler_1, downsampler_2)
    #
    # downsampler = times(downsampler_1, downsampler_2)
    # downsampler_2 = downsampler
    # for n in range(2):
    #     downsampler_2=times(downsampler_2, downsampler)
    #
    # print(downsampler_2.components)


    subInds = downsampler.downSample(50)
    print(downsampler.gridLabels)





    #
    #
    #
    # downsampler = projLSH(gauss2D, 1000, 10, 10, 0.1)
    # downsampler.optimize_param('gridSize', 100, inverted = True)
    #
    #
    # downsampler = randomGridLSH(gauss2D, numHashes = 10, numBands = 2, bandSize = 2, gridSize = 0.01)

    #downsampler = cosineLSH(gauss2D, numHashes = 1000, numBands = 1, bandSize=500)
    # t0 = time()
    # subInds = downsampler.fastDownsample(500)
    # t1 = time()
    # print('fast downsampling took {} seconds'.format(t1-t0))


    # randomGrid_exp(gauss2D, '293t_randomgrid_lsh_ktest', [100,500],
    #     targets=[10,20,30, 40, 60, 80, 100, 200, 400, 800],iter=1
    # )
    #
    #
    # N=50



    # rg = randomGridLSH(gauss2D, numHashes = 5, numBands = 3, bandSize = 1, gridSize = 0.01)
    # rg.optimize_param('gridSize', target=4, inverted = True)


    #experiment(rg, gauss2D, 'rglshtest2', lsh=True)
    #
    # print('grid size is {}'.format(rg.gridSize))
    # #rg.optimize_param('bandSize', N, inverted = False)
    #
    #
    # # t0 = time()
    # # subInds = rg.downSample(5)
    # # t1 = time()
    # print('random grid took {} seconds to downsample'.format(t1-t0))
    # print(rg.hash)




    # proj = projLSH(gauss2D, numHashes = 10, numBands = 2, bandSize = 2, gridSize = 0.01)
    #
    # t0 = time()
    # subInds = proj.downSample(1000)
    # t1 = time()
    # print('random projection took {} seconds to downsample'.format(t1-t0))
    # print(proj.hash)

    # t0 = time()
    # subInds = downsampler.downSample(50)
    # t1 = time()
    # print('slow downsampling took {} seconds'.format(t1-t0))
    # print(downsampler.hash)

    mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])
    mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')
    mpl.show()

    # downsampler = gridLSH(gauss2D, gridSize=0.1)
    # downsampler.makeHash()
    # #print(downsampler.hash)
    #
    # subInds = downsampler.downSample(50)
    # print(downsampler.lastCounts)


    #subInds = lshSketch(X=gauss2D, N=100, numHashes=3000, numBands=2, bandSize=500)

    # mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')
    # mpl.show()#

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
