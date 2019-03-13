# for testing stuff
import numpy as np
import pandas as pd
import timeit
import math
from time import time
import cProfile
import matplotlib.pyplot as mpl
from hashers import *
from experiments import *
from copy import deepcopy
from sklearn.metrics.pairwise import pairwise_distances


#from lsh_experiments import start_experiment
#from lsh_sketch import *




def gauss_test(n=[100], d=1, m=1, stdev=[1]):
    np.random.seed()
    'n points, m gaussias, d dimensions, specified sds'
    result = numpy.random.randn(0, d)
    centers = numpy.random.normal(0*d, 10, [m, d])
    for i in range(len(centers)):
        print(n[i])
        print(d)
        c = centers[i]
        result = numpy.concatenate((result,
                                    numpy.random.normal(c, stdev[i], (n[i], d))),
                                    axis = 0)
    return result

def prettydata(n=1000, n_polys=1, deg=3, sds=[1]):
    #random polynomial coefs
    x=np.arange(n)
    x = [float(y)/n for y in x]

    y = [0]*len(x)

    result = None
    for j in range(n_polys):
        np.random.seed()
        coefs = np.random.normal(size=deg)
        coefs=[c*10 for c in coefs]
        print(coefs)

        powers = list(range(deg))

        for i, xi in enumerate(x):
            powerterms = [xi**p for p in powers]
            yi = [a*b for a,b in zip(coefs, powerterms)]
            yi = sum(yi)
            yi = yi + np.random.normal(scale=sds[j])
            y[i] = yi

        current = np.column_stack((x,y))

        theta = np.random.uniform(0, 2*math.pi)
        rotmatrix = [[math.cos(theta), -math.sin(theta)],[math.sin(theta),math.cos(theta)]]


        print('rotation: {}'.format(rotmatrix))
        for i in range(current.shape[0]):
            #print('before: {}'.format(current[i,:]))
            current[i,:] = np.matmul(current[i,:], rotmatrix)
            #print('after: {}'.format(current[i,:]))

        if result is None:
            result = current
        else:
            result = np.concatenate((result, current), axis=0)



    result[:,0] = result[:,0]-result[:,0].min(0)
    result[:,1] = result[:,1]-result[:,1].min(0)

    result[:,0] /= result[:,0].max()
    result[:,1] /= result[:,1].max()
    return result



def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)




if __name__ == '__main__':
    # gauss = numpy.random.randn(100, 99)
    # print(gauss.ndim)
    # scheme = rp.lsh(numProj=100)  # make 20 hash functions
    # print scheme.project(gauss)
    # scheme.makeFinder(5, 3)
    # print scheme.findCandidates(1)

    #start_experiment('pug', ['height','weight'],['fat','size'])

    #np.random.seed()

    sizes=[1000]*4+[5]

    N=sum(sizes)
    gauss2D = gauss_test(sizes, 20, 5, [2,0.1,0.1,0.1,0.1])
    #gauss2D -= gauss2D.min()
    #gauss2D_2 = gauss_test([5000, 200],2,1,[10])
    #print(gauss2D)

    gauss2D -= gauss2D.min(0)
    gauss2D /= gauss2D.max()

    np.random.seed()


    # downsampler = dpp(gauss2D, steps=1000)



    # downsampler = slowBallSampler(gauss2D, ballSize=0.1)
    # downsampler.downsample('auto')
    # downsampler.vizSample(full=True, anno=True)

    downsampler = softGridSampler(gauss2D, gridSize = 0.1, ball= True)
    downsampler.downsample('auto')
    downsampler.vizSample(full=True, anno=True)

    sample = downsampler.sample
    dists = pairwise_distances(gauss2D[sample,:])
    for i in range(len(sample)):
        dists[i,i]=float('Inf')
    print(dists)
    print(np.min(dists))

    # downsampler = softGridSampler(gauss2D, gridSize=2)
    # downsampler.downsample('auto')
    #
    # sample = downsampler.sample
    # dists = pairwise_distances(gauss2D[sample,:])
    # for i in range(len(sample)):
    #     dists[i,i]=float('Inf')
    # print(dists)
    # print(np.min(dists))
    #
    #
    # downsampler.vizSample(full=True, anno=True)



    # downsampler = multiscaleSampler(gauss2D, scales=[0.07])
    # downsampler.makeWeights()
    # # downsampler.downsample(100)
    # # downsampler.vizSample()
    # i=1
    # downsampler.vizWeights(file='/Users/bdemeo/Desktop/cell_cover/weights_wistia{}'.format(i),dpi=1000, s=5, cmap='Wistia')



    # sampler = 'dpp'
    # filename = 'dpp_test'
    # picklename = None
    #
    # iter = 1
    # testParams = {
    #     'steps': [10000],
    # }
    #
    # tests = ['time','max_min_dist']
    #
    #
    # testResults = try_params(gauss2D, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               Ns=[1000],
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    #
    # sampler = 'gsLSH'
    # filename = 'gsLSH_test'
    # picklename = None
    #
    # iter = 1
    # testParams = {
    #     'target': ['N'],
    #     'opt_grid':[True]
    # }
    #
    # tests = ['time','max_min_dist']
    #
    #
    # testResults = try_params(gauss2D, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               Ns=[1000],
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #
    #
    # sampler = 'diverseLSH'
    # filename = 'diverseLSH_test'
    # picklename = None
    #
    # iter = 1
    # testParams = {
    #     'steps': [10000],
    #     'numCenters':[10,20, 30]
    # }
    #
    # tests = ['time','max_min_dist']
    #
    #
    # testResults = try_params(gauss2D, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               Ns=[1000],
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')


    #
    # poly = prettydata(n=1000, deg=5, n_polys=5, sds=[50, 50, 50, 50, 50])
    #
    # print('poly:')
    # print(poly)
    # downsampler = gsLSH(poly, gridSize=0.2, opt_grid=False)
    #
    #
    # downsampler.makeHash()
    # print(downsampler.weights)
    # downsampler.vizData(c=[math.log(x) for x in downsampler.weights])



    # downsampler = densitySampler(gauss2D)
    # downsampler.makeWeights()
    # downsampler.vizWeights()
    #
    # downsampler = centerSampler(gauss2D, numCenters=20, weighted=True)
    # downsampler.downsample(200)
    # downsampler.vizSample(full=True)



    # downsampler = sigSampler(gauss2D, bins=10)
    # downsampler.sigTransform()
    # print(downsampler.data)
    # downsampler.downsample(200)
    # downsampler.vizSample(full=True, anno=True, annoMax=200)


    # gauss2D -= gauss2D.min(0)
    # downsampler = dpp(gauss2D, steps=10000, normalize=True)
    # downsampler = detSampler(gauss2D, batch=1000)
    # downsampler.downsample(50)
    # downsampler.vizSample(full=True)


    # downsampler = centerSampler(gauss2D, numCenters=6, steps=10000, spherical=True)
    # downsampler.downsample(100)
    # print(downsampler.sample)
    # downsampler.vizSample(full=True)
    # downsampler.vizSample()
    # print(downsampler.sampleEmbedding.shape)
    # print(downsampler.embedding.shape)


    # downsampler = detSampler(gauss2D, batch=1000)
    # downsampler.downsample(20)
    # print('determinant using greedy method')
    # print(downsampler.det)



    # downsampler = diverseLSH(gauss2D, batch=100, numCenters=5, labels = [1,2]*430+[3])
    # downsampler.qTransform(q=10)
    #
    #
    #
    #
    # downsampler.downsample(400)
    # print(max(downsampler.sample))
    # downsampler.vizHash()
    #downsampler.vizSample(full=False, c=np.array(downsampler.labels)[downsampler.sample])



    # downsampler = ballLSH(gauss2D, epsilon=6, ord=float('inf'))
    # downsampler.makeHash()
    # print(downsampler.occSquares)
    # downsampler.vizHash()
    # downsampler.downsample(100)
    # downsampler.vizSample()
    #print(downsampler.hash)
    # downsampler.makeFinder()
    # downsampler.downsample(100)
    # print(downsampler.finder)
    # print(downsampler.bands)
    # #downsampler.vizSample()
    # downsampler.vizHash()


    # downsampler = rankLSH(gauss2D, numCenters=6)
    # downsampler.makeHash()
    # downsampler.makeFinder()
    # downsampler.downsample(100)
    # print(downsampler.finder)
    # print(downsampler.bands)
    # #downsampler.vizSample()
    # downsampler.vizHash()

    # downsampler = diverseLSH(gauss2D, numCenters=5)
    # downsampler.makeHash()
    # downsampler.makeFinder()
    # downsampler.downsample(100)
    # print(downsampler.finder)
    # print(downsampler.bands)
    # downsampler.vizSample()
    # downsampler.vizHash()


    # downsampler = treeLSH(gauss2D_2, splitSize=0.1, children=4


    #downsampler = gridLSH(gauss2D_2, gridSize=0.1)

    # downsampler = svdSampler(gauss2D, batch=100)
    # #downsampler.normalize(method='l1')
    #
    # # downsampler.makeHash()
    # # print(downsampler.hash)
    # size=5
    # t0 = time()
    # downsampler.downsample(size)
    # t1=time()
    # print('it took {} seconds'.format(t1-t0))
    # downsampler.vizSample(c=range(size), cmap='hot', anno=True, full=True)

    # downsampler = pRankSampler(gauss2D)
    #
    # downsampler.rank()
    # print(downsampler.ranking)
    # downsampler.vizRanking()
    #
    # downsampler.downsample(100)
    # downsampler.vizSample()
    # downsampler = angleSampler(gauss2D, strength=3)
    #
    # downsampler.makeWeights()
    #
    # downsampler.vizWeights(file='plots/weights', log=False)
    #
    # sample = downsampler.downsample(100)
    # downsampler.vizSample(file='plots/sample')

    # subInds = downsampler.downsample(100)
    # print(subInds)
    #
    # mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])
    # mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')




    # downsampler.makeHash()
    # print('this is the hash')
    # print(downsampler.hash)
    #
    #
    # experiment(downsampler, gauss2D_2, 'aprilpug.png', cell_labels='grid', kmeans=False,
    # downsample = False, lsh=True, visualize_orig = False)

    # print(downsampler.data)
    # print(quantilate(downsampler.data[:,0], 0.1, 3))




    # downsampler.makeHash()
    # print(downsampler.hash)
    #
    # subInds = downsampler.downSample(10)
    #
    # print(downsampler.lastCounts)
    # #print(downsampler.gridLabels)
    # mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])
    # mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')


    mpl.show()




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

    #mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])


    # downsampler = gsLSH(gauss2D, target = int(math.sqrt(N)))
    # downsampler_2 = deepcopy(downsampler)
    #
    # downsampler = times(downsampler, downsampler_2)
    # downsampler_2  =randomGridLSH(gauss2D, 0.1, 1,1,1)
    # #
    # # downsampler = plus(downsampler_1, downsampler_2)
    #
    # downsampler = times(downsampler_1, downsampler_2)
    # downsampler_2 = downsampler
    # for n in range(2):
    #     downsampler_2=times(downsampler_2, downsampler)
    #
    # print(downsampler_2.components)

    # alpha=1
    #
    # experiment(downsampler, gauss2D, 'testpng2', filename = 'testpng2', cell_labels='grid',
    #     gene_names=[], genes=[], gene_expr=gauss2D,
    #     kmeans=False,
    #     visualize_orig=False,
    #     sample_type='gsLSH_wt',
    #     lsh=True, optimize_grid_size=False,
    #     weighted = True, alpha = alpha)
    #

    # downsampler = gridLSH(gauss2D, 0.1, randomize_origin = True)


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

    # mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])
    # mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')
    # mpl.show()

    # downsampler = gridLSH(gauss2D, gridSize=0.1)
    # downsampler.makeHash()
    # #print(downsampler.hash)
    #
    # subInds = downsampler.downSample(50)
    # print(downsampler.lastCounts)


    #subInds = lshSketch(X=gauss2D, N=100, numHashes=3000, numBands=2, bandSize=500)



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


# def quantilate(vals, splitSize, max_splits = 2):
#     """converts arrays of values to which quantile division they belong to"""
#     diam = max(vals) - min(vals)
#
#     if diam < splitSize:
#         return([0]*len(vals))
#
#     splits = min(np.ceil(diam / float(splitSize)), max_splits)
#
#     return pd.qcut(vals, splits, labels=False)
#     # split_quants = np.arange(0, 101, 100./splits)
#     #
#     # quantiles = np.percentile(vals, split_quants)
#     # print(quantiles)
#     #
#     # result = [None]*len(vals)
#     #
#     # for i in range(len(quantiles))[:-1]:
#     #     print(i)
#     #     print('quantiles is {}'.format(quantiles))
#     #     inds = [j for j in range(len(vals)) if vals[j] >= quantiles[i] and
#     #         vals[j] <= quantiles[i+1]]
#     #
#     #     print(inds)
#     #     for ind in inds:
#     #         result[ind] = i
#     #
#     # unclassified = [vals[i] for i in range(len(vals)) if result[i] is None]
#     #
#     # print('unclassified values: {}'.format(unclassified))
#     # return result
