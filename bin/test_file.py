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
from transformers import *
from vp_tree import *
from vptree import *
from fttree import *
from mcmc_sampler import *


#from lsh_experiments import start_experiment
#from lsh_sketch import *




def gauss_test(n=[100], d=1, m=1, stdev=[1]):
    #np.random.seed()
    'n points, m gaussias, d dimensions, specified sds'
    result = numpy.random.randn(0, d)
    centers = numpy.random.normal(0*d, 10, [m, d])
    for i in range(len(centers)):
        # print(n[i])
        # print(d)
        c = centers[i]
        result = numpy.concatenate((result,
                                    numpy.random.normal(c, stdev[i], (n[i], d))),
                                    axis = 0)
    return result



def gauss_embedding(n=[100], var = 1, extrinsic = 100, intrinsic = 2, normalize=True):
    np.random.seed()
    n_centers = len(n)
    centers = numpy.random.normal(0, 10, (n_centers,extrinsic))
    print(centers)

    result = np.empty([1, extrinsic])
    for i in range(centers.shape[0]):

        basis = rvs(dim=extrinsic) # random ortho basis

        subspace = basis[:,numpy.random.choice(extrinsic, size=intrinsic)]

        for j in range(n[i]):
            coefs = np.random.normal(0,var, (1,intrinsic))

            pt = np.matmul(coefs, np.transpose(subspace))

            pt = pt + centers[i,:]
            result = numpy.concatenate((result,pt), axis=0)

    if(normalize):
        result -= result.min()
        result /= result.max()
    return(result)


def random_embedding(data, shift_var=10, extrinsic = 100):
    """Given arbitrary data, embed randomly in a high dimensional space"""
    basis = rvs(dim = extrinsic)
    subspace = basis[:, numpy.random.choice(extrinsic, size=data.shape[1], replace=False)]

    # print('norms...')
    # print(np.linalg.norm(subspace[:,0]))
    # print(np.linalg.norm(subspace[:,1]))
    embedded = np.matmul(data, np.transpose(subspace))

    shift = np.random.normal(0,shift_var, extrinsic)
    # print(np.mean(embedded[:,0]))
    # print(np.mean(embedded[:,1]))
    # print(np.mean(embedded))

    for i in range(embedded.shape[0]):
        embedded[i,:] = embedded[i,:]+shift

    return(embedded)

# def gauss_embedding(n=[100], var=1, extrinsic=100, intrinsic=2, normalize=True):
#     np.random.seed()
#     n_centers = len(n)
#     centers = numpy.random.normal(0, 10, (n_centers,extrinsic))
#     print(centers)
#
#     result = np.empty([1, extrinsic])
#     for i in range(centers.shape[0]):
#



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

    np.random.seed()
    gauss = gauss_test([1000], 2, 1, [1,1])
    gauss -= gauss.min(0)
    gauss /= gauss.max()

    #print(gauss)
    def euclidean(p1, p2):
        return np.sqrt(np.sum(np.power(p2 - p1, 2)))


    # sampler = sampler(gauss)
    # sampler.sample=FTraverse(gauss,max_out=20)
    # sampler.vizSample(full=True, anno=True)

    #
    sampler = FTSampler_refined(gauss,euclidean)
    sampler.downsample(100)
    sampler.vizSample(full=True, anno=True)


    sampler = FTSampler(gauss,euclidean)
    sampler.downsample(200)
    sampler.vizSample(full=True, anno=True)
    #
    #
    # T = FTTree(gauss)
    # print(T)
    # ordering = T.traverse()
    # print(ordering)
    # print(len(ordering))
    # print(gauss.shape)
    # print(T.ind)
    # order = []
    # while(len(ordering) > 0):
    #     new = heappop(ordering)
    #     print(new)
    #     order.append(new[1])
    # print(order)
    #
    #
    #
    # sampler = sampler(gauss)
    # sampler.sample = order[:10]
    # # sampler.sample = [T.ind, T.right.ind, T.left.right.ind, T.right.right.ind]
    # # sampler.vizSample(full=True, anno=True)
    # # sampler.sample = T.left.left_inds
    # sampler.vizSample(full=True, anno=True)
    # sampler = mcmcSampler(gauss, iters=1000)
    # sampler.downsample(100)
    # sampler.vizSample(full=True)
    # #
    # sampler = fastBall(gauss, 0.05, euclidean, DIMRED=4)
    # sampler.downsample()
    # print(sampler.sample)
    # sampler.vizSample(full=True)
    #
    #
    #
    # #print(gauss)
    # t0 = time()
    # tree2=VPTree(gauss, euclidean, PCA=True)
    # t1 = time()
    # tree2.add([0]*20,1308)
    # print(tree2.get_nearest_neighbor([0.001]*20))
    # print(tree2)
    # print('BUILD TREE: other method took {} seconds'.format(t1-t0))
    # #
    # # print(tree2)
    # # tree2.add([0,0,0],11111)
    # # tree2.add([100,100,100],2222)
    # # print(tree2)
    #
    #
    # t0 = time()
    # tree1 = vpTree(gauss)
    # t1 = time()
    # print('BUILD TREE: my method took {} seconds'.format(t1-t0))
    #
    #
    # #
    # t0 = time()
    # mine = tree1.NNSearch(gauss[10,:],rad=0.1, nearest=True)
    # t1 = time()
    # print('QUERY TREE: my method took {} seconds'.format(t1-t0))
    #
    # t0 = time()
    # #theirs = tree2.get_nearest_neighbor(gauss[10,:])
    # theirs = tree2.get_nearest_neighbor(gauss[1000,:]+np.random.normal(20,0))
    # t1 = time()
    # print('QUERY TREE: their method took {} seconds'.format(t1-t0))
    #
    # print(sorted(mine))
    # print(sorted(theirs))



    # #
    # sampler = vpSampler(gauss, 0.1)
    # sampler.downsample('auto')
    # sampler.vizSample(full=True)
    #
    #
    #
    # tree = vpTree(gauss)
    # print(tree.tree.tostr())
    #
    # query = [0.5,0.5]
    #
    # print(sorted(tree.NNSearch(query, .1)))
    #
    #
    # nns = []
    # for i in range(gauss.shape[0]):
    #     norm = np.linalg.norm(gauss[i,:]-query)
    #     if norm <= .1:
    #         nns += [i]
    #
    # print(sorted(nns))
    #
    #
    #
    #


    #
    #
    # embedding = random_embedding(gauss, shift_var=10, extrinsic =100)
    #
    # tester = PCALSH(embedding, gridSize=0.296, target=1000)
    # tester.downsample(1000)
    # tester.data = gauss
    # tester.numFeatures=2
    # tester.vizSample(full=True)
    # # tester.downsample(1000)
    # #print(tester.hash)
    #
    # tester.data = gauss
    # tester.numFeatures = 2
    #
    # tester.vizHash()
    # print(tester.occSquares)
    #
    # gridTester = gridLSH(embedding, gridSize=0.1)
    # gridTester.makeHash()
    # gridTester.data = gauss
    # gridTester.numFeatures = 2
    # gridTester.vizHash()
    #
    # print(gridTester.occSquares)

    # start_table = {():range(gauss.shape[0])}
    #
    # updated_table = tester.update(start_table, gauss)
    #
    # print(gauss)
    # print(updated_table)

    # print(gauss)
    # print(flyTransform(gauss))
    #
    # visualizer = gsLSH(gauss)
    # downsampler = gridLSH(flyTransform(gauss), gridSize=0.8)
    # downsampler.makeHash
    # downsampler.downsample(4)
    #
    # visualizer.hash = downsampler.hash
    # visualizer.sample = downsampler.sample
    # visualizer.vizSample(full=True)

    # print(gauss.shape)
    # gauss -= gauss.min(0)
    # gauss /= gauss.max()
    #
    # sampler = softGridSampler(gauss,gridSize=0.25)
    # sampler.downsample(100)
    # print(sampler.sample)
    # print(len(sampler.sample))
    # sampler.vizSample(full=True)









    # gauss = numpy.random.randn(100, 99)
    # print(gauss.ndim)
    # scheme = rp.lsh(numProj=100)  # make 20 hash functions
    # print scheme.project(gauss)
    # scheme.makeFinder(5, 3)
    # print scheme.findCandidates(1)

    #start_experiment('pug', ['height','weight'],['fat','size'])

    #np.random.seed()

    # sizes=[10000]*4+[5]
    #
    # N=sum(sizes)
    # dim = 10
    # clusters=1
    # gauss2D = gauss_test(sizes, dim, clusters, [10,5,5,5,5])
    # #gauss2D -= gauss2D.min()
    # #gauss2D_2 = gauss_test([5000, 200],2,1,[10])
    # #print(gauss2D)
    #
    # gauss2D -= gauss2D.min(0)
    # gauss2D /= gauss2D.max()
    #
    # np.random.seed()
    #
    # # extrinsic = 2
    # # intrinsic = 2
    # # gauss = gauss_embedding([2000]*1, extrinsic=extrinsic, intrinsic=intrinsic,
    # #                         normalize=True, var=10)
    #
    # gauss = gauss_test([10000], 10, 1, [1])
    # print(gauss.shape)
    # gauss -= gauss.min(0)
    # gauss /= gauss.max()
    #
    # # downsampler = sampler(gauss)
    # # downsampler.downsample(1)
    # # downsampler.vizSample(full=True)
    # #
    # # seeds = gauss_test([50], dim, 1, [1000])
    #
    #
    # seedsampler = gsLSH(gauss, target=100)
    # seeds = seedsampler.downsample(20)
    # print(seeds)
    #
    # #seedsampler.vizSample(full=True)
    #
    # #downsampler = randomSoftGridSampler(gauss2D, numGrids=2, gridSize=0.7)
    # #downsampler.downsample(100)
    #
    # downsampler = fastBallSampler(gauss, seeds=seeds, gridSize = 0.03, radius = 5, ball=True)
    # #downsampler.sample = downsampler.findCandidates(1)
    # downsampler.sample = downsampler.findCandidates(100)
    #downsampler.vizSample(full=True)
    #
    # downsampler.downsample('auto')
    #
    #
    # print(downsampler.numExamined)
    #
    #
    #
    # sample = downsampler.sample
    # dists = pairwise_distances(gauss[sample,:])
    # for i in range(len(sample)):
    #     dists[i,i]=float('Inf')
    # print(dists)
    # print(np.min(dists))
    # print(len(np.unique(downsampler.sample)))
    #
    #
    # downsampler.vizSample(full=True)
    # seedsampler.vizSample(full=True, file = 'seeds', show=False)


    # print('hi')
    #
    # sampler = 'softGridSampler'
    # filename = 'pbmc_gauss_tests'
    #
    # iter = 1
    # testParams = {
    #     'gridSize':[0.1],
    #     'ball': [True],
    #     'radius':[5]
    # }
    #
    # tests = ['time','max_min_dist',
    #           'lastCounts']
    #
    #
    # testResults = try_params(gauss2D, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               Ns=['auto'],
    #                               backup=filename+'_backup')
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')





    # downsampler = dpp(gauss2D, steps=1000)



    # downsampler = slowBallSampler(gauss2D, ballSize=0.1)
    # downsampler.downsample('auto')
    # downsampler.vizSample(full=True, anno=True)

    # downsampler = softGridSampler(gauss2D, gridSize = 0.0001, ball= True, radius=8000)
    # downsampler.downsample('auto')
    #
    # sample = downsampler.sample
    # dists = pairwise_distances(gauss2D[sample,:])
    # for i in range(len(sample)):
    #     dists[i,i]=float('Inf')
    # print(dists)
    # print(np.min(dists))
    #
    # downsampler.vizSample(full=True, anno=False)









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
