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
from test_file import gauss_test, random_embedding


def test_gridSizes(data, sampler = 'PCALSH', filename = sampler, iter=1, seeds=1, sizes = np.arange(1, 0.01, -0.01).tolist()):
    sampler = sampler
    filename = '{}_{}_gridTest'.format(filename,sampler)

    testParams = {
        'gridSize':sizes
    }
    tests = ['time','max_min_dist','occSquares']

    testResults = try_params(data, sampler,
                                  params=testParams,
                                  tests=tests,
                                  n_seeds=seeds,
                                  Ns=['auto'],
                                  backup=filename+'_backup')

    testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')



def multi_gauss(N=1000, centers=1, intrinsic = 2, extrinsic=100, var = 1, shift_var=1):
    """randomly embedded Gaussians"""
    np.random.seed()
    result = np.empty((1, extrinsic))

    for i in range(centers):
        gauss = gauss_test([N], d=intrinsic, m=1, stdev=[1])
        gauss -= gauss.min(0)
        gauss /= gauss.max()
        #print(gauss)

        embedding = random_embedding(gauss, shift_var=shift_var, extrinsic=extrinsic)
        result = numpy.concatenate((result, embedding), axis=0)

    return(result)
    #test_gridSizes(result, sampler, 'pcaLSH', seeds=10, iter=iter)

if __name__ == '__main__':
    multi = multi_gauss(N=1000, centers=10)
    print(multi.shape)
    print(multi)
    # test_gridSizes(multi, sampler = 'PCALSH', seeds=1, filename='PCALSH_multigauss', sizes=np.arange(0.2, 0, -0.002))

    test_gridSizes(multi, sampler = 'gridLSH', sizes=np.arange(0.2, 0, -0.002), seeds=1, filename='gridLSH_multigauss')
    #experiment_1(sampler = 'gridLSH', seeds=5)


# def experiment_1(data, sampler='PCALSH', iter=2, seeds=1):
#     #first experiment: 2D Gauss in 100-D space, randomly embedded
#
#     np.random.seed(10)
#     gauss = gauss_test([10000], 2, 1, [1,1,1,1,1])
#     gauss -= gauss.min(0)
#     gauss /= gauss.max()
#
#     embedding = random_embedding(gauss, shift_var=10, extrinsic =100)
#
#     sampler = sampler
#     filename = 'randomGauss_{}_test'.format(sampler)
#
#     testParams = {
#         'gridSize':np.arange(1, 0.01, -0.01).tolist()
#     }
#     tests = ['time','max_min_dist','occSquares']
#
#     testResults = try_params(embedding, sampler,
#                                   params=testParams,
#                                   tests=tests,
#                                   n_seeds=seeds,
#                                   Ns=['auto'],
#                                   backup=filename+'_backup')
#
#     testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
#
