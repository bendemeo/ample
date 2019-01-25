from fbpca import pca
import numpy as np
import os
import sys
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

from lsh_tester import *
from hashers import *
import pandas as pd
import math
import pickle

NAMESPACE = '293t_jurkat_lsh'
METHOD = 'svd'
DIMRED = 100





data_names = [ 'data/293t_jurkat/jurkat_293t_99_1' ]

def plot(X, title, labels, bold=None):
    plot_clusters(X, labels)
    if bold:
        plot_clusters(X[bold], labels[bold], s=20)
    plt.title(title)
    plt.savefig('{}.png'.format(title))

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)

    k = DIMRED
    U, s, Vt = pca(normalize(X), k=k)
    X_dimred = U[:, :k] * s[:k]

    labels = (
        open('data/cell_labels/jurkat_293t_99_1_clusters.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(labels)
    cell_labels = le.transform(labels)

    rare_label = le.transform(['293t'])[0]

    def gsLSH_exp(data, filename, Ns, ks=None, iter=1):
        results = None

        if ks is None:
            ks = [int(math.sqrt(numObs))]

        numObs = data.shape[0]

        params = {'target':ks}
        print('k is {}'.format(ks))

        results = try_params(X_dimred, 'gsLSH', params,
            ['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare',
            'guess','actual','error', 'gridSize', 'maxCounts'],
            cell_labels=cell_labels, rare_label = le.transform(['293t'])[0],
            Ns=Ns,
            n_seeds=3
        )
        results.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')


    def randomGrid_exp(data, filename, Ns, targets, iter=1, n_grids=3):
        ''' random grid experiment formed with OR of some random grids'''

        params = {
            'gridSize':[0.01],
            'numHashes':[n_grids],
            'numBands':[n_grids],
            'bandSize':[1],
            'target': targets
        }

        results = try_params(data, 'randomGridLSH', params,
            ['max_min_dist','time','kmeans_ami','lastCounts','maxCounts','remnants','rare',
            'guess','actual','error', 'gridSize'],
            cell_labels=cell_labels, rare_label = le.transform(['293t'])[0],
            Ns=Ns,
            n_seeds=3,
            optimizeParams=['gridSize'],
            inverted=[True]
        )
        results.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')



    # testresults_proj = try_params(X_dimred, 'projLSH',
    #     params={
    #     'numHashes':[100],
    #     'bandSize':[50],
    #     'numBands':[2],
    #     'target':[10,20,30,40,50,60,80,100,200,300,400,500],
    #     'gridSize':[0.01],
    #     },
    #     tests=['max_min_dist','time','kmeans_ami','lastCounts','maxCounts','remnants','rare',
    #         'guess','actual','error', 'gridSize'],
    #     optimizeParams=['gridSize'],
    #     inverted=[True],
    #     n_seeds=3,
    #     cell_labels=cell_labels, rare_label=le.transform(['293t'])[0]
    # )
    #
    #
    # filename='proj_gridTest'
    # iter=4
    # testresults_proj.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')



    # test_targets=[10,20,30, 40, 60, 80, 100, 200, 400, 500]
    #
    #
    # gsLSH_exp(X_dimred, '293t_gs_lsh_ktest', [100,500], [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,220,230,240,250,300], iter=4)
    #
    # randomGrid_exp(X_dimred, '293t_randomgrid_lsh_ktest', [100,500],
    #     targets=[10,20,30, 40, 50, 60, 70, 80, 90, 100, 110, 150, 200, 300, 400, 500],iter=4
    # )

    #
    #
    # filename='gsGridTest'
    # iter=4
    # gsGridTestParams = {
    #     'opt_grid':[False],
    #     'gridSize': np.arange(start=1,stop=0.01,step=-0.01).tolist()
    # }
    #
    # gsGridTests = ['max_min_dist','time','kmeans_ami','lastCounts','maxCounts','remnants','rare']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'gsLSH',
    #     params=gsGridTestParams,
    #     tests=gsGridTests,
    #     n_seeds=10,
    #     cell_labels=cell_labels,
    #     rare_label=rare_label,
    #     Ns=np.arange(start=10,stop=100,step=10).tolist()
    #     )
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    #
    #
    #
    # filename='randomGridTest'
    # iter = 1
    # randomGridParams = {
    #     'gridSize': np.arange(start=1,stop=0.01,step=-0.01).tolist(),
    #     'numHashes':[3],
    #     'numBands':[1],
    #     'bandSize':[3],
    # }
    #
    # randomGridTests = ['max_min_dist','time','kmeans_ami','lastCounts','maxCounts','remnants','rare']
    #
    # randomGridLSH_gridTest = try_params(X_dimred, 'randomGridLSH',
    #     params = randomGridParams,
    #     tests = randomGridTests,
    #     n_seeds=10,
    #     cell_labels=cell_labels,
    #     rare_label=rare_label,
    #     Ns=[100,300,500,800,1000])
    #
    # randomGridLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    # filename='cosTest'
    # iter = 1
    # cosParams = {
    #     'numHashes':np.arange(1,100,1).tolist(),
    #     'numBands':[1],
    #     'bandSize':np.arange(1,100,1).tolist()
    # }
    #
    # cosTests = ['max_min_dist','time','kmeans_ami','lastCounts','maxCounts','remnants','rare']
    #
    # cosLSH_test = try_params(X_dimred, 'cosineLSH',
    #     params = cosParams,
    #     tests = cosTests,
    #     n_seeds=10,
    #     cell_labels=cell_labels,
    #     rare_label=rare_label,
    #     Ns=[100,300,500,800,1000])
    #
    # cosLSH_test.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    # filename='projTest'
    # iter = 1
    # projParams = {
    #     'gridSize': np.arange(start=1,stop=0.01,step=-0.01).tolist(),
    #     'numHashes':[100],
    #     'numBands':[1],
    #     'bandSize':[100],
    # }
    #
    # projTests = ['max_min_dist','time','kmeans_ami','lastCounts','maxCounts','remnants','rare']
    #
    # projLSH_Test = try_params(X_dimred, 'projLSH',
    #     params = projParams,
    #     tests = projTests,
    #     n_seeds=10,
    #     cell_labels=cell_labels,
    #     rare_label=rare_label,
    #     Ns=[100,300,500,800,1000])
    #
    # projLSH_Test.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')


    # filename='randomGrid_50hash'
    # iter = 1
    # gridSizes=np.arange(start=1,stop=0.01,step=-0.01).tolist()
    #
    # Params = {
    #     'gridSize': gridSizes,
    #     'numHashes':[50],
    #     'numBands':[5],
    #     'bandSize':[5]
    # }
    #
    # Tests = ['max_min_dist','time','kmeans_ami','lastCounts','maxCounts','remnants','rare']
    #
    # Test = try_params(X_dimred, 'randomGridLSH',
    #     params = Params,
    #     tests = Tests,
    #     n_seeds=10,
    #     cell_labels=cell_labels,
    #     rare_label=rare_label,
    #     Ns=[100,300,500,800,1000])
    #
    # Test.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #

    #orig_exp(X_dimred, '293t_gs_orig')



    # ktest_cosine = try_params(X_dimred, 'cosineLSH',
    #     params={
    #         'numHashes':[100],
    #         'bandSize':[10],
    #         'numBands':[5],
    #         'target':test_targets
    #     },
    #     tests=['max_min_dist','time','kmeans_ami','lastCounts','maxCounts','remnants','rare',
    #     'guess','actual','error'],
    #     n_seeds=3,
    #     cell_labels=cell_labels, rare_label = le.transform(['293t'])[0]
    # )
    #
    # cosfile='293t_cosineLSH_ktest'
    # iter=1
    # ktest_cosine.to_csv('target/experiments/{}.txt.{}'.format(cosfile, iter), sep='\t')
    #
    # filename='gsGridTest_weighted'
    # iter=3
    # gsGridTestParams = {
    #  'opt_grid':[False],
    #  'gridSize': np.arange(start=0.5,stop=0.1,step=-0.01).tolist()
    # }
    #
    # gsGridTests = ['max_min_dist','time','kmeans_ami','rare']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'gsLSH',
    #  params=gsGridTestParams,
    #  tests=gsGridTests,
    #  n_seeds=10,
    #  cell_labels=cell_labels,
    #  rare_label=rare_label,
    #  weighted=True,
    #  Ns=np.arange(start=10,stop=100,step=10).tolist()
    #  )
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #
    # filename='gsGridTest_alpha_2'
    # iter=1
    # gsGridTestParams = {
    #  'opt_grid':[False],
    #  'gridSize': np.arange(start=0.5,stop=0.1,step=-0.01).tolist()
    # }
    #
    # gsGridTests = ['max_min_dist','time','kmeans_ami','rare']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'gsLSH',
    #  params=gsGridTestParams,
    #  tests=gsGridTests,
    #  n_seeds=10,
    #  cell_labels=cell_labels,
    #  rare_label=rare_label,
    #  Ns=np.arange(10,100,10).tolist(),
    #  weighted=True,
    #  alpha=2)
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    # filename='gsGridTest_clustcounts_wt'
    # iter=1
    # gsGridTestParams = {
    #  'opt_grid':[False],
    #  'gridSize': [0.3]
    # }
    #
    # gsGridTests = ['max_min_dist','time','cluster_counts']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'gsLSH',
    #  params=gsGridTestParams,
    #  tests=gsGridTests,
    #  n_seeds=1,
    #  cell_labels=cell_labels,
    #  cluster_labels = labels,
    #  weighted=True,
    #  Ns=[1000]
    #  )
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #
    # exit()


    # filename='293t_treeLSHTest_clustcounts'
    #
    # iter=1
    # print('filename will be {}'.format('target/experiments/{}.txt.{}'.format(filename,iter)))
    #
    #
    # splitSizes = np.repeat(np.arange(0.01, 1, 0.02), 4)
    # children = [2,3,4,5]*len(np.arange(0.01, 1, 0.02))
    #
    # TestParams = {
    #  'splitSize': splitSizes,
    #  'children': children
    # }
    #
    # gsGridTests = ['time','max_min_dist', 'occSquares','cluster_counts', 'rare']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'treeLSH',
    #  params=TestParams,
    #  tests=gsGridTests,
    #  n_seeds=3,
    #  cell_labels=cell_labels,
    #  rare_label = rare_label,
    #  cluster_labels = labels,
    #  weighted=False,
    #  Ns=[100,300, 500, 700, 1000]
    #  )
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #
    # filename='293t_treeLSHTest_clustcounts'
    #
    # iter=3
    # print('filename will be {}'.format('target/experiments/{}.txt.{}'.format(filename,iter)))
    #
    #
    # splitSizes = np.arange(0.01, 1, 0.02).tolist()
    # children = [float("inf")]*len(np.arange(0.01, 1, 0.02))
    #
    # TestParams = {
    #  'splitSize': splitSizes*3,
    #  'children': children*3,
    #  'minPoints': np.repeat([10,20,100],len(splitSizes))
    # }
    #
    # gsGridTests = ['time','max_min_dist', 'occSquares','cluster_counts', 'rare']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'treeLSH',
    #  params=TestParams,
    #  tests=gsGridTests,
    #  n_seeds=3,
    #  cell_labels=cell_labels,
    #  rare_label = rare_label,
    #  cluster_labels = labels,
    #  weighted=False,
    #  Ns=[100,300, 500, 700, 1000]
    #  )
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')


    filename='293t_angleSampler_clustcounts'

    iter=1
    print('filename will be {}'.format('target/experiments/{}.txt.{}'.format(filename,iter)))

    downsampler = angleSampler(X_dimred, strength=3)
    downsampler.makeWeights()
    downsampler.vizWeights(file='target/experiments/293t_weights')

    TestParams = {
        'strength':[1,2,3]
    }

    gsGridTests = ['time','max_min_dist','cluster_counts', 'rare']

    gsLSH_gridTest = try_params(X_dimred, 'angleSampler',
     params=TestParams,
     tests=gsGridTests,
     n_seeds=3,
     cell_labels=cell_labels,
     rare_label = rare_label,
     cluster_labels = labels,
     weighted=False,
     Ns=[100,300, 500, 700, 1000]
     )

    gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')


    # Ns=[500]
    # bandSizes=np.arange(10,20,1)
    # hashSizes=[100]*len(bandSizes)
    # bandNums=[x//y for x, y in zip(hashSizes,bandSizes)]
    #
    #
    # experiments(
    #     X_dimred, NAMESPACE,
    #     cell_labels=cell_labels,
    #     kmeans_ami=True, louvain_ami=False,
    #     rare=True,
    #     rare_label=le.transform(['293t'])[0],
    #     #entropy=True,
    #     #max_min_dist=True
    # )


    # params_gs = {'gridSize': [0.01]}
    #
    # testresults_gs = try_params(X_dimred, 'gsLSH', params_gs,
    #     ['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare',
    #     'guess','actual','error'],
    #     cell_labels=cell_labels, rare_label = le.transform(['293t'])[0],
    #     Ns=[200,300,500,800, 1000],
    #     optimizeParams=['gridSize'], inverted = [True], n_seeds = 3
    #     )
    # #
    # params_randomGrid = {
    #     'numHashes':[30]*4,
    #     'numBands': [5,6,7,8],
    #     'bandSize': [8,7,6,5],
    #     'gridSize': [0.1]
    # }
    #
    # testresults_randomGrid = try_params(X_dimred, 'randomGridLSH', params_randomGrid,
    # ['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare','guess','actual','error'],
    # cell_labels = cell_labels, rare_label = le.transform(['293t'])[0],
    # Ns=[150, 300, 400, 600],
    # optimizeParams=['gridSize'], inverted = [True], n_seeds = 5
    # )
    #
    # params_proj = {
    #     'numHashes':hashSizes,
    #     'numBands':bandNums,
    #     'bandSize': bandSizes,
    #     'gridSize':[0.1]
    # }
    #
    # testresults_proj = try_params(X_dimred, 'projLSH', params_proj,
    # ['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare',
    # 'guess','actual','error'],
    # cell_labels=cell_labels, rare_label = le.transform(['293t'])[0],
    # Ns=[100, 200, 500, 1000],
    # optimizeParams=['gridSize'], inverted = [True], n_seeds=5
    # )
    #
    # params_grid = {
    #     'gridSize': [0.1]
    # }
    #
    # testresults_grid = try_params(X_dimred, 'gridLSH',params_grid,
    #     tests=['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare',
    #     'guess','actual','error'],cell_labels=cell_labels,
    #     rare_label=le.transform(['293t'])[0],
    #     Ns=[100,200,400,800],
    #     optimizeParams=['gridSize'], inverted=[True], n_seeds=5
    # )
    #
    # params_cosine = {
    #     'numHashes':hashSizes,
    #     'numBands':bandNums,
    #     'bandSize':bandSizes
    # }
    #
    # testresults_cosine = try_params(X_dimred, 'cosineLSH', params_cosine,
    # ['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare',
    # 'guess','actual','error'],
    # cell_labels=cell_labels, rare_label = le.transform(['293t'])[0],
    # Ns=[200,300,800]
    # )




    #
    #
    # testresults = pd.concat([testresults_cosine,testresults_grid, testresults_proj, testresults_randomGrid, testresults_gs])

    # testresults = testresults_gs
    # print(testresults)
    # testresults.to_csv('target/experiments/{}.txt.5'.format(NAMESPACE), sep='\t')
    #


    exit()

    from differential_entropies import differential_entropies
    differential_entropies(X_dimred, labels)

    experiment_kmeans_ce(X_dimred, NAMESPACE, cell_labels, n_seeds=10, N=100)
    experiment_louvain_ce(X_dimred, NAMESPACE, cell_labels, n_seeds=10, N=100)

    rare(X_dimred, NAMESPACE, cell_labels, le.transform(['293t'])[0])

    balance(X_dimred, NAMESPACE, cell_labels)


        # try_lsh_params(
        #     X_dimred, 'cosineLSH', name=NAMESPACE, hashSizes=hashSizes, bandSizes=bandSizes, bandNums=bandNums, tests=['kmeans_ami','max_min_dist','rare', 'lastCounts','remnants'], cell_labels=cell_labels, rare_label=le.transform(['293t'])[0],
        #     n_seeds=5, Ns=Ns, makeVisualization = True,
        #     cell_types=labels
        # )

        # experiments_modular(
        #     X_dimred, sampling_fns=[lshSketch],
        #     name=NAMESPACE,
        #     cell_labels = cell_labels,
        #     kmeans_ami = True,
        #     louvain_ami = False,
        #     rare=True,
        #     rare_label=le.transform(['293t'])[0],
        #
        # )


    # params_randomGrid = {
    #     'numHashes':[4],
    #     'numBands':[2],
    #     'bandSize':[2],
    #     'gridSize':[0.01]
    # }
