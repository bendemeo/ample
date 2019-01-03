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


    Ns=[500]
    bandSizes=np.arange(10,20,1)
    hashSizes=[100]*len(bandSizes)
    bandNums=[x//y for x, y in zip(hashSizes,bandSizes)]


    params_randomGrid = {
        'numHashes':[30]*4,
        'numBands': [5,6,7,8],
        'bandSize': [8,7,6,5],
        'gridSize': [0.1]
    }

    testresults_randomGrid = try_params(X_dimred, 'randomGridLSH', params_randomGrid,
    ['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare','guess','actual','error'],
    cell_labels = cell_labels, rare_label = le.transform(['293t'])[0],
    Ns=[150, 300, 400, 600],
    optimizeParams=['gridSize'], inverted = [True], n_seeds = 5
    )

    params_proj = {
        'numHashes':hashSizes,
        'numBands':bandNums,
        'bandSize': bandSizes,
        'gridSize':[0.1]
    }

    testresults_proj = try_params(X_dimred, 'projLSH', params_proj,
    ['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare',
    'guess','actual','error'],
    cell_labels=cell_labels, rare_label = le.transform(['293t'])[0],
    Ns=[100, 200, 500, 1000],
    optimizeParams=['gridSize'], inverted = [True], n_seeds=5
    )

    params_grid = {
        'gridSize': [0.1]
    }

    testresults_grid = try_params(X_dimred, 'gridLSH',params_grid,
        tests=['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare',
        'guess','actual','error'],cell_labels=cell_labels,
        rare_label=le.transform(['293t'])[0],
        Ns=[100,200,400,800],
        optimizeParams=['gridSize'], inverted=[True], n_seeds=5
    )

    params_cosine = {
        'numHashes':hashSizes,
        'numBands':bandNums,
        'bandSize':bandSizes
    }

    testresults_cosine = try_params(X_dimred, 'cosineLSH', params_cosine,
    ['max_min_dist','time','kmeans_ami','lastCounts','remnants','rare',
    'guess','actual','error'],
    cell_labels=cell_labels, rare_label = le.transform(['293t'])[0],
    Ns=[200,300,800]
    )







    testresults = pd.concat([testresults_cosine,testresults_grid, testresults_proj, testresults_randomGrid])
    print(testresults)
    testresults.to_csv('target/experiments/{}.txt.4'.format(NAMESPACE), sep='\t')



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

        # experiments(
        #     X_dimred, NAMESPACE,
        #     cell_labels=cell_labels,
        #     kmeans_ami=True, louvain_ami=False,
        #     rare=True,
        #     rare_label=le.transform(['293t'])[0],
        #     #entropy=True,
        #     #max_min_dist=True
        # )

    # params_randomGrid = {
    #     'numHashes':[4],
    #     'numBands':[2],
    #     'bandSize':[2],
    #     'gridSize':[0.01]
    # }
