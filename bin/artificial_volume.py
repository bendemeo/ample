from fbpca import pca
import math
import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

NAMESPACE = 'artificial_volume'
METHOD = 'svd'
DIMRED = 100

data_names = [ 'data/293t_jurkat/293t' ]

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)

    k = DIMRED
    U, s, Vt = pca(normalize(X), k=k)
    X_dimred = U[:, :k] * s[:k]

    Xs = []
    labels = []
    translate = X_dimred.max(0)
    for i in range(3):
        Xs.append((X_dimred / (10. ** i)) + (translate * 2 * i))
        labels += list(np.zeros(X_dimred.shape[0]) + i)

    X_dimred = np.concatenate(Xs)
    cell_labels = np.array(labels, dtype=int)

    expected = np.array([ 1, 1/10., 1/100.])
    expected = np.array(expected) / sum(expected)

    #see how grid size affects this

    filename='gsGridTest_equalvol'
    iter=1
    gsGridTestParams = {
        'opt_grid':[False],
        'gridSize': np.arange(start=0.99,stop=0.01,step=-0.01).tolist()
    }

    gsGridTests = ['max_min_dist','time','kmeans_ami','lastCounts','maxCounts','remnants','rare', 'gridSize']

    gsLSH_gridTest = try_params(X_dimred, 'gsLSH',
        params=gsGridTestParams,
        tests=gsGridTests,
        n_seeds=10,
        cell_labels=cell_labels,
        rare_label=2
        Ns=[100,300,500,800,1000])


    gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')




    # experiments(
    #     X_dimred, NAMESPACE,
    #     rare=True, cell_labels=cell_labels, rare_label=2,
    #     entropy=True,
    #     kl_divergence=True, expected=expected,
    #     max_min_dist=True
    # )
