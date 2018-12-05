from fbpca import pca
import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *

from lsh_experiments import *
from cosineLSH import *

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

    bandSizes=np.arange(10,20,1)
    hashSizes=[200]*len(bandSizes)
    bandNums=[x//y for x, y in zip(hashSizes,bandSizes)]

    Ns=[500]


    try_lsh_params(
        X_dimred, 'cosineLSH', name=NAMESPACE, hashSizes=hashSizes, bandSizes=bandSizes, bandNums=bandNums, tests=['kmeans_ami','max_min_dist','rare', 'lastCounts','remnants'], cell_labels=cell_labels, rare_label=le.transform(['293t'])[0],
        n_seeds=5, Ns=Ns, visualize = True
    )

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
    exit()

    from differential_entropies import differential_entropies
    differential_entropies(X_dimred, labels)

    experiment_kmeans_ce(X_dimred, NAMESPACE, cell_labels, n_seeds=10, N=100)
    experiment_louvain_ce(X_dimred, NAMESPACE, cell_labels, n_seeds=10, N=100)

    rare(X_dimred, NAMESPACE, cell_labels, le.transform(['293t'])[0])

    balance(X_dimred, NAMESPACE, cell_labels)
