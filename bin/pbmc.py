from fbpca import pca
import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *
from LSH import *
from hashers import *


NAMESPACE = 'pbmc_facs'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/pbmc/10x/b_cells',
    'data/pbmc/10x/cd14_monocytes',
    'data/pbmc/10x/cd4_t_helper',
    'data/pbmc/10x/cd56_nk',
    'data/pbmc/10x/cytotoxic_t',
    'data/pbmc/10x/memory_t',
    'data/pbmc/10x/regulatory_t',
]

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

    labels = []
    names = []
    curr_label = 0
    for i, a in enumerate(datasets):
        labels += list(np.zeros(a.shape[0]) + curr_label)
        names.append(data_names[i])
        curr_label += 1
    labels = np.array(labels, dtype=int)

    k = DIMRED
    U, s, Vt = pca(normalize(X), k=k)
    X_dimred = U[:, :k] * s[:k]

    labels = (
        open('data/cell_labels/pbmc_facs_cluster.txt')
        .read().rstrip().split()
    )
    le = LabelEncoder().fit(labels)
    cell_labels = le.transform(labels)

    viz_genes = []

    # experiment(gs_gap, X_dimred, NAMESPACE, filename='orig_fn', cell_labels=cell_labels,
    #             gene_names=viz_genes, genes=genes, gene_expr=vstack(datasets),
    #             kmeans=False,
    #             visualize_orig=False,
    #             sample_type='gsLSH_wt',
    #             lsh=False,
    #             weighted = True, alpha = alpha
    # )


    downsampler = gsLSH(X_dimred)
    for alpha in np.arange(1,10,0.5):
        filename='gsLSHTest_weighted_alpha_{}'.format(alpha)
        experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels=cell_labels,
            gene_names=viz_genes, genes=genes, gene_expr=vstack(datasets),
            kmeans=False,
            visualize_orig=False,
            sample_type='gsLSH_wt',
            lsh=True, optimize_grid_size=True,
            weighted = True, alpha = alpha)

    # experiment_gs(
    #     X_dimred, NAMESPACE, cell_labels=cell_labels,
    #     kmeans=False, visualize_orig=False
    # )
    # experiment_uni(
    #     X_dimred, NAMESPACE, cell_labels=cell_labels,
    #     kmeans=False, visualize_orig=False
    # )
    # experiment_srs(
    #     X_dimred, NAMESPACE, cell_labels=cell_labels,
    #     kmeans=False, visualize_orig=False
    # )
    # experiment_kmeanspp(
    #     X_dimred, NAMESPACE, cell_labels=cell_labels,
    #     kmeans=False, visualize_orig=False
    # )
    # exit()
    #
    # experiments(
    #     X_dimred, NAMESPACE,
    #     cell_labels=cell_labels,
    #     kmeans_ami=True, louvain_ami=True,
    #     #rare=True,
    #     #rare_label=le.transform(['cd14_monocytes'])[0],
    #     #entropy=True,
    #     #max_min_dist=True
    # )
