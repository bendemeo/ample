import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize, LabelEncoder
from LSH import *
from hashers import *

from experiments import *
from process import load_names
from utils import *
from mouse_brain import *

np.random.seed(0)

NAMESPACE = 'mouse_brain'
METHOD = 'svd'
DIMRED = 100


def keep_valid(datasets):
    n_valid = 0
    for i in range(len(datasets)):
        valid_idx = []
        with open('{}/meta.txt'.format(data_names[i])) as f:
            n_lines = 0
            for j, line in enumerate(f):
                fields = line.rstrip().split()
                if fields[1] != 'NA':
                    valid_idx.append(j)
                n_lines += 1
        assert(n_lines == datasets[i].shape[0])
        datasets[i] = datasets[i][valid_idx, :]
        print('{} has {} valid cells'
              .format(data_names[i], len(valid_idx)))
        n_valid += len(valid_idx)
    print('Found {} valid cells among all datasets'.format(n_valid))


data_names = [
    'data/mouse_brain/dropviz/Cerebellum_ALT',
    'data/mouse_brain/dropviz/Cortex_noRep5_FRONTALonly',
    'data/mouse_brain/dropviz/Cortex_noRep5_POSTERIORonly',
    'data/mouse_brain/dropviz/EntoPeduncular',
    'data/mouse_brain/dropviz/GlobusPallidus',
    'data/mouse_brain/dropviz/Hippocampus',
    'data/mouse_brain/dropviz/Striatum',
    'data/mouse_brain/dropviz/SubstantiaNigra',
    'data/mouse_brain/dropviz/Thalamus',
]




if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    keep_valid(datasets)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)

    if not os.path.isfile('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('data/dimred/{}_{}.txt'.format(METHOD, NAMESPACE))

    viz_genes = []

    labels = (
        open('data/cell_labels/mouse_brain_cluster.txt')
        .read().rstrip().split('\n')
    )



    le = LabelEncoder().fit(labels)
    cell_names = sorted(set(labels))
    cell_labels = le.transform(labels)

    print('there are {} labels'.format(len(cell_labels)))


    experiment(gs_gap, X_dimred, NAMESPACE, filename='orig_fn_2',         cell_labels=cell_labels,
                gene_names=viz_genes, genes=genes, gene_expr=vstack(datasets),
                kmeans=False,
                visualize_orig=False,
                sample_type='gsLSH_wt',
                lsh=False
    )



    #
    # iter = 1
    #
    #
    # downsampler = gsLSH(X_dimred)
    #
    # for alpha in np.arange(2,10,0.5):
    #     filename='gsLSHTest_weighted_alpha_{}'.format(alpha)
    #     experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels=cell_labels,
    #         gene_names=viz_genes, genes=genes, gene_expr=vstack(datasets),
    #         kmeans=False,
    #         visualize_orig=False,
    #         sample_type='gsLSH_wt',
    #         lsh=True, optimize_grid_size=True,
    #         weighted = True, alpha = alpha)
