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

np.random.seed(0)

NAMESPACE = 'mouse_brain'
METHOD = 'svd'
DIMRED = 100

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

if __name__ == '__main__':
    datasets, genes_list, n_cells = load_names(data_names, norm=False)
    keep_valid(datasets)
    datasets, genes = merge_datasets(datasets, genes_list)
    X = vstack(datasets)

    if not os.path.isfile('dimred/{}_{}.txt'.format(METHOD, NAMESPACE)):
        print('performing PCA...')
        log('Dimension reduction with {}...'.format(METHOD))
        X_dimred = reduce_dimensionality(
            normalize(X), method=METHOD, dimred=DIMRED
        )
        log('Dimensionality = {}'.format(X_dimred.shape[1]))
        np.savetxt('dimred/{}_{}.txt'.format(METHOD, NAMESPACE), X_dimred)
    else:
        X_dimred = np.loadtxt('dimred/{}_{}.txt'.format(METHOD, NAMESPACE))

    viz_genes = [
        'Gja1', 'Flt1', 'Gabra6', 'Syt1', 'Gabrb2', 'Gabra1',
        'Meg3', 'Mbp', 'Rgs5', 'Pcp2', 'Dcn', 'Pvalb', 'Nnat',
        'C1qb', 'Acta2', 'Syt6', 'Lhx1', 'Sox4', 'Tshz2', 'Cplx3',
        'Shisa8', 'Fibcd1', 'Drd1', 'Otof', 'Chat', 'Th', 'Rora',
        'Synpr', 'Cacng4', 'Ttr', 'Gpr37', 'C1ql3', 'Fezf2',
    ]

    labels = (
        open('data/cell_labels/mouse_brain_cluster.txt')
        .read().rstrip().split('\n')
    )

    print(len(labels))

    le = LabelEncoder().fit(labels)
    cell_names = sorted(set(labels))
    cell_labels = le.transform(labels)

    print(X_dimred.shape)


    sampler = bSampler(X_dimred, 0.6, backup_interval=500)
    sampler.downsample(filename = 'mouse_brain_perfect')
    sampler.downsample(filaname = 'mouse_brain_viz')

    #
    # iter = 2
    # sampler = 'gsLSH'
    # filename = 'mouse_brain_gsLSH_hausdorff_{}'.format(iter)
    #
    # testParams = {
    #     'gridSize':np.arange(0.4, 0.01,-0.05).tolist(),
    #     'opt_grid':[False]
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts', 'occSquares']
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               cell_labels=cell_labels,
    #                               Ns=['auto'],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = None)
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #

    #
    # sampler = 'PCALSH'
    # filename = 'mouse_brain_PCALSH_hausdorff'
    # iter = 1
    # testParams = {
    #     'gridSize':np.arange(1, 0.01,-0.05).tolist()
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts', 'occSquares']
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               cell_labels=cell_labels,
    #                               Ns=['auto'],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = None)
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')



    #
    # downsampler = PCALSH(X_dimred, alpha=0.01, gridSize = 0.2)
    #
    #
    # experiment(downsampler, X_dimred, NAMESPACE, cell_labels=cell_labels,
    # gene_names=viz_genes, genes=genes,
    # gene_expr=vstack(datasets),
    # kmeans=False,
    # visualize_orig=False,
    # sample_type='PCALSH',
    # lsh=True, optimize_grid_size=False,
    # filename='mouse_brain_PCALSH')
    #
    # sampler = 'centerSampler'
    # filename = 'mouse_brain_centerSamplerTest'
    # iter = 1
    # testParams = {
    #     'numCenters':np.arange(1, 100, 1).tolist(),
    #     'steps': [1000]
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=3,
    #                               cell_labels=cell_labels,
    #                               Ns=[1000],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup')
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv(
    #     'target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')



    #
    #
    # experiment(downsampler, X_dimred, NAMESPACE, cell_labels=cell_labels,
    # gene_names=viz_genes, genes=genes,
    # gene_expr=vstack(datasets),
    # kmeans=False,
    # visualize_orig=False,
    # sample_type='softGridLSH',
    # lsh=True, optimize_grid_size=False)

    # downsampler = randomGridLSH(X_dimred, 0.01, 7, 2,3)
    # downsampler = cosineLSH(X_dimred,
    #     numHashes=500,
    #     numBands=20,
    #     bandSize=30
    # )

    # experiment(downsampler, X_dimred, NAMESPACE, cell_labels=cell_labels,
    # gene_names=viz_genes, genes=genes,
    # gene_expr=vstack(datasets),
    # kmeans=False,
    # visualize_orig=False,
    # sample_type='randomGridLSH_2_',
    # lsh=True, optimize_grid_size=True)


    # experiment_lsh(
    #     X_dimred, NAMESPACE, cell_labels=cell_labels,
    #     gene_names=viz_genes, genes=genes,
    #     gene_expr=vstack(datasets),
    #     kmeans=False,
    #     visualize_orig=False
    # )

    # experiments(
    #     X_dimred, NAMESPACE, n_seeds=2,
    #     cell_labels=cell_labels,
    #     kmeans_ami=True, louvain_ami=True,
    #     rare=True,
    #     rare_label=le.transform(['Endothelial_Tip'])[0],
    # )
    # exit()
    # from differential_entropies import differential_entropies
    # differential_entropies(X_dimred, labels)
    # experiment_gs(
    #     X_dimred, NAMESPACE, cell_labels=cell_labels,
    #     gene_names=viz_genes, genes=genes,
    #     gene_expr=vstack(datasets),
    #     kmeans=False, visualize_orig=False
    # )
    # experiment_uni(
    #     X_dimred, NAMESPACE, cell_labels=cell_labels,
    #     gene_names=viz_genes, genes=genes,
    #     gene_expr=vstack(datasets),
    #     kmeans=False, visualize_orig=False
    # )
    # experiment_srs(
    #     X_dimred, NAMESPACE, cell_labels=cell_labels,
    #     gene_names=viz_genes, genes=genes,
    #     gene_expr=vstack(datasets),
    #     kmeans=False, visualize_orig=False
    # )
    # experiment_kmeanspp(
    #     X_dimred, NAMESPACE, cell_labels=cell_labels,
    #     gene_names=viz_genes, genes=genes,
    #     gene_expr=vstack(datasets),
    #     kmeans=False, visualize_orig=False
    # )

    from ample import gs
    samp_idx = gs(X_dimred, 1000, replace=False)
    save_sketch(X, samp_idx, genes, NAMESPACE + '1000')

    for scale in [ 10, 25, 100 ]:
        N = int(X.shape[0] / scale)
        samp_idx = gs(X_dimred, N, replace=False)
        save_sketch(X, samp_idx, genes, NAMESPACE + str(N))
