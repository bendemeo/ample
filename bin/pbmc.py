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
import sys
import pickle


NAMESPACE = 'pbmc_facs'
METHOD = 'svd'
DIMRED = 100

data_names = [
    'data/pbmc/68k'
]

def plot(X, title, labels, bold=None):
    plot_clusters(X, labels)
    if bold:
        plot_clusters(X[bold], labels[bold], s=20)
    plt.title(title)
    plt.savefig('{}.png'.format(title))

if __name__ == '__main__':

    if 'pickle_short' in sys.argv:
        X_dimred = pickle.load(open('pickles/pbmcshort', 'rb'))
        labels = pickle.load(open('pickles/pbmclabelsshort', 'rb'))
        ext='short'
    elif 'pickle' in sys.argv:
        ext = ''
        X_dimred = pickle.load(open('pickles/pbmc', 'rb'))
        labels = pickle.load(open('pickles/pbmclabels', 'rb'))
    else:

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
            open('data/cell_labels/pbmc_68k_cluster.txt')
            .read().rstrip().split()
        )

        ext = ''
        if 'short' in sys.argv:
            print('shortening it!')
            inds = np.random.choice(range(X_dimred.shape[0]), 1000)
            X_dimred = X_dimred[inds,:]
            labels = [ labels[i] for i in inds ]
            ext = 'short'

        if 'pickleit' in sys.argv:
            print('pickling it!')
            numsamples = X_dimred.shape[0]
            picklename = 'pbmc{}'.format(ext) #short or nothing
            labelname = 'pbmclabels{}'.format(ext)
            pickle.dump(X_dimred, open('pickles/{}'.format(picklename), 'wb'))
            pickle.dump(labels, open('pickles/{}'.format(labelname), 'wb'))

    le = LabelEncoder().fit(labels)
    cell_labels = le.transform(labels)


    viz_genes = []
    genes = []

    filename = 'pbmc_dpp_subsample'
    downsampler = dpp(X_dimred, steps=100000)

    experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels=cell_labels,
        gene_names=viz_genes, genes=genes,
        kmeans=False,
        visualize_orig=False,
        sample_type='dpp',
        lsh=True)



    # print(np.unique(cell_labels))
    # print(labels[1])
    # labels = np.array(labels)
    # print(np.unique(labels))
    # print(labels.size)
    #
    # downsampler = centerSampler(X_dimred, steps=1000, numCenters=100)
    # downsampler.downsample(5000)
    # downsampler.embedSample()
    #
    #
    # labels = labels[downsampler.sample]
    # print(np.unique(labels))
    #
    # labels = np.transpose(labels)
    # labels = np.reshape(labels, (labels.size, 1))
    # print(downsampler.sampleEmbedding.shape)
    # print(labels.shape)
    # plotData = np.concatenate((downsampler.sampleEmbedding, labels), axis=1)
    # plotData = pd.DataFrame(plotData, columns = ['x','y','cell_type'])
    # plotData.to_csv('plotData/pbmc_centerSampler_plotData_5000_100centers', sep='\t')

    #
    # sampler = 'diverseLSH'
    # filename = 'pbmc_diverseLSHTest_mcmc'
    # iter = 1
    # testParams = {
    #     'numCenters':np.arange(2, 100, 2).tolist() * 2,
    #     'steps': [1000]*49 + [1000]*49
    # }
    #
    # tests = ['max_min_dist', 'time', 'maxCounts',
    #           'cluster_counts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=5,
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
    #
    #
    # viz_genes = []
    # genes = []
    #
    sampler = 'centerSampler'
    filename = 'pbmc_centerSamplerTest_spherical'
    iter = 1
    testParams = {
        'numCenters':np.arange(1, 100, 1).tolist(),
        'steps': [1000],
        'spherical':[True]
    }

    tests = ['time','max_min_dist',
              'cluster_counts']


    testResults = try_params(X_dimred, sampler,
                                  params=testParams,
                                  tests=tests,
                                  n_seeds=6,
                                  cell_labels=cell_labels,
                                  Ns=[500],
                                  cluster_labels = labels,
                                  backup=filename+'_backup')
    # with open("gsLSH_gridTest.file", "wb") as f:
    #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)

    testResults.to_csv(
        'target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    # downsampler = gsLSH(X_dimred, gridSize=0.4)
    # # downsampler.downsample(5000)
    # # downsampler.vizSample(full=False, c=np.array(downsampler.labels)[downsampler.sample],
    # #                       file='pbmc_diverseLSH_sample_5000')
    #
    #
    # #downsampler.qTransform(q=4)
    #
    # viz_genes = []
    # genes = []
    # filename = 'pbmc_gsLSH_sample_5000'
    # experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels=cell_labels,
    #     gene_names=viz_genes, genes=genes,
    #     kmeans=False,
    #     visualize_orig=False,
    #     sample_type='diverseLSH',
    #     lsh=True)

    # downsampler = diverseLSH(X_dimred, numCenters=20, batch=1000, labels=cell_labels)
    # # downsampler.downsample(5000)
    # # downsampler.vizSample(full=False, c=np.array(downsampler.labels)[downsampler.sample],
    # #                       file='pbmc_diverseLSH_sample_5000')
    #
    #
    # downsampler.qTransform(q=4)
    #
    # viz_genes = []
    # genes = []
    # filename = 'pbmc_diverseLSH_sample_5000_q4'
    # experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels=cell_labels,
    #     gene_names=viz_genes, genes=genes,
    #     kmeans=False,
    #     visualize_orig=False,
    #     sample_type='diverseLSH',
    #     lsh=True)



    #
    # sampler = 'diverseLSH'
    # filename = 'pbmc_diverseLSHTest_q4'
    # iter = 1
    # testParams = {
    #     'numCenters':np.arange(2, 100, 2).tolist() * 2,
    #     'batch': [1000]*49 + [5000]*49
    # }
    #
    # tests = ['max_min_dist', 'time', 'maxCounts',
    #           'cluster_counts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=5,
    #                               cell_labels=cell_labels,
    #                               Ns=[500],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               q=4)
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv(
    #     'target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #


    # filename = 'pbmc_ballLSHTest'
    # iter = 1
    # testParams = {
    #     'epsilon': np.arange(start=1.8, stop=1, step=-0.02).tolist(),
    #     'ord':[float('inf')],
    #     'batch':[2000]
    # }
    #
    # tests = ['max_min_dist', 'time', 'maxCounts',
    #           'cluster_counts', 'occSquares', 'centers']
    #
    # X_dimred_scaled = X_dimred / X_dimred.max()
    # ballLSH_gridTest = try_params(X_dimred_scaled, 'ballLSH',
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=3,
    #                               cell_labels=cell_labels,
    #                               Ns=[100, 500,1000],
    #                               cluster_labels = labels
    #                               )
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # ballLSH_gridTest.to_csv(
    #     'target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #

    # size=1000
    # downsampler = svdSampler(X_dimred, batch=500)
    # # downsampler.normalize()
    # downsampler.downsample(size)
    # print('visualizing...')
    # downsampler.vizSample(file='pbmc_downsample_{}'.format(size),
    #                       c=list(range(size)), cmap='hot', anno=True, full=False)
    #
    #
    #
    #
    # filename='pbmc_svdTest_500'
    # experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels='grid',
    #     gene_names=viz_genes, genes=genes, gene_expr=vstack(datasets),
    #     kmeans=False,
    #     visualize_orig=False,
    #     sample_type='gsLSH_wt',
    #     lsh=True, optimize_grid_size=False,
    #     weighted = True, alpha = alpha)

    # experiment(gs_gap, X_dimred, NAMESPACE, filename='orig_fn', cell_labels=cell_labels,
    #             gene_names=viz_genes, genes=genes, gene_expr=vstack(datasets),
    #             kmeans=False,
    #             visualize_orig=False,
    #             sample_type='gsLSH_wt',
    #             lsh=False,
    #             weighted = True, alpha = alpha
    # )


    # downsampler = gsLSH(X_dimred)
    # for alpha in np.arange(1,10,0.5):
    #     filename='pbmc_gsLSHTest_weighted_alpha_{}'.format(alpha)
    #     experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels=cell_labels,
    #         gene_names=viz_genes, genes=genes, gene_expr=vstack(datasets),
    #         kmeans=False,
    #         visualize_orig=False,
    #         sample_type='gsLSH_wt',
    #         lsh=True, optimize_grid_size=True,
    #         weighted = True, alpha = alpha)

    # print('cell labels originally: ')
    # print(cell_labels)
    #
    # downsampler = gsLSH(X_dimred, target='N')
    # alpha=2
    # filename='pbmc_gsLSHTest_N_gridviz_weighted_{}'.format(alpha)
    #

    #
    #
    #
    # downsampler = gsLSH(X_dimred)
    # alpha=2
    # filename='pbmc_gsLSHTest_sqrtN_gridviz_weighted_{}'.format(alpha)
    #
    # experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels='grid',
    #     gene_names=viz_genes, genes=genes, gene_expr=vstack(datasets),
    #     kmeans=False,
    #     visualize_orig=False,
    #     sample_type='gsLSH_wt',
    #     lsh=True, optimize_grid_size=False,
    #     weighted = True, alpha = alpha)

    # filename='gsGridTest_clustcounts_wt'
    # iter=1
    # gsGridTestParams = {
    #  'opt_grid':[False],
    #  'gridSize': np.arange(start=1,stop=0.01,step=-0.01).tolist()
    # }
    #
    # gsGridTests = ['max_min_dist','time','cluster_counts', 'maxCounts']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'gsLSH',
    #  params=gsGridTestParams,
    #  tests=gsGridTests,
    #  n_seeds=3,
    #  cell_labels=cell_labels,
    #  cluster_labels = labels,
    #  weighted=True,
    #  Ns=[1000]
    #  )
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    #
    # filename='pbmc_gsGridTest_clustcounts_nonwt'
    # iter=1
    # gsGridTestParams = {
    #  'opt_grid':[False],
    #  'gridSize': np.arange(start=1,stop=0.01,step=-0.01).tolist()
    # }
    #
    # gsGridTests = ['max_min_dist','time','cluster_counts', 'maxCounts']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'gsLSH',
    #  params=gsGridTestParams,
    #  tests=gsGridTests,
    #  n_seeds=3,
    #  cell_labels=cell_labels,
    #  cluster_labels = labels,
    #  weighted=False,
    #  Ns=[1000]
    #  )
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    # # experiment_gs(
    # #     X_dimred, NAMESPACE, cell_labels=cell_labels,
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

    #
    # filename='pbmc_gridLSHTest_clustcounts'
    # iter=3
    # gsGridTestParams = {
    #  'randomize_origin':[False],
    #  'gridSize': np.arange(start=1,stop=0.01,step=-0.02).tolist()
    # }
    #
    # gsGridTests = ['max_min_dist','time','cluster_counts', 'maxCounts']
    #
    # gsLSH_gridTest = try_params(X_dimred,'gridLSH',
    #  params=gsGridTestParams,
    #  tests=gsGridTests,
    #  n_seeds=3,
    #  cell_labels=cell_labels,
    #  cluster_labels = labels,
    #  weighted=False,
    #  Ns=[500]
    #  )
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}_{}.txt.{}'.format(filename, ext, iter), sep='\t')
    #
    #
    # filename='pbmc_gridLSHTest_clustcounts_randomorigin'
    #
    # iter=4
    # print('filename will be {}'.format('target/experiments/{}_{}.txt.{}'.format(filename, ext, iter)))
    #
    #
    # gsGridTestParams = {
    #  'randomize_origin':[True],
    #  'gridSize': np.arange(start=1,stop=0,step=-0.1).tolist()
    # }
    #
    # gsGridTests = ['time','cluster_counts', 'maxCounts', 'remnants', 'occSquares']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'gridLSH',
    #  params=gsGridTestParams,
    #  tests=gsGridTests,
    #  n_seeds=1,
    #  cell_labels=cell_labels,
    #  cluster_labels = labels,
    #  weighted=False,
    #  Ns=[1000]
    #  )
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}_{}.txt.{}'.format(filename, ext, iter), sep='\t')
    # filename='pbmc_gridLSHTest_clustcounts'
    # iter=3
    # gsGridTestParams = {
    #  'randomize_origin':[False],
    #  'gridSize': np.arange(start=1,stop=0.01,step=-0.05).tolist(),
    #  'cell_labels': [cell_labels],
    #  'cluster_labels': [labels],
    #  'record_counts': [True]
    # }
    #
    # gsGridTests = ['max_min_dist','time','cluster_counts', 'maxCounts', 'remnants','cluster_scores']
    #
    # gsLSH_gridTest = try_params(X_dimred,'gridLSH',
    #  params=gsGridTestParams,
    #  tests=gsGridTests,
    #  n_seeds=1,
    #  cell_labels=cell_labels,
    #  cluster_labels = labels,
    #  weighted=False,
    #  Ns=[100]
    #  )
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}_{}.txt.{}'.format(filename, ext, iter), sep='\t')

    # downsampler = splitLSH(X_dimred, minDiam=0.35)
    # downsampler.makeHash()
    # print('vizualizing...')
    # downsampler.vizHash('splithash_pbmc', maxPoints=10000)



    # filename='pbmc_treeLSHTest_clustcounts'
    #
    # iter=1
    # print('filename will be {}'.format('target/experiments/{}_{}.txt.{}'.format(filename, ext, iter)))
    #
    #
    # TestParams = {
    #  'splitSize':[0.2]*4,
    #  'children': [2,3,4,5]
    # }
    #
    # gsGridTests = ['time','max_min_dist', 'occSquares','cluster_counts']
    #
    # gsLSH_gridTest = try_params(X_dimred, 'treeLSH',
    #  params=TestParams,
    #  tests=gsGridTests,
    #  n_seeds=3,
    #  cell_labels=cell_labels,
    #  cluster_labels = labels,
    #  weighted=False,
    #  Ns=[100,300, 500, 700, 1000]
    #  )
    #
    # gsLSH_gridTest.to_csv('target/experiments/{}_{}.txt.{}'.format(filename, ext, iter), sep='\t')
