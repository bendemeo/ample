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
from fttree import *
import sys
import pickle
from sklearn.metrics import adjusted_rand_score
from anndata import AnnData
from scanpy.api.tl import louvain
from scanpy.api.pp import neighbors
from norms import *


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

    def CountFrequency(my_list):
        # Creating an empty dictionary
        freq = {}
        for item in my_list:
            if (item in freq):
                freq[item] += 1
            else:
                freq[item] = 1

        print(freq)

    CountFrequency(labels)

    def euclidean(p1, p2):
        return np.sqrt(np.sum(np.power(p2 - p1, 2)))
    viz_genes = []
    genes = []

    # np.savetxt('pbmc_dimred', X_dimred, delimiter = '\t')
    # print('saved pbmc')
    print(labels[1:10])
    pd.DataFrame(labels).to_csv("pbmc_labels")
    #np.savetxt('pbmc_labels', np.array(labels), delimiter='\t')


    # X_dimred = X_dimred[:,:5]
    # t0 = time()
    # tree2 = VPTree(X_dimred, dist_fn = euclidean)
    # t1 = time()
    # print('{} to make tree on whole data'.format(t1-t0))
    #
    #
    # t0 = time()
    # #theirs = tree2.get_nearest_neighbor(gauss[10,:])
    # theirs = tree2.get_nearest_neighbor(X_dimred[10,:])
    # t1 = time()
    # print('QUERY TREE: their method took {} seconds'.format(t1-t0))

    #




    # with open('target/experiments/pbmc_ft.txt', 'r') as f:
    #     order = f.readlines()[0].split('\t')
    #     order = [int(x) for x in order]
    #
    # print(order)
    #
    # full_sample = X_dimred[order,:]
    # adata = AnnData(X=full_sample)
    # neighbors(adata, use_rep='X')
    # louvain(adata, resolution=1., key_added='louvain')
    # louv_full = np.array(adata.obs['louvain'].tolist())
    # print(louv_full)
    #
    #
    # #
    # for size in range(10, 1000, 100):
    #     cur_sample = X_dimred[order[:size]]
    #     adata = AnnData(X=cur_sample)
    #     neighbors(adata, use_rep='X')
    #     louvain(adata, resolution=1., key_added='louvain')
    #
    #     louv_current = np.array(adata.obs['louvain'].tolist())
    #     print(louv_current)
    #
    #     rand_score = adjusted_rand_score(louv_full[:size], louv_current)
    #     print(rand_score)
    #
    #
    # sampler = uniformSampler(X_dimred)
    # print('Uniform stats')
    # for size in range(10, 1000, 100):
    #     sampled_inds = np.random.choice(list(range(len(order))), size, replace=False)
    #
    #     cur_sample = X_dimred[[order[i] for i in sampled_inds],:]
    #     adata = AnnData(X=cur_sample)
    #     neighbors(adata, use_rep='X')
    #     louvain(adata, resolution=1., key_added='louvain')
    #
    #     louv_current = np.array(adata.obs['louvain'].tolist())
    #     print(louv_current)
    #
    #     rand_score = adjusted_rand_score(louv_full[sampled_inds], louv_current)
    #     print(rand_score)


    sampler = FTSampler_exact(X_dimred, distfunc = neg_pearson)

    for N in np.arange(10,X_dimred.shape[0], 100):
        sampler.downsample(N)


        order = print(sampler.sample, sep='\t')

        file = open(r"target/experiments/pbmc_ft_pearson.txt", "w+")
        file.write('\t'.join([str(x) for x in sampler.sample]))
        file.close()





    #
    #
    # sampler = 'FTSampler_refined'
    # filename = 'pbmc_ft_refined'
    # picklename = None
    #
    # iter = 1
    # #dimreds = [5,6,7,8,9]+np.arange(10, 100, 5).tolist()
    #
    #
    # #radii=np.arange(1, 0.01, -0.01).tolist()
    # testParams = {
    #     'dist_fn':[euclidean]
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               cell_labels=cell_labels,
    #                               Ns=np.arange(1,20000, 200),
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #
    # #
    # #
    # sampler = 'fastBall'
    # filename = 'pbmc_fastball_PCA_adaptive'
    # picklename = None
    #
    # iter = 3
    # #dimreds = [5,6,7,8,9]+np.arange(10, 100, 5).tolist()
    # dimreds=[2,3,4,5,7,10,20]
    #
    #
    #
    # #radii=np.arange(1, 0.01, -0.01).tolist()
    #
    # sizes = np.arange(1, 30000, 500).tolist()
    # N=X_dimred.shape[0]
    # radii = np.arange(1, 0.7, -0.1).tolist()+np.arange(0.65, 0.35, -0.05).tolist()+np.arange(0.3, 0.1, -0.02).tolist()
    #
    #
    #
    # [1-(math.log(s)/math.log(N)) for s in sizes]
    # print(radii)
    #
    # testParams = {
    #     'rad':radii*len(dimreds),
    #     'dist_fn':[euclidean],
    #     'DIMRED':np.repeat(dimreds, len(radii)).tolist(),
    #     'maxSize':[25000],
    #     'PCA':[True]
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts',
    #           'occSquares']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               cell_labels=cell_labels,
    #                               Ns=['auto'],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    # filename = 'pbmc_gsLSH_subsample'
    # downsampler = gsLSH(X_dimred, opt_grid=True, target='N')
    #
    # experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels=cell_labels,
    #     gene_names=viz_genes, genes=genes,
    #     kmeans=False,
    #     visualize_orig=False,
    #     sample_type='gsLSH',
    #     lsh=True)


    #
    # sampler = 'vpSampler'
    # filename = 'pbmc_vp_tests'
    # picklename = None
    #
    # iter = 2
    # testParams = {
    #     'radius':np.arange(.4,0.01,-0.01).tolist()*3
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts',
    #           'lastCounts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               cell_labels=cell_labels,
    #                               Ns=['auto'],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # with open("gsLSH_gridTest.file", "wb") as f:
    #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    #
    # sampler = 'uniformSampler'
    # filename = 'pbmc_Uniform_hausdorff'
    # iter = 1
    # testParams = {'p':[1]}
    #
    # tests = ['time','max_min_dist']
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=5,
    #                               cell_labels=cell_labels,
    #                               Ns=np.arange(1000, 50000, 1000).tolist(),
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = None)
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')


    # #
    # filename = 'pbmc_dpp_subsample'
    # downsampler = dpp(X_dimred, steps=100000)
    #
    # experiment(downsampler, X_dimred, NAMESPACE, filename = filename, cell_labels=cell_labels,
    #     gene_names=viz_genes, genes=genes,
    #     kmeans=False,
    #     visualize_orig=False,
    #     sample_type='dpp',
    #     lsh=True)

    # sampler = 'gsLSH'
    # filename = 'pbmc_gsLSH_tests'
    # picklename = 'pbmc_gsLSH_downsamples'
    #
    # iter = 1
    # testParams = {
    #     'gridSize':np.arange(1,0.01,-0.02),
    #     'opt_grid': [False]
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts',
    #           'lastCounts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=3,
    #                               cell_labels=cell_labels,
    #                               Ns=['auto'],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')



    # sampler = 'fastBallSampler'
    # filename = 'pbmc_ringBall_tests'
    # picklename = 'pbmc_ringBall_downsamples'
    #
    #
    # # seeds10 = gsLSH(X_dimred, target=10).downsample(10)
    # # seeds30 = gsLSH(X_dimred, target=30).downsample(30)
    # seeds200 = gsLSH(X_dimred, target=150).downsample(150)
    #
    # iter = 1
    # testParams = {
    #     'gridSize':[0.003],
    #     'ball': [True],
    #     'radius': [100],
    #     'seeds':[seeds200]*1
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts',
    #           'lastCounts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               cell_labels=cell_labels,
    #                               Ns=['auto'],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')




    #
    # sampler = 'slowBallSampler'
    # filename = 'pbmc_slowBall_tests'
    # picklename = 'pbmc_slowBall_downsamples'
    #
    # iter = 1
    # testParams = {
    #     'ballSize':np.arange(0.6,0.01,-0.01).tolist()*3
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts',
    #           'lastCounts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               cell_labels=cell_labels,
    #                               Ns=['auto'],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    # sampler = 'softGridSampler'
    # filename = 'pbmc_softGrid_faster_tests'
    # picklename = 'pbmc_softGrid_faster_downsamples'
    #
    # iter = 1
    # testParams = {
    #     'gridSize':np.arange(0.9,0.01,-0.01).tolist()*3
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts',
    #           'lastCounts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               cell_labels=cell_labels,
    #                               Ns=['auto'],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')

    #
    # sampler = 'dpp'
    # filename = 'pbmc_dpp_tests_5'
    # picklename = 'pbmc_dpp_downsamples_hausdorff_2'
    #
    # iter = 1
    # testParams = {
    #     'steps': [10000]
    # }
    #
    # tests = ['time','max_min_dist',
    #           'cluster_counts']
    #
    #
    # testResults = try_params(X_dimred, sampler,
    #                               params=testParams,
    #                               tests=tests,
    #                               n_seeds=1,
    #                               cell_labels=cell_labels,
    #                               Ns=[100, 500, 1000, 2000, 5000, 10000, 20000],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    #
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')


    # sampler = 'gsLSH'
    # filename = 'pbmc_gsLSH_hausdorff'
    # picklename = 'pbmc_gsLSH_hausdorff'
    # iter = 1
    # Ns = np.arange(1, 100, 1).tolist()
    # testParams = {
    #     'opt_grid': [True],
    #     'target': ['N']
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
    #                               Ns=np.arange(10,100,1).tolist(),
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup',
    #                               picklename = picklename)
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')


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

    # sampler = 'centerSampler'
    # filename = 'pbmc_centerSamplerTest_cosinedensityWeighted'
    # iter = 1
    # testParams = {
    #     'numCenters':np.arange(2, 100, 1).tolist(),
    #     'steps': [1000],
    #     'weighted':[True]
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
    #                               Ns=[500],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup')
    # with open("gsLSH_gridTest.file", "wb") as f:
    #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv('target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')





    # subsample = np.random.choice(range(X_dimred.shape[0]), 20000)
    # downsampler = multiscaleSampler[]

    # downsampler = multiscaleSampler(X_dimred, scales = np.arange(0.01, 1, 0.01))
    #
    # sampler = 'multiscaleSampler'
    # filename = 'pbmc_multiscaleTest'
    # iter = 1
    # testParams = {
    #     'scales':[np.arange(0.01, 1, 0.02)]
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
    #                               Ns=[500,1000,5000,10000],
    #                               cluster_labels = labels,
    #                               backup=filename+'_backup')
    # # with open("gsLSH_gridTest.file", "wb") as f:
    # #     pickle.dump(gsLSH_gridTest, f, pickle.HIGHEST_PROTOCOL)
    #
    # testResults.to_csv(
    #     'target/experiments/{}.txt.{}'.format(filename, iter), sep='\t')
    #


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
