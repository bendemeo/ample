import pandas
import hashers
import numpy as np
import numpy as np
import os
from scanorama import *
#import scanpy.api as sc
import scipy.stats
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from subprocess import Popen
import sys
from time import time

from process import load_names
from save_mtx import save_mtx
from supervised import adjusted_mutual_info_score
from ample import *
from utils import *
from lsh_sketch import *
from test_file import *


def try_params(X_dimred, hasher, params, tests, n_seeds=1, **kwargs):
    "version where params is a dict to be unpacked"

    #each param should have either 1 value (recycled for all) or k values
    paramLengths = [len(params[k]) for k in params.keys()]
    uniqueLengths = np.unique(paramLengths)
    assert(len(uniqueLengths)<=2)
    if len(uniqueLengths)==2:
        assert(1 in uniqueLengths)

    #how many different parameter settings to try
    numSettings = max(paramLengths)

    #recycle if length 1
    for p in params:
        if len(params[p])==1:
            params[p]=params[p]*numSettings

    print(params)

    if 'Ns' in kwargs:
        Ns=kwargs['Ns']
    else:
        Ns = [100]

    numTests = len(Ns)*n_seeds*numSettings

    #empty lists for all params and test values
    results = {t:[] for t in tests}
    for p in params.keys():
        results[p]=[]

    results['sampler'] = [hasher]*numTests
    results['N']=[]

    for i in range(numSettings):
        currentParams={p:val[i] for (p,val) in params.items()}
        hasherfunc = getattr(hashers, hasher)

        downsampler = hasherfunc(X_dimred, **currentParams)

        for N in Ns:
            for seed in range(n_seeds):

                #store current parameters
                for p in currentParams.keys():
                    results[p].append(currentParams[p])

                log('sampling {}'.format(hasher))
                t0 = time.time()
                samp_idx=downsampler.downSample(N)

                t1 = time.time()
                log('sampling {} done'.format(hasher))

                #record N
                results['N'].append(N)

                #record all test values
                for t in tests:
                    if t == 'time':
                        results['time'].append(t1-t0)
                    elif t == 'lastCounts':
                        results['lastCounts'].append(downsampler.getMeanCounts())
                    elif t == 'remnants':
                        results['remnants'].append(downsampler.getRemnants())
                    elif t == 'rare':
                        cell_labels = kwargs['cell_labels']
                        rare_label = kwargs['rare_label']
                        cluster_labels = cell_labels[samp_idx]
                        results['rare'].append(sum(cluster_labels == rare_label))
                    elif t == 'entropy':
                        cell_labels = kwargs['cell_labels']
                        cluster_labels = cell_labels[samp_idx]
                        clusters = sorted(set(cell_labels))
                        max_cluster = max(clusters)
                        cluster_hist = np.zeros(max_cluster + 1)
                        for c in range(max_cluster + 1):
                            if c in clusters:
                                cluster_hist[c] = np.sum(cluster_labels == c)
                        results['entropy'].append(normalized_entropy(cluster_hist))
                    elif t == 'kl_divergence':
                        cell_labels = kwargs['cell_labels']
                        expected = kwargs['expected']
                        cluster_labels = cell_labels[samp_idx]
                        clusters = sorted(set(cell_labels))
                        max_cluster = max(clusters)
                        cluster_hist = np.zeros(max_cluster + 1)
                        for c in range(max_cluster + 1):
                            if c in clusters:
                                cluster_hist[c] = np.sum(cluster_labels == c)
                        cluster_hist /= np.sum(cluster_hist)
                        results['kl_divergence'].append(scipy.stats.entropy(expected, cluster_hist))
                    elif t == 'max_min_dist':
                        dist = pairwise_distances(
                            X_dimred[samp_idx, :], X_dimred, n_jobs=-1
                        )
                        results['max_min_dist'].append(dist.min(0).max())
                    elif t == 'kmeans_ami':
                        cell_labels = kwargs['cell_labels']

                        k = len(set(cell_labels))
                        km = KMeans(n_clusters=k, n_init=1, random_state=kwargs['seed'])
                        km.fit(X_dimred[samp_idx, :])

                        full_labels = label_approx(X_dimred, X_dimred[samp_idx, :], km.labels_)

                        ami = adjusted_mutual_info_score(cell_labels, full_labels)

                        results['kmeans_ami'].append(ami)
                    elif t == 'kmeans_bami':
                        cell_labels = kwargs['cell_labels']
                        k = len(set(cell_labels))
                        km = KMeans(n_clusters=k, n_init=1, random_state=kwargs['seed'])
                        km.fit(X_dimred[samp_idx, :])

                        full_labels = label_approx(X_dimred, X_dimred[samp_idx, :], km.labels_)

                        bami = adjusted_mutual_info_score(
                            cell_labels, full_labels, dist='balanced'
                        )
                        results['kmeans_bami'].append(bami)
                    elif t == 'louvain_ami' or t == 'louvain_bami':
                        cell_labels = kwargs['cell_labels']

                        adata = AnnData(X=X_dimred[samp_idx, :])
                        sc.pp.neighbors(adata, use_rep='X')
                        sc.tl.louvain(adata, resolution=1., key_added='louvain')
                        louv_labels = np.array(adata.obs['louvain'].tolist())

                        full_labels = label_approx(X_dimred, X_dimred[samp_idx, :], louv_labels)

                        if t == 'louvain_ami':
                            ami = adjusted_mutual_info_score(cell_labels, full_labels)
                            results['louvain_ami'].append(ami)
                        elif t == 'louvain_bami':
                            bami = adjusted_mutual_info_score(
                                cell_labels, full_labels, dist='balanced'
                            )
                            results['louvain_bami'].append(bami)
    return results


if __name__ == '__main__':
    testdata = gauss_test([10,20,100,200], 2, 4, [0.1,1,0.01,2])

    params = {
        'numHashes':[100,100,100,1000],
        'numBands':[10,5,1,1],
        'bandSize':[10,20,100,1000]
    }


    testresults = try_params(testdata,'cosineLSH',params, ['max_min_dist','time','lastCounts','remnants'])

    print(testresults)
