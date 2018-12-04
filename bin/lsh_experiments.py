from experiments import *

def experiments_modular(X_dimred, sampling_fns, name, n_seeds=10, **kwargs):
        columns = [
            'name', 'sampling_fn', 'replace', 'N', 'seed', 'time'
        ]

        if 'rare' in kwargs and kwargs['rare']:
            if not 'cell_labels' in kwargs:
                err_exit('cell_labels')
            if not 'rare_label' in kwargs:
                err_exit('rare_label')
            columns.append('rare')
        if 'entropy' in kwargs and kwargs['entropy']:
            if not 'cell_labels' in kwargs:
                err_exit('cell_labels')
            columns.append('entropy')

        if 'kl_divergence' in kwargs and kwargs['kl_divergence']:
            if not 'cell_labels' in kwargs:
                err_exit('cell_labels')
            if not 'expected' in kwargs:
                err_exit('expected')
            columns.append('kl_divergence')

        if 'max_min_dist' in kwargs and kwargs['max_min_dist']:
            columns.append('max_min_dist')

        if 'kmeans_ami' in kwargs and kwargs['kmeans_ami']:
            if not 'cell_labels' in kwargs:
                err_exit('cell_labels')
            columns.append('kmeans_ami')
            columns.append('kmeans_bami')

        if 'louvain_ami' in kwargs and kwargs['louvain_ami']:
            if not 'cell_labels' in kwargs:
                err_exit('cell_labels')
            columns.append('louvain_ami')
            columns.append('louvain_bami')

        of = open('target/experiments/{}.txt.1'.format(name), 'a')
        of.write('\t'.join(columns) + '\n')

        if 'Ns' in kwargs:
            Ns=kwargs['Ns']
        else:
            Ns = [ 100, 500, 1000, 5000, 10000, 20000 ]

        not_replace = set([ 'kmeans++', 'dropClust' ])

        sampling_fn_names = [x.__name__ for x in sampling_fns]

        assert(len(sampling_fns) == len(sampling_fn_names))

        for s_idx, sampling_fn in enumerate(sampling_fns):

            if sampling_fn_names[s_idx] == 'dropClust':
                dropclust_preprocess(X_dimred, name)

            for replace in [ True, False ]:
                if replace and sampling_fn_names[s_idx] in not_replace:
                    continue

                counts_means, counts_sems = [], []

                for N in Ns:
                    if N > X_dimred.shape[0]:
                        continue
                    log('N = {}...'.format(N))

                    counts = []

                    for seed in range(n_seeds):

                        if sampling_fn_names[s_idx] == 'dropClust':
                            log('Sampling dropClust...')
                            t0 = time()
                            samp_idx = dropclust_sample('data/' + name, N, seed=seed)
                            t1 = time()
                            log('Sampling dropClust done.')
                        elif sampling_fn_names[s_idx] == 'gs_gap_N':
                            log('Sampling gs_gap_N...')
                            t0 = time()
                            samp_idx = sampling_fn(X_dimred, N, k=N, seed=seed,
                                                   replace=replace)
                            t1 = time()
                            log('Sampling gs_gap_N done.')
                        elif sampling_fn_names[s_idx] == 'lshSketch':
                            if 'bandSize' in kwargs:
                                bandSize = kwargs['bandSize']
                            else:
                                bandSize = 10
                            if 'numHashes' in kwargs:
                                numHashes = kwargs['numHashes']
                            else:
                                numHashes = 10
                            kwargs['bandSize'] = bandSize
                            kwargs['numHashes'] = numHashes
                            log('Sampling {}'.format(sampling_fn_names[s_idx]))
                            t0 = time()
                            samp_idx = sampling_fn(X_dimred, N, numHashes = numHashes, bandSize=bandSize, replace = replace)
                            t1 = time()
                            log('Sampling {} done'.format(sampling_fn_names[s_idx]))
                        else:
                            log('Sampling {}...'.format(sampling_fn_names[s_idx]))
                            t0 = time()
                            samp_idx = sampling_fn(X_dimred, N, seed=seed,
                                                   replace=replace)
                            t1 = time()
                            log('Sampling {} done.'.format(sampling_fn_names[s_idx]))

                        kwargs['sampling_fn'] = sampling_fn_names[s_idx]
                        kwargs['replace'] = replace
                        kwargs['N'] = N
                        kwargs['seed'] = seed
                        kwargs['time'] = t1 - t0

                        experiment_stats(of, X_dimred, samp_idx, name, **kwargs)

        of.close()


def lsh_stats(of, X_dimred, samp_idx, name, **kwargs):
        stats = [
            name,
            kwargs['sampling_fn'],
            kwargs['replace'],
            kwargs['N'],
            kwargs['seed'],
            kwargs['time'],
            kwargs['bandSize'],
            kwargs['numHashes']
        ]

        if 'rare' in kwargs and kwargs['rare']:
            cell_labels = kwargs['cell_labels']
            rare_label = kwargs['rare_label']
            cluster_labels = cell_labels[samp_idx]
            stats.append(sum(cluster_labels == rare_label))

        if 'entropy' in kwargs and kwargs['entropy']:
            cell_labels = kwargs['cell_labels']
            cluster_labels = cell_labels[samp_idx]
            clusters = sorted(set(cell_labels))
            max_cluster = max(clusters)
            cluster_hist = np.zeros(max_cluster + 1)
            for c in range(max_cluster + 1):
                if c in clusters:
                    cluster_hist[c] = np.sum(cluster_labels == c)
            stats.append(normalized_entropy(cluster_hist))

        if 'kl_divergence' in kwargs and kwargs['kl_divergence']:
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
            stats.append(scipy.stats.entropy(expected, cluster_hist))

        if 'max_min_dist' in kwargs and kwargs['max_min_dist']:
            dist = pairwise_distances(
                X_dimred[samp_idx, :], X_dimred, n_jobs=-1
            )
            stats.append(dist.min(0).max())

        if 'kmeans_ami' in kwargs and kwargs['kmeans_ami']:
            cell_labels = kwargs['cell_labels']

            k = len(set(cell_labels))
            km = KMeans(n_clusters=k, n_init=1, random_state=kwargs['seed'])
            km.fit(X_dimred[samp_idx, :])

            full_labels = label_approx(X_dimred, X_dimred[samp_idx, :], km.labels_)

            ami = adjusted_mutual_info_score(cell_labels, full_labels)
            bami = adjusted_mutual_info_score(
                cell_labels, full_labels, dist='balanced'
            )
            stats.append(ami)
            stats.append(bami)

        if 'louvain_ami' in kwargs and kwargs['louvain_ami']:
            cell_labels = kwargs['cell_labels']

            adata = AnnData(X=X_dimred[samp_idx, :])
            sc.pp.neighbors(adata, use_rep='X')
            sc.tl.louvain(adata, resolution=1., key_added='louvain')
            louv_labels = np.array(adata.obs['louvain'].tolist())

            full_labels = label_approx(X_dimred, X_dimred[samp_idx, :], louv_labels)

            ami = adjusted_mutual_info_score(cell_labels, full_labels)
            bami = adjusted_mutual_info_score(
                cell_labels, full_labels, dist='balanced'
            )
            stats.append(ami)
            stats.append(bami)

        of.write('\t'.join([ str(stat) for stat in stats ]) + '\n')
        of.flush()
