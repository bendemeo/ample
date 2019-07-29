import numpy as np
import umap.umap_ as umap
from fbpca import pca
from ipywidgets import interact, interact_manual
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from copy import deepcopy
import hashers

class dataset(): #an ordered, annotated set of points
    def __init__(self, data, ft=None, path='', randomize=False):
        self.data = pd.DataFrame(data)
        self.numObs, self.numFeatures = data.shape

        if(randomize): #random datapoint ordering
            randomOrder=np.random.choice(self.numObs,self.numObs, replace=False)
            self.data = self.data[randomOrder,:]

        self.path = path #where things computed on this data will be saved
        self.subsamples = {} #sub-datasets, each with its own embedding
        self.embedding = None #computed lazily as needed


    #UMAP to embed some of the points, carrying over their annotations
    def make_embedding(self, max_size=20000):
        print('making embedding')
        size=min(max_size, self.numObs)


        print('embedding size {}'.format(size))
        embedded = self.data.iloc[:size,:self.numFeatures]



        if self.numFeatures == 2:
            self.embedding = self.data

        else:
            reducer = umap.UMAP()
            #print(embedded.values)
            self.embedding = reducer.fit_transform(embedded.values)

        self.embedding = pd.DataFrame(self.embedding)

        df = self.data.iloc[:size,self.numFeatures:]
        annos = pd.DataFrame(df)



        self.embedding.index = range(self.embedding.values.shape[0])
        annos.index = range(annos.values.shape[0])
        self.embedding = pd.concat([self.embedding, annos], axis=1, join='inner')
        #self.embedding = pd.DataFrame(np.concatenate((self.embedding.values, annos.values), axis=1))

        print(self.embedding.values.shape)


    def load_subsample(self, path, name, delimiter = '\t'): #store subsample saved on disk
        with open(path) as f:
            reader = csv.reader(f, delimiter = delimiter)
            order = np.array(list(reader)).astype('int')[0]

            # #make a place for the subsample to live
            # if not os.path.isdir(self.path+'/subsamples/'):
            #     os.mkdir(self.path+'/subsamples/')

            subsample = dataset(self.data.iloc[order,:], randomize=False, path=self.path+'/subsamples/')
            subsample.numFeatures = self.numFeatures

            # for a in self.annos.keys():
            #     subsample.annos[a]=[self.annos[a][x] for x in order]

            self.subsamples[name] = subsample

    def rawdata(self):
        return self.data.values[:,:self.numFeatures]

    def make_subsample(self, downsampler, name, max_size=20000, **kwargs):

        #construct subsampler
        sampler_func = getattr(hashers, downsampler)
        sampler = sampler_func(data=self.data.iloc[:,:self.numFeatures].values, **kwargs)

        #make the subsample
        size = min(max_size, self.numObs)
        sampler.downsample(size)
        sample = sampler.sample
        #
        # #make a place for the subsample to live
        # if not os.path.isdir(self.path+'/subsamples/'):
        #     os.mkdir(self.path+'/subsamples/')

        #bundle it into a dataset
        subsample = dataset(self.data.iloc[sample,:], randomize=False, path=self.path+'/subsamples/' )

        # for a in self.annos:
        #     subsample.annos[a]=[self.annos[a][x] for x in sample]


        if hasattr(downsampler, 'ptrs'):
            subsample['ptr'] = downsampler.ptrs

        subsample.numFeatures = self.numFeatures

        self.subsamples[name] = subsample


    def save(self, filename='data', sep='\t', include_annos=True):
        """save yourself as a CSV text file"""
        self.data.to_csv(self.path+filename, sep=sep)

    def sort_values(self, by, **kwargs):
        self.data = self.data.sort_values(by, **kwargs)

        if self.embedding is not None:
            print('sorting embedding')
            self.embedding = self.embedding.sort_values(by, **kwargs)



    def subset(self, annoName, values):


        result = dataset(self.data.loc[[x in values for x in self.data[annoName]]])
        result.numFeatures = self.numFeatures
        # anno = self.annos[annoName]
        # match_inds = []
        #
        # for i,a in enumerate(anno):
        #     if a == value:
        #         match_inds.append(i)


        #
        # result = dataset(self.data[match_inds,:])
        # for a in self.annos.keys():
        #     result.annos[a] = [self.annos[a][x] for x in match_inds]

        return(result)


    def grow(self, max_size=20000, cmap='Set1'):
#         if color is None:
#             color=self.annos.keys()

        if self.embedding is None:
            print('making embedding')
            self.make_embedding(max_size)

        max_n = min(self.embedding.shape[0],max_size)

        #annotations (besides coordinates) are possible colorings
        color = list(self.embedding.columns)[2:]


        @interact
        def build_plot(N=(1,max_n, 1), color=color):
            if color is None or len(color) == 0:
                colors = [1]*self.numObs
            else:
                le = LabelEncoder().fit(self.data.loc[:,color])
                colors = le.transform(self.data.loc[:,color])

            x=self.embedding.iloc[:N,0].values
            c=self.embedding.iloc[:N,:].loc[:,color].values.tolist()
            numPts = min([N, self.numObs])
            plt.scatter(self.embedding.values[:N,0],
                        self.embedding.values[:N,1],
                       c=colors[:N],
                        cmap = cmap)

    def grow_all(self, samples=None):
        if samples is None:
            samples = self.subsamples.keys()
        @interact
        def execute(sample=samples):
            self.subsamples[sample].grow()


    def heatmap(self, metric='euclidean', **kwargs):
        rawdata = self.data.values[:,:self.numFeatures]
        print('computing distances...')
        dists = pairwise_distances(rawdata, metric=metric, **kwargs)
        plt.imshow(1-dists, cmap='hot',interpolation='nearest')
        return(dists)



    def embed_all(self, max_size=20000):
        for s in self.subsamples.values():
            s.make_embedding(max_size=max_size)

    def hasEmbedding(self):
        return(self.embedding is not None)

    def pca_dimred(self, n_components=100, filename = None):
        U,s,Vt = pca(self.rawdata().astype(float), k=n_components)
        dimred = U[:, :n_components]*s[:n_components]
        self.data = pd.DataFrame(dimred)


        if(filename is not None):
            np.savetxt(self.path+filename+'.txt', dimred, delimiter='\t')


    def plot_sampling_rate(self, anno, normalize=True): #make a line plot showing how fast the annotation is sampled
        vals = self.data[anno].tolist()
        for val in np.unique(vals):
            positions = [i for i,x in enumerate(vals) if x == val]
            y = list(range(len(positions)))

            if normalize:
                exp_rate = float(len(positions))/float(self.numObs)
                expected = [exp_rate*x for x in positions]
                y = np.divide(y,expected)
                #y = [x/float(len(y)) for x in y]
            plt.plot(positions, y, label = '{} (N = {})'.format(val, len(positions)))
        plt.legend(loc='upper left')
        plt.show()


    def nearest_label(self, anno, name=None): #make an annotation assigning points to nearest center of another annotation
        if name is None:
            name = anno + '_nearest'
        vals = self.data[anno].tolist()
        centers = self.data.groupby(anno).mean()

        new_anno = []
        for i in range(self.numObs):
            #find the closest center
            pt = np.array(self.data.iloc[i,:self.numFeatures])

            min_dist = float("inf")
            for j in range(len(centers.index)):
                lab = centers.index[j]
                #print(lab)
                center = np.array(centers.iloc[j,:].tolist())
                dist = np.linalg.norm(pt-center)
                if dist < min_dist:
                    min_dist = dist
                    cur_anno = lab
            new_anno.append(cur_anno)

        self.data[name] = new_anno



#         for val in np.unique(vals):
#             positions = [i for i,x in enumerate(vals) if x == val]
#             points = self.data.iloc[positions,:self.numFeatures]

#             center = mean(points)

    def gaussian_label(self, anno, name=None, scaling=1): #make an annotation by fitting gaussian models to each class
        if name is None:
            name = anno + '_gaussfit'

        vals = self.data[anno].tolist()

        # to store means and covariance matrices
        means = {}
        covs = {}

        #fit Gaussians on each label
        for val in np.unique(vals):
            positions = [i for i,x in enumerate(vals) if x == val]
            points = self.data.values[positions, :self.numFeatures]
            mean = np.mean(points, axis=0)
            cov = np.cov(np.array(points).astype(float), rowvar=False)
            means[val] = mean
            covs[val] = cov*scaling

        new_anno = []
        for i in range(self.numObs):
            #print(i)
            pt = np.array(self.data.iloc[i,:self.numFeatures])

            max_prob = float(0)
            for val in np.unique(vals):
                prob = multivariate_normal.pdf([pt], mean=means[val], cov=covs[val], allow_singular=True)
                if prob > max_prob:
                    max_prob = prob
                    cur_anno = val
            new_anno.append(cur_anno)

        self.data[name] = new_anno


    def __getitem__(self, key):
        result = dataset(data=self.data.iloc[key,:])
        result.numFeatures = self.numFeatures
        return(result)
