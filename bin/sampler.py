# base class for all downsamplers

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as mpl
import math
import os
import itertools
from utils import *
from MulticoreTSNE import MulticoreTSNE


class sampler:

    def __init__(self, data, NAMESPACE='', labels=None, replace=False):
        self.numObs, self.numFeatures = data.shape
        self.replace = replace
        self.data = data
        self.embedding = None
        self.sampleEmbedding = None
        self.labels = labels  # existing cluster labels if any
        self.sample = None

    def downsample(self, sampleSize):
        self.sampleEmbedding = None # clear previous sample embeddings
        self.sample = np.random.choice(range(self.numObs), sampleSize)
        return self.sample

    def embed(self, subset=None, **kwargs):
        if subset is None:
            subset = list(range(self.numObs))
        if self.numFeatures == 2:
            self.embedding = self.data[subset, :]
        else:
            tsne = MulticoreTSNE(**kwargs)
            # tsne = sk.manifold.TSNE(**kwargs)
            fit = tsne.fit(self.data[subset, :])
            self.embedding = tsne.embedding_

    def embedSample(self, **kwargs):
        if self.numFeatures == 2:
            self.sampleEmbedding = self.data[self.sample,:]

        else:
            tsne = MulticoreTSNE(**kwargs)
            # tsne = sk.manifold.TSNE(**kwargs)
            fit = tsne.fit(self.data[self.sample,:])
            self.sampleEmbedding = tsne.embedding_

    def normalize(self, method='l2'):
        """normalize observations, default by L2 norm"""
        self.data = sk.preprocessing.normalize(self.data, axis=1, norm=method)


    def qTransform(self, q=4):
        bins = np.empty(self.data.shape)
        for i in range(self.numFeatures):
            bins[:,i] = pd.qcut(self.data[:,i], q, labels=False)

        print(bins)
        result = np.empty([self.numObs, q*self.numFeatures])
        for i in range(self.numObs):
            row = [[0]*q for x in range(self.numFeatures)]
            for j in range(self.numFeatures):
                row[j][int(bins[i,j])]=1

            result[i,:]= [x for y in row for x in y]

        self.numObs, self.numFeatures = result.shape
        self.data = result
        # print(result)



    #def viz(self, file=None, size=self.numObs, c='b'):


    def vizData(self, file=None, **kwargs):
        if self.embedding is None:
            self.embed()
        mpl.scatter(self.embedding[:,0], self.embedding[:,1], **kwargs)
        mpl.show()
        mpl.close()


    def vizSample(self, file=None, full=False,  c='m', show=True, cmap='Set1',anno=False, annoMax=100, **kwargs):
        if(full):
            if self.embedding is None:
                self.embed()
            mpl.scatter(self.embedding[:,0], self.embedding[:,1])
            mpl.scatter(self.embedding[self.sample, 0], self.embedding[self.sample,1], c=c, cmap=cmap)

            if(anno):
                for i in range(min([len(self.sample),annoMax])):
                    mpl.annotate(i, (self.embedding[self.sample[i],0], self.embedding[self.sample[i],1]))

        else:
            print('embedding sample only')
            if self.sampleEmbedding is None:
                self.embedSample()
            print(self.sampleEmbedding.shape)
            mpl.scatter(self.sampleEmbedding[:,0], self.sampleEmbedding[:,1], c=c, cmap=cmap)

            if(anno):
                for i in range(min([len(self.sample),annoMax])):
                    mpl.annotate(i, (self.sampleEmbedding[i,0], self.sampleEmbedding[i,1]))


        mpl.legend()

        if file is not None:
            mpl.savefig('{}.png'.format(file))

        if(show):
            mpl.show()
            mpl.close()

class rankSampler(sampler):
    """any sampler that ranks the cells in an order and samples deterministically"""


    def __init__(self,data, replace=False):
        sampler.__init__(self, data, replace)
        self.ranking = None
        self.embedding=None


    def rank(self):
        self.ranking = range(len(self.numObs))

    def downsample(self, sampleSize):
        inds = self.ranking[0:sampleSize]
        self.sample = inds
        return(inds)


    def vizRanking(self, file=None, **kwargs):
        if self.embedding is None:
            self.embed(**kwargs)

        mpl.scatter(self.embedding[:,0], self.embedding[:,1], c = self.ranking, cmap='viridis')

        if file is not None:
            mpl.savefig('{}.png'.format(file))

        mpl.show()

class seqSampler(sampler):
    """anything that iteratively adds to the sample based on some criterion"""

    def __init__(self, data, replace=False):
        sampler.__init__(self,data, replace)
        self.sample = []
        self.avail = list(range(self.numObs))

    def addSample(self):
        self.sample.append(np.random.choice([x for x in range(self.numObs)
                                             if not(x in self.sample)]))
    def downsample(self, sampleSize, viz=False, file=None, **kwargs):
        while(len(self.sample) < sampleSize):
            self.addSample()

        if viz and (len(self.sample) % 10) == 0:
            self.vizSample(file=file, **kwargs)

        return(self.sample)

class weightedSampler(sampler):
    def __init__(self, data, strength=1, replace=False):
        sampler.__init__(self, data, replace)
        self.strength=strength
        self.wts = None

    def makeWeights(self):
        self.wts = [float(1)/self.numObs]*self.numObs

    def downsample(self, sampleSize):
        # print('doing weighted downsampling')
        # print(self.wts)
        if self.wts is None:
            self.makeWeights()

        self.sample = np.random.choice(range(self.numObs), sampleSize, p=self.wts, replace = self.replace)
        return(self.sample)

    def vizWeights(self, log=True, file = None, dpi=None, **kwargs):
        print('wts at time of viz')
        print(self.wts)
        # tsne = sk.manifold.TSNE(**kwargs)
        #
        # fit = tsne.fit(self.data)
        self.embed()
        embedding = self.embedding

        if log:
            colors = [math.log(w) for w in self.wts]
        else:
            colors = self.wts
        mpl.scatter(embedding[:,0], embedding[:,1], c = colors, **kwargs)

        if file is not None:
            mpl.savefig('{}.png'.format(file), dpi=dpi)

        mpl.show()
        mpl.close()

class neighborhoodSampler(sampler):

    def __init__(self, data, replace=False):
        sampler.__init__(self, data, replace)
        self.available = range(self.numObs)
        self.included = [True] * self.numObs
        self.valid_sample = [True] * self.numObs
        self.count = 0
        self.reset = False #whether we have reset

    def findCandidates(self, idx):
        print('calling base findCandidates')
        ##override this
        return([idx])

    def fullReset(self):
        """everything that needs to be done to start sampling again"""
        """override this as necessary"""

    def downsample(self, sampleSize = 'auto'):
        self.lastCounts = []
        sample = []
        self.count = 0
        self.reset = False
        while True:
            if len(self.available) == 0: ## reset if still have more to sample
                self.reset = True
                log("sampled {} out of {} before reset".format(self.count, sampleSize))
                self.lastCounts.append(self.count)

                if sampleSize == 'auto':  # stop sampling when you run out
                    break

                self.count = 0
                self.available = list(itertools.compress(range(self.numObs), self.valid_sample))

                #reset included so only available indices are true
                self.included = [False]*self.numObs
                for i in self.available:
                    self.included[i] = True

                self.fullReset()

            next = np.random.choice(self.available)
            sample.append(next)
            self.valid_sample[next] = False
            if (sampleSize != 'auto') and (len(sample) >= sampleSize):
                break

            self.count += 1

            toRemove = self.findCandidates(next)
            print('removed {}'.format(len(toRemove)))

            for i in toRemove:
                self.included[i] = False

            self.available = list(itertools.compress(range(self.numObs), self.included))

            print('{} left'.format(len(self.available)))
        if not self.reset:
            self.remnants = len(self.available)
        else:
            self.remnants = 0

        self.sample = sorted(np.unique(sample))

        return(self.sample)



        # tsne = TSNEApprox(n_iter=500, perplexity=1200,
        # verbose=2, random_state=69,
        # learning_rate=200.,
        # early_exaggeration=12.
        # )
        #
        # tsne.fit(self.data)
    #
    # tsne.fit(np.concatenate(assembled))
    # embedding = tsne.embedding_
