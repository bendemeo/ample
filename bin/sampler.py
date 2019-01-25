## base class for all downsamplers

import numpy as np
import sklearn as sk
import matplotlib.pyplot as mpl
import math

class sampler:

    def __init__(self, data, replace=False):
        self.numObs, self.numFeatures = data.shape
        self.replace = replace
        self.data = data


    def downsample(self, sampleSize):
        return np.random.choice(range(self.numObs), sampleSize)

class weightedSampler(sampler):
    def __init__(self, data, strength=1, replace=False):
        sampler.__init__(self, data, replace)
        self.strength=strength
        self.wts = None

    def makeWeights(self):
        self.wts = [float(1)/self.numObs]*self.numObs

    def downsample(self, sampleSize):
        if self.wts is None:
            self.makeWeights()

        return(np.random.choice(range(self.numObs), sampleSize, p=self.wts, replace = self.replace))

    def vizWeights(self, log=True, file = None, **kwargs):
        tsne = sk.manifold.TSNE(**kwargs)

        fit = tsne.fit(self.data)
        embedding = tsne.embedding_

        if log:
            colors = [math.log(w) for w in self.wts]
        else:
            colors = self.wts
        mpl.scatter(embedding[:,0], embedding[:,1], c = colors, cmap='viridis')

        if file is not None:
            mpl.savefig('{}.png'.format(file))


        mpl.show()



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
