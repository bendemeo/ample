from __future__ import division
from scipy.stats import ortho_group
import numpy as np
from LSH import *
from test_file import *
from utils import *
from time import time


class cosineLSH(LSH):

    def makeHash(self):
        projector = np.random.randn(self.numFeatures, self.numHashes)
        projection = (np.sign(np.matmul(self.data,projector))+1)/2
        self.hash = projection

class projLSH(LSH):

    def __init__(self, data, numHashes, numBands, bandSize, gridSize=0.1, replace=False):

        LSH.__init__(self, data, numHashes=numHashes, numBands=numBands, bandSize=bandSize, replace=replace)

        self.gridSize=gridSize

    def makeHash(self):
        """
        projects randomly, and bins projection values
        """

        projector = np.random.randn(self.numFeatures, self.numHashes)
        projection = np.matmul(self.data,projector)

        projection = projection - projection.min(0)
        projection = projection/projection.max()

        projection = np.floor(projection/self.gridSize)

        self.hash=projection



class randomGridLSH(LSH):
    """like gridLSH, but grid axes are randomly chosen orthogonal basis"""
    def __init__(self, data, gridSize, numHashes, numBands, bandSize, replace = False):

        LSH.__init__(self,data, numHashes=numHashes, numBands=numBands, bandSize=bandSize,
        replace=replace)

        self.gridSize = gridSize

    def makeHash(self):
        hashes = np.empty((self.numObs,self.numHashes))

        for hashno in range(self.numHashes):
            basis = rvs(dim = self.numFeatures) # random orthonormal basis

            t0 = time()
            newData=np.matmul(self.data, basis)
            t1 = time()

            print('making random basis took {} seconds'.format(t1-t0))

            #do gridLSH in this new basis

            #make positive and max 1
            X = newData - newData.min(0)
            X /= X.max()

            grid = {}

            #make dict mapping grid squares to points in it
            for i in range(self.numObs):
                coords = X[i,:]

                gridsquare = tuple(np.floor(coords / float(self.gridSize)).astype(int))

                if gridsquare not in grid:
                    grid[gridsquare]=set()
                grid[gridsquare].add(i)

            #enumerate grid squares, and assign each obs to its square index
            keys = list(grid.keys())
            for square in range(len(keys)):
                for idx in grid[keys[square]]:
                    hashes[idx,hashno] = square


        self.hash=hashes



class gridLSH(LSH):
    """just bins coordinates to make an orthogonal grid"""

    def __init__(self,data,gridSize, replace=False):
        numBands = 1
        bandSize = 1
        numHashes = 1
        LSH.__init__(self,data, numHashes=numHashes, numBands=numBands, bandSize=bandSize,
        replace=replace)
        self.gridSize=gridSize

    def makeHash(self):

        #make positive and max 1
        X = self.data - self.data.min(0)
        X /= X.max()

        #hashes = np.empty((self.numObs,self.numFeatures))
        hashes = np.empty((self.numObs, 1))

        grid = {}

        #make dict mapping grid squares to points in it
        for i in range(self.numObs):
            coords = X[i,:]

            gridsquare = tuple(np.floor(coords / float(self.gridSize)).astype(int))

            if gridsquare not in grid:
                grid[gridsquare]=set()
            grid[gridsquare].add(i)

        #enumerate grid squares, and assign each obs to its square index
        keys = list(grid.keys())
        for square in range(len(keys)):
            for idx in grid[keys[square]]:
                hashes[idx,0] = square


        self.hash=hashes

        #
        # grid_axes = ortho_group.rvs(self.numFeatures)
        #
        # for i in range(grid_axes.shape[0]):
        #     # project onto ith grid axis
        #     projection = np.matmul



if __name__ == '__main__':
    gauss2D = gauss_test([10,20,100,200], 2, 4, [0.1, 1, 0.01, 2])
    mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])

    # downsampler = gridLSH(gauss2D,0.1)
    downsampler = projLSH(gauss2D, 10, 2, 5, 0.05)

    subInds = downsampler.downSample(50)

    print(subInds)

    mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')
    mpl.show()
