from __future__ import division
from scipy.stats import ortho_group
import numpy as np
from LSH import *
from test_file import *
from utils import *
from time import time



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

class gsLSH(LSH):
    def __init__(self, data, gridSize=None, replace = False):
        LSH.__init__(self, data, numHashes=1, numBands=1, bandSize=1, replace=replace)

        self.gridSize=gridSize


    def makeHash(self): #re-implementation of gs, formulated as an LSH
        n_samples, n_features = X.shape

        X = self.data - self.data.min(0)
        X -= X.max()

        X_ptp = X.ptp(0)

        low_unit, high_unit = 0., max(X_ptp)

        if gridSize is None:
            gridSize=(low_unit + high_unit) / 4

        unit = gridSize

        grid_table = np.zeros((n_samples, n_features))

        for d in range(n_features):
            if X_ptp[d] <= unit:
                continue

            points_d = X[:, d]
            curr_start = None
            curr_interval = -1
            for sample_idx in np.argsort(points_d):
                if curr_start is None or \
                   curr_start + unit < points_d[sample_idx]:
                    curr_start = points_d[sample_idx]
                    curr_interval += 1
                grid_table[sample_idx, d] = curr_interval

        grid = {}

        for sample_idx in range(n_samples):
            grid_cell = tuple(grid_table[sample_idx, :])
            if grid_cell not in grid:
                grid[grid_cell] = []
            grid[grid_cell].append(sample_idx)

        del grid_table

        #enumerate grid squares, and assign each obs to its square index
        keys = list(grid.keys())
        for square in range(len(keys)):
            for idx in grid[keys[square]]:
                hashes[idx,0] = square


        self.hash=hashes





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
        t0 = time()
        hashes = np.empty((self.numObs,self.numHashes))
        t1 = time()

        print('initiating took {} seconds'.format(t1-t0))

        for hashno in range(self.numHashes):
            t0 = time()
            basis = rvs(dim = self.numFeatures) # random orthonormal basis


            newData=np.matmul(self.data, basis)
            t1 = time()

            #print('making random basis took {} seconds'.format(t1-t0))

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
