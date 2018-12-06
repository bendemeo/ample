from scipy.stats import ortho_group
import numpy as np
from LSH import *
from test_file import *


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



class gridLSH(LSH):
    """just bins coordinates to make an orthogonal grid"""

    def __init__(self,data,gridSize):
        numBands = 1
        bandSize = data.shape[1]
        numHashes = data.shape[1]
        LSH.__init__(self,data, numHashes=numHashes, numBands=numBands, bandSize=bandSize)

        self.gridSize=gridSize

    def makeHash(self):

        #make positive and max 1
        X = self.data - self.data.min(0)
        X /= X.max()

        hashes = np.empty((self.numObs,self.numFeatures))
        for i in range(self.numObs):
            coords = X[i,:]
            hashes[i,:] = [np.floor(y/self.gridSize) for y in coords]

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
    print downsampler.hash

    subInds = downsampler.downSample(50)

    print(subInds)

    mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')
    mpl.show()
