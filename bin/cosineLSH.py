import numpy as np
from LSH import *

class cosineLSH(LSH):

    def makeHash(self):
        projector = np.random.randn(self.numFeatures, self.numHashes)
        projection = (np.sign(np.matmul(self.data,projector))+1)/2
        self.hash = projection
