## base class for all downsamplers

import numpy as np

class sampler:

    def __init__(self, data, replace=False):
        self.numObs, self.numFeatures = self.data.shape
        self.replace = replace

    def downsample(self, sampleSize):
        return np.random.choice(range(self.numObs), sampleSize)
        
