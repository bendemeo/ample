# for random projection hashing
import numpy

# adding stuff for git

class lsh:
    '''class to construct a random projection of data'''

    def __init__(self, data = None, numProj = 100, numBands = 10, bandSize = 10):
        ''' k is number of random projections it makes'''
        ''' dim is number of dimensions '''
        self.numProj = numProj
        self.hasProjector = False
        self.hasProjection = False
        self.hasFinder = False
        self.numBands = numBands
        self.bandSize = bandSize
        self.data = data

    def makeProjector(self, dim):
        self.projector = numpy.random.randn(dim, self.numProj)
        self.hasProjector = True

    def project(self, data):  # make a  projection
        '''data assumed to have one col per dimension
        and one observation per row'''
        if not self.hasProjector:
            self.makeProjector(data.shape[1])
        self.projection = (numpy.sign(numpy.matmul(data, self.projector))+1)/2
        self.hasProjection = True

    def getProjection(self, data):  # accessor for projection
        if not self.hasProjection:
            self.project(data)
        return (numpy.sign(numpy.matmul(data, self.projector))+1)/2

    def makeFinder(self, numBands=5, bandSize=3, data=None):
        ''' make a list of numBands dictionaries for randomly chosen subsets
        of bits'''

        if not (data is None):
            self.project(data)
        elif not self.hasProjection:
            print("can't make hashmaps without data!")
        subsets = []
        dicts = []
        for i in range(numBands):
            inds = numpy.random.choice(self.numProj, bandSize, replace=False)
            subsets.append(inds)
            keys = ([tuple(self.projection[row, inds]) for row in range(self.projection.shape[0])])
            newDict = {}
            for j in range(len(keys)):
                k = keys[j]
                if k in newDict:
                    # print('key ', k)
                    # print('value ', newDict[k])
                    # print('whole dict', newDict)
                    newDict[k].append(j)
                else:
                    newDict[k] = [j]
            dicts.append(newDict)
        # print dicts
        self.finder = dicts
        self.bands = subsets  # indicies corresponding to 'bands'
        self.hasFinder = True

    def findCandidates(self, ind, numBands=None, bandSize=None):
        if not self.hasFinder:
            if numBands is not None and bandSize is not None:
                self.makeFinder(numBands, bandSize)
            else:
                print("please specify hash size (bandSize) and # of hashes (numBands)")
                return(None)
        candidates = []
        for i in range(len(self.bands)):
            band = self.bands[i]
            d = self.finder[i]
            key = tuple(self.projection[ind, band])
            candidates = candidates + d.get(key)  # this is O(n) on gauss
        #
        # for d in self.finder:
        #     for v in list(d.values()):
        #         # print('value ', v)
        #         if ind in v:
        #             candidates = candidates + v
        return numpy.unique(candidates)

    def downSample(self, sampleSize=100, replace = False):
        if not self.hasFinder:
            self.makeFinder(numBands=self.numBands, bandSize = self.bandSize, data =self.data)
        available = range(self.projection.shape[0])
        sample = []
        while len(available) > 0 or len(sample) < sampleSize:
            if len(available) == 0:  # reset available if not enough
                if replace:
                    available = range(self.projection.shape[0])
                else:
                    available = [x for x in range(self.projection.shape[0]) if x not in sample]
            next = numpy.random.choice(available)
            sample.append(next)
            toRemove = self.findCandidates(next)
            available = [x for x in available if x not in toRemove]
        return numpy.unique(sample)
