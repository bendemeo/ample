# for random projection hashing
import numpy

# adding stuff for git
# adding more stuff
class lsh:
    '''class to construct a random projection of data'''

    def __init__(self, numProj, data = None, numBands=None, bandSize=None, replace=False):
        ''' numProj is number of random projections it makes'''
        ''' dim is number of dimensions '''
        self.numProj = numProj
        self.hasProjector = False
        self.hasProjection = False
        self.hasFinder = False
        self.numBands = numBands
        self.bandSize = bandSize
        self.data = data
        self.replace = replace

        self.hasData = (data is not None)


    def makeProjector(self, dim):
        self.projector = numpy.random.randn(dim, self.numProj)
        self.hasProjector = True

    def project(self, data = None):  # make a  projection
        '''data assumed to have one col per dimension
        and one observation per row'''

        if not self.hasData:
            if data is None:
                raise ValueError('need data to project')
            else:
                self.data = data
                self.hasData = True

        if not self.hasProjector:
            self.makeProjector(self.data.shape[1])
        self.projection = (numpy.sign(numpy.matmul(self.data, self.projector))+1)/2
        self.hasProjection = True

    def getProjection(self, data = None):  # accessor for projection
        if not self.hasData:
            if data is None:
                raise ValueError('need data to make a projector')
            else:
                self.data = data
                self.hasData = True

        if not self.hasProjection:
            self.project(self.data)
        return (numpy.sign(numpy.matmul(self.data, self.projector))+1)/2

    def makeFinder(self, numBands=None, bandSize=None, data=None):
        ''' make a list of numBands dictionaries for randomly chosen subsets
        of bits'''

        if numBands is None:
            if self.numBands is None:
                raise ValueError('Number of bands not specified')
        else:
            self.numBands = numBands

        if bandSize is None:
            if self.bandSize is None:
                raise ValueError('band size not specified')
        else:
            self.bandSize = bandSize



        if not (data is None):
            self.project(data)
        elif not self.hasProjection:
            print "can't make hashmaps without data!"
        dicts = []
        subsets = []
        for i in range(self.numBands):
            inds = numpy.random.choice(self.numProj, self.bandSize, replace=False)
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
            elif self.numBands is None or self.bandSize is None:
                raise ValueError("please specify hash size (bandSize) and # of hashes (numBands)")
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

    def downSample(self, sampleSize=100, replace = True):
        if not self.hasProjection:
            self.project()

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
            toRemove = self.findCandidates(next, numBands=self.numBands, bandSize=self.bandSize)
            available = [x for x in available if x not in toRemove]
        return numpy.unique(sample)
