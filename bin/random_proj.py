# for random projection hashing
import numpy

# adding stuff for git

class lsh:
    '''class to construct a random projection of data'''

    def __init__(self, k):
        ''' k is number of random projections it makes'''
        ''' dim is number of dimensions '''
        self.k = k
        self.hasProjector = False
        self.hasProjection = False
        self.hasFinder = False

    def makeProjector(self, dim):
        self.projector = numpy.random.randn(dim, self.k)
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

    def makeFinder(self, L=5, M=3, data=None):
        ''' make a list of L dictionaries for randomly chosen subsets
        of bits'''

        if not (data is None):
            self.project(data)
        elif not self.hasProjection:
            print("can't make hashmaps without data!")
        subsets = []
        for i in range(L):
            inds = numpy.random.choice(self.k, M, replace=False)
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

    def findCandidates(self, ind, L=None, M=None):
        if not self.hasFinder:
            if L is not None and M is not None:
                self.makeFinder(L, M)
            else:
                print("please specify hash size (M) and # of hashes (L)")
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

    def downSample(self, k=100):
        available = range(self.projection.shape[0])
        sample = []
        while len(available) > 0 or len(sample) < k:
            if len(available) == 0:  # reset available if not enough
                available = range(self.projection.shape[0])
            next = numpy.random.choice(available)
            sample.append(next)
            toRemove = self.findCandidates(next)
            available = [x for x in available if x not in toRemove]
        return numpy.unique(sample)
