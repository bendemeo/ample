# for random projection hashing
import numpy
import sys
from utils import *

# adding stuff for git
# adding more stuff
class LSH:
    '''class to construct a random projection of data'''

    def __init__(self, data, numHashes, numBands, bandSize, replace=False, keepStats=True):
        ''' numHashes is number of random projections it makes'''
        ''' dim is number of dimensions '''
        self.numHashes = numHashes
        self.data = data
        self.numBands = numBands
        self.bandSize = bandSize
        self.replace = replace
        self.numObs, self.numFeatures = self.data.shape
        self.lastCounts = None  # how many sampled before reset
        self.remnants = None  # how many are still fair game after sampling
        self.keepStats = keepStats

        #to be updated by further function calls
        self.hash = None
        self.finder = None
        self.bands = None


    def makeHash(self):
        "child classes extend this"
        pass

    def getHash(self):
        return self.hash





    #
    # def makeProjector(self, dim):
    #     self.projector = numpy.random.randn(self.numObs, self.numHashes)
    #     self.hasProjector = True
    #
    # def project(self, data = None):  # make a  projection
    #     '''data assumed to have one col per dimension
    #     and one observation per row'''
    #
    #     if not self.hasData:
    #         if data is None:
    #             raise ValueError('need data to project')
    #         else:
    #             self.data = data
    #             self.hasData = True
    #
    #     if not self.hasProjector:
    #         self.makeProjector(self.data.shape[1])
    #     self.projection = (numpy.sign(numpy.matmul(self.data, self.projector))+1)/2
    #     self.hasProjection = True
    #
    # def getProjection(self, data = None):  # accessor for projection
    #     if not self.hasData:
    #         if data is None:
    #             raise ValueError('need data to make a projector')
    #         else:
    #             self.data = data
    #             self.hasData = True
    #
    #     if not self.hasProjection:
    #         self.project(self.data)
    #     return (numpy.sign(numpy.matmul(self.data, self.projector))+1)/2


    def makeFinder(self):
        ''' make a list of numBands dictionaries for randomly chosen subsets
        of bits. numbands and bandsize can replace global'''
        dicts = []  # list of dictionaries
        subsets = []  #list of indices comprising bands

        for i in range(self.numBands):
            inds = numpy.random.choice(self.numHashes, self.bandSize, replace=False)
            subsets.append(inds)

            #hash values on newly created band
            keys = ([tuple(self.hash[row, inds]) for row in range(self.hash.shape[0])])

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

    def findCandidates(self, ind):
        "find all indices agreeing in at least one band"

        if self.finder is None: #  construct finder if none exists
            self.makeFinder()

        candidates = []
        for i in range(len(self.bands)):
            band = self.bands[i]
            d = self.finder[i]
            key = tuple(self.hash[ind, band])
            candidates = candidates + d.get(key)  # this is O(n) on gauss

        return numpy.unique(candidates)


    def getMeanCounts(self):
        if len(self.lastCounts) == 0:
            return -1  # default if you never reset
        else:
            return sum(self.lastCounts)/len(self.lastCounts)

    def getRemnants(self):
        return self.remnants

    def downSample(self, sampleSize=100, replace = True):

        #randomly make new hashes for each downsampling
        self.makeHash()
        self.makeFinder()

        available = range(self.hash.shape[0])
        sample = []
        count = 0 # how many have been added since reset
        reset = False  # whether we have reset

        if self.keepStats:
            self.lastCounts=[]

        while len(sample) < sampleSize:
            if len(available) == 0:  # reset available if not enough
                reset = True
                log("sampled {} out of {} before reset".format(count, sampleSize))
                if(self.keepStats):
                    self.lastCounts.append(count)
                count = 0
                if replace:
                    available = range(self.hash.shape[0])
                else:
                    available = [x for x in range(self.hash.shape[0]) if x not in sample]

            next = numpy.random.choice(available)
            sample.append(next)

            count = count + 1

            toRemove = self.findCandidates(next)
            available = [x for x in available if x not in toRemove]
            log('{} remaining'.format(len(available)))

        if self.keepStats:
            if not reset:
                self.remnants = len(available)
            else:
                self.remnants = 0

        return numpy.unique(sample)
