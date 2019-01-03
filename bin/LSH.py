# for random projection hashing
import numpy
import sys
from time import time
from utils import *
import itertools

# adding stuff for git
# adding more stuff
class LSH:
    '''class to construct a random projection of data'''

    def __init__(self, data, numHashes, numBands, bandSize, replace=False, keepStats=True,
    allowRepeats = True):
        ''' numHashes is number of random projections it makes'''
        ''' dim is number of dimensions '''
        self.numHashes = numHashes
        self.data = data
        self.numBands = numBands
        self.bandSize = bandSize
        self.replace = replace
        self.numObs, self.numFeatures = self.data.shape
        self.lastCounts = []  # how many sampled before reset
        self.remnants = None  # how many are still fair game after sampling
        self.keepStats = keepStats
        self.allowRepeats = allowRepeats

        #to be updated by further function calls
        self.hash = None
        self.finder = None
        self.bands = None

        self.guess = None  # analytic guess for how many would be sampled before reset
        self.actual = None
        self.error = None  # difference from actual value

        self.nbhdSizes = [] # store neighborhood sizes
        self.takenBands=[[]]*numBands  #band values that have been taken
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

            available = range(self.numHashes)

            if self.allowRepeats:
                inds = numpy.random.choice(int(self.numHashes), int(self.bandSize), replace=False)
            else:
                positions = numpy.random.choice(range(len(available)), self.bandSize, replace=False)

                inds = [available[i] for i in positions]

                for position in sorted(positions, reverse = True):
                    del available[position]

            subsets.append(inds)

            if len(self.hash.shape)==1:
                keys=[tuple(self.hash[inds])]
            #hash values on newly created band
            else:
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
        self.bands = subsets  # indices corresponding to 'bands'


    def makeBands(self):
        subsets=[]
        for i in range(self.numBands):
            inds = numpy.random.choice(self.numHashes, self.bandSize, replace=False)
            subsets.append(inds)

        self.bands = subsets







    def findCandidates(self, ind):
        "find all indices agreeing in at least one band"

        if self.finder is None: #  construct finder if none exists
            self.makeFinder()

        candidates = []
        for i in range(len(self.bands)):
            band = self.bands[i]
            d = self.finder[i]

            if len(self.hash.shape)==1:
                key=tuple(self.hash[band])
            else:
                key = tuple(self.hash[ind, band])
            candidates = candidates + d.get(key)  # this is O(n) on gauss

        if self.keepStats:
            self.nbhdSizes.append(len(candidates)+1)
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
        print('making hash...')
        self.makeHash()
        print('made hash')

        print('making finder...')
        self.makeFinder()
        print('made finder')

        available = range(self.hash.shape[0])
        included = [True] * self.numObs
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


                included = [False]*self.numObs
                for i in available:
                    included[i] = True


            next = numpy.random.choice(available)
            sample.append(next)

            count = count + 1

            toRemove = self.findCandidates(next)
            for i in toRemove:
                included[i]=False

            available = list(itertools.compress(range(self.numObs), included))

            #possibly O(n)
            #available = [x for x in available if x not in toRemove]
            # log('{} remaining'.format(len(available)))

        if self.keepStats:
            if not reset:
                self.remnants = len(available)
            else:
                self.remnants = 0

            # guess of neighborhood size
            # print(self.numObs)
            # print(self.lastCounts)



            meanSize = sum(self.nbhdSizes) / len(self.nbhdSizes)

            self.guess = (float(self.numObs) / meanSize) * self.numBands
            self.actual = self.getMeanCounts()


            self.error = float(abs(self.guess - self.actual))/self.actual
            if self.error < 0:
                self.error = -1




        assert(len(sample)==sampleSize)
        return numpy.unique(sample)

    def optimize_param(self, param, N, inverted=False, step = 1, binary = True, max_iter = 20, verbose = True):
        if verbose:
            print('optimizing {}'.format(param))
        cur_val = getattr(self, param)

        subsample = self.downSample(N)
        counts = self.getMeanCounts()

        iter = 1
        low = None
        high = None
        while iter < max_iter:
            if counts > 0 and counts < N:
                # too low, increase value
                low = cur_val
                if high is None:
                    if not inverted:
                        cur_val *= 2.
                    else:
                        cur_val /= 2.
                else:
                    cur_val = (low + high) / 2.
                if verbose:
                    log('changing from {} to {}'.format(low, cur_val))

            elif counts < 0:
                # too high, decrease value
                high = cur_val
                if low is None:
                    if not inverted:
                        cur_val /= 2
                    else:
                        cur_val *= 2
                else:
                    cur_val = (low + high) / 2
                if verbose:
                    log('changing from {} to {}'.format(high, cur_val))

            elif counts == N:
                if verbose:
                    print('got it perfect')
                break
            setattr(self, param, cur_val)
            subsample = self.downSample(N)
            counts = self.getMeanCounts()
            if verbose:
                print('sampled {}'.format(counts))
            iter += 1

        setattr(self, param, cur_val)

        # if inverted:
        #     step = -1*step
        #
        # while counts > 0:
        #     cur_val = cur_val + step #increase
        #
        #     setattr(self, param, cur_val)
        #     subsample = self.downSample(N)
        #     counts = self.getMeanCounts() #  how many did you sample?
        #
        #     print(counts)
        #     print('current value: {}, adding step'.format(cur_val))
        #
        #
        # while(counts < 0):
        #     cur_val = cur_val - step
        #     setattr(self, param, cur_val)
        #     subsample = self.downSample(N)
        #     counts = self.getMeanCounts()
        #     print(counts)
        #     print('current value {}'.format(cur_val))
        #
        # cur_val = cur_val + step
        #
        # print('optimized value is {}, subtracting step'.format(cur_val))
        # setattr(self, param, cur_val)









#this didn't work
def fastDownsample(self, sampleSize=100, replace = False):
    self.makeHash()
    self.makeBands()

    sample= []
    available = range(self.numObs)
    reset = False
    count = 0
    t0=time()
    while len(sample) < sampleSize:
        if len(available) == 0:
            t1=time()
            reset = True

            log("sampled {} out of {} before reset".format(count, sampleSize))
            log("it took {} seconds".format(t1-t0))
            log("sampled {} total".format(len(sample)))
            self.lastCounts.append(count)

            self.takenBands = [[]]*self.numBands
            if replace:
                available = range(self.hash.shape[0])
            else:
                available = [x for x in range(self.hash.shape[0]) if x not in sample]
            count = 0
            t0=time()

        nextind = numpy.random.choice(available)
        discard = False
        for band in range(self.numBands):
            #print('band {}'.format(band))
            cur_band = self.bands[band]

            #print('cur band {}'.format(cur_band))

            #print('hash value {}'.format(self.hash[nextind, cur_band]))
            hashval = tuple(self.hash[nextind, cur_band])  # hashes in this band

            if hashval in self.takenBands[band]:
                #print('discarding')
                discard = True
                available = [x for x in available if x != nextind]

                break

        if not discard:  # bands of this point are taken
            sample.append(nextind)
            # print('appended')
            count = count + 1

            #remove from taken bands
            for band in range(self.numBands):
                hashval = tuple(self.hash[nextind, cur_band])
                self.takenBands[band].append(hashval)

    return sample
