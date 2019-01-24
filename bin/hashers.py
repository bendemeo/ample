from __future__ import division
from scipy.stats import ortho_group
from copy import deepcopy
import numpy as np
from LSH import *
from test_file import *
from utils import *
from time import time
import random


class angleSampler(sampler):
"""weights points by the mean angle with an axis"""

    def __init__(self, data, replace=False, strength = 1):
        #translate to first quadrant and normalize
        for i in range(data.shape[0]):
            data[i,:] -= data[i,:].min(0)
        data /= data.max()
        sampler.__init__(self, data, replace)
        self.strength = strength

    def downsample(self, sampleSize):
        wts = None * self.numObs
        for i in range(self.numObs):
            mag = sum([x^2 for x in self.data[i,:]])

            angles = [math.atan(float(x)/math.sqrt(mag - x^2))
                for x in self.data[i,:]]

            wts[i] = mean(angles)

        wts = [float(1) / (w ** self.strength) for w in wts]

        total = sum(wts)
        wts = [float(w)/total for w in wts]

        print(wts)

        return(np.random.choice(range(self.numObs,p=wts)))




class treeLSH(LSH):
    """rp-tree like hashing scheme"""


    def __init__(self, data, splitSize, children=2, minPoints = 0):
        numBands = 1
        bandSize = 1
        numHashes = 1

        LSH.__init__(self,data, numHashes=numHashes, numBands=numBands, bandSize=bandSize)

        self.data = data
        self.children = children
        self.splitSize = splitSize
        self.minPoints = minPoints

    @staticmethod
    def quantilate(vals, splitSize, children = 2):
        """converts arrays of values to which quantile division they belong to"""
        diam = max(vals) - min(vals)


        # print('vals {}'.format(vals))
        # print('diameter {}'.format(diam))
        if diam < splitSize:
            return([0]*len(vals))

        #print('diameter is {}'.format(diam))
        splits = min(np.ceil(diam / float(splitSize)), children)

        return pd.qcut(vals, int(splits), labels=False)


    @staticmethod
    def dimHash(data, splitSize, children, max_splits):
        #print('shape is {}'.format(data.shape))

        #print('data has size {}'.format(data.shape[0]))
        hashes = np.empty([data.shape[0],1])
        hashes[:,0] = np.array(treeLSH.quantilate(data[:,0], splitSize, max_splits = max_splits))

        if data.shape[1] == 1:
            return hashes
        if data.shape[0] == 1:
            hashes = np.empty(data.shape)
            hashes [0,:] = [0]*data.shape[1]
            return hashes

        result = np.empty(data.shape)
        result[:,0]= hashes[:,0]

        if len(np.unique(hashes)) > 1:
            print('splitting into {}'.format(len(np.unique(hashes))))
        #print('there are {} splits'.format(len(np.unique(hashes))))
        for val in np.unique(hashes):
            inds = [i for i in range(len(hashes)) if hashes[i] == val]
            #print('recursing')
            subframe = treeLSH.dimHash(data[inds, 1:], splitSize, children, max_splits)

            result[inds, 1:]=subframe
        #print(result)
        return(result)


    def makeHash(self):
        result = np.empty((self.numObs, 1))
        # #fill first column of table
        # table[:,0] = treeLSH.quantilate(self.data[:,0], self.splitSize, self.children)

        cur_dict = {}
        cur_dict[tuple([])] = range(self.numObs) #start: everything in empty square
        for i in range(self.data.shape[1]):
            print('dealing with dimension {}'.format(i))
            new_dict = {}
            for k in cur_dict.keys():
                #print('partition {}'.format(k))
                inds = cur_dict[k] # which indices have this signature
                if len(inds) <= self.minPoints:
                    #don't partition
                    new_dict[k] = inds
                    continue

                new_keys = treeLSH.quantilate(self.data[inds,i], self.splitSize, self.children)
                if len(np.unique(new_keys)) == 1:
                    new_dict[k] = inds
                    continue #no partitioning to do, so don't bother
                for nk in np.unique(new_keys):
                    new_dict[k + tuple([nk])] = [inds[j] for j in range(len(inds)) if new_keys[j] == nk]
            cur_dict = new_dict
            print('updated dict has length {}'.format(len(cur_dict)))


        self.occSquares = len(cur_dict)

        keys = list(cur_dict.keys())
        for square in range(len(keys)):
            for idx in cur_dict[keys[square]]:
                result[idx,0] = square

        self.hash = result
            # for q in np.unique(table[:,i]):
            #     cur_dict[tuple([q])] = [i for i in range(self.numObs) if table[i,0] == q]



    # def makeHash(self):
    #     hash =  treeLSH.dimHash(self.data, self.splitSize, max_splits = self.max_splits, children = self.children)
    #
    #     result = np.empty((self.numObs, 1))
    #
    #     hash_dict = {} # make a grid from it
    #
    #     for i in range(self.numObs):
    #         k = tuple(hash[i,:].astype(int))
    #
    #         if k in hash_dict:
    #             hash_dict[k].append(i)
    #         else:
    #             hash_dict[k] = [i]
    #
    #     keys = list(hash_dict.keys())
    #     for square in range(len(keys)):
    #         for idx in hash_dict[keys[square]]:
    #             result[idx,0] = square
    #     #
    #     # result = np.empty([num_obs,1])
    #     #
    #     # for i in range(num_obs):
    #     #     result[i,0] = tuple(result[i,:])
    #     #
    #
    #     #result = np.array([tuple(x) for x in result])
    #     #print(result)
    #
    #     self.hash = result
    #     """ assumes dimensions are sorted by variance, ala PCA"""
    #
    #     self.data[1,:]



class gridLSH(LSH):
    """just bins coordinates to make an orthogonal grid"""

    def __init__(self,data,gridSize, replace=False, target=10, randomize_origin = True, cell_labels=None, cluster_labels=None, record_counts=False):
        numBands = 1
        bandSize = 1
        numHashes = 1
        LSH.__init__(self,data, numHashes=numHashes, numBands=numBands, bandSize=bandSize,
        replace=replace, target=target)
        self.gridSize=gridSize
        self.randomize_origin= randomize_origin
        self.cell_labels = cell_labels
        self.cluster_labels = cluster_labels
        self.record_counts = record_counts
        self.occSquares = None

    def makeHash(self):

        #make positive and max 1
        X = self.data - self.data.min(0)
        X /= X.max()

        #hashes = np.empty((self.numObs,self.numFeatures))
        hashes = np.empty((self.numObs, 1))

        grid = {}

        if(self.randomize_origin):
            for i in range(self.numFeatures):
                #print(X[:,i])
                #determine how much you can shift without altering no. grid squares
                shift_min = (-1)*X[:,i].min()
                shift_max = (self.gridSize * (np.floor(X[:,i].max()/float(self.gridSize))) + 1) - X[:,i].max()
                # shift_min = X[:,i].max() - (self.gridSize * (np.floor(X[:,i].max()/float(self.gridSize))) + 1)

                # print('data min: {}'.format(X[:,i].min()))
                # print('data max: {}'.format(X[:,i].max()))
                # print('min: {}'.format(shift_min))
                # print('max: {}'.format(shift_max))


                shift = random.uniform(shift_min, shift_max)

                # print('shift: {}'.format(shift))
                for j in range(self.numObs):
                    X[j,i] += shift

                # print('new data min: {}'.format(X[:,i].min()))
                # print('new data max: {}'.format(X[:,i].max()))
                # print('data: {}'.format(X[:,i]))


        #make dict mapping grid squares to points in it
        for i in range(self.numObs):
            coords = X[i,:]
            # if(randomize_origin): #shift data by random vector
            #     for x in coords:
            #         x += random.uniform(0,self.gridSize)


            gridsquare = tuple(np.floor(coords / float(self.gridSize)).astype(int))

            if gridsquare not in grid:
                grid[gridsquare]=set()
                # print(gridsquare)
                # print(coords)
                # print(i)
            grid[gridsquare].add(i)





        self.occSquares = len(grid)
        if self.record_counts:
            cluster_labels = self.cluster_labels
            counts = {}
            scores = {}
            labels = sorted(set(self.cluster_labels))
            print('labels: {}'.format(labels))

            for lab in labels:
                #print(len(grid.values()))
                #print(grid.values())
                counts=[len([i  for i in square if self.cluster_labels[i] == lab]) for square in grid.values()]
                #print('counts: {}'.format(counts))
                counts = [count for count in counts if count > 0]

                #print('counts: {}'.format(counts))
                #normalize to percentages
                counts = [float(count)/sum(counts) for count in counts]
                #print('counts: {}'.format(counts))

                counts.sort(reverse=True)
                pct_covered = [sum(counts[:(i+1)]) for i in range(len(counts))]
                print(pct_covered)
                good_inds = [i for i in range(len(pct_covered)) if pct_covered[i]>0.5]
                score = min(good_inds)

                # score = sum([count for count in counts])/len(counts)
                #score = max(counts)
                scores[lab] = score
            self.clustScores = scores
            print(scores)
        #enumerate grid squares, and assign each obs to its square index
        keys = list(grid.keys())
        for square in range(len(keys)):
            for idx in grid[keys[square]]:
                hashes[idx,0] = square


        self.hash=hashes



class multiLSH(LSH):
    def __init__(self, components):
        self.components = components

        # get data from a representative
        rep = components[0]
        self.data = rep.data
        self.target = rep.target
        self.numObs, self.numFeatures = self.data.shape
        self.keepStats = rep.keepStats
        self.nbhdSizes = []


    def makeHash(self):
        for lsh in self.components:
            lsh.makeHash()

    def makeFinder(self):
        for lsh in self.components:
                lsh.makeFinder()



class sumLSH(multiLSH):
    def __init__(self, summands):
        multiLSH.__init__(self, summands)

    def findCandidates(self, ind): #OR of all summand candidates
        candidates = []
        for lsh in self.components:
            candidates.append(lsh.findCandidates(ind))

        candidates = set().union(*candidates)

        if self.keepStats:
            self.nbhdSizes.append(len(candidates)+1)
        return candidates


class prodLSH(multiLSH):
    def __init__(self, factors):
        multiLSH.__init__(self, factors)

    def findCandidates(self, ind): #AND of all product candidates
        candidates = []
        for lsh in self.components:
            candidates.append(lsh.findCandidates(ind))

        candidates = set().intersection(*candidates)

        if self.keepStats:
            self.nbhdSizes.append(len(candidates)+1)

        return candidates



class gsLSH(LSH):
    def __init__(self, data, target='auto', gridSize=None, replace = False, alpha=0.1,
    max_iter = 200, verbose = True, opt_grid=True):

        LSH.__init__(self, data, numHashes=1, numBands=1, bandSize=1, replace=replace,
        target=target)


        self.gridSize=gridSize #starting grid size before optimization

        self.alpha = alpha
        #self.target = target # downsampling size you're built for
        self.verbose = verbose
        self.max_iter = max_iter
        self.opt_grid = opt_grid
        self.occSquares = None

        print('target: {}'.format(self.target))

    def makeHash(self): #re-implementation of gs, formulated as an LSH
        n_samples, n_features = self.data.shape

        print('target is {} at time of hashing'.format(self.target))

        X = self.data - self.data.min(0)
        X -= X.max()

        hashes = np.empty((self.numObs, 1))  # table of grid square assignments

        X_ptp = X.ptp(0)  # list of ranges for each feature

        low_unit, high_unit = 0., max(X_ptp)

        if self.gridSize is None:
            self.gridSize=(low_unit + high_unit) / 4

        unit = self.gridSize

        n_iter = 0
        while True:
            if self.verbose:
                log('n_iter = {}'.format(n_iter))

            grid_table = np.zeros((n_samples, n_features))

            for d in range(n_features):
                if X_ptp[d] <= unit:
                    # entire range fits in a grid square
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

            if self.verbose:
                log('found {} non-empty grid cells'.format(len(grid)))


            if not self.opt_grid:
                break

            if len(grid) > self.target * (1 + self.alpha):
                #too many grid cells
                low_unit = unit
                if high_unit is None:
                    unit *= 2
                else:
                    unit = (unit + high_unit) / 2.

                if self.verbose:
                    log('Grid size {}, increase unit to {}'
                        .format(len(grid), unit))

            elif len(grid) < self.target / (1 + self.alpha):
                # Too few grid cells, decrease unit.
                high_unit = unit
                if low_unit is None:
                    unit /= 2.
                else:
                    unit = (unit + low_unit) / 2.

                if self.verbose:
                    log('Grid size {}, decrease unit to {}'
                        .format(len(grid), unit))

            else:
                break

            if high_unit is not None and low_unit is not None and \
               high_unit - low_unit < 1e-20:
                break

            if n_iter >= self.max_iter:
                # Should rarely get here.
                print('WARNING: Max iterations reached, try increasing '
                                 ' alpha parameter.\n')
                break
            n_iter += 1



        self.occSquares = len(grid)

        #enumerate grid squares, and assign each obs to its square index
        keys = list(grid.keys())
        for square in range(len(keys)):
            for idx in grid[keys[square]]:
                hashes[idx,0] = square


        self.hash=hashes
        self.gridLabels = [int(x) for y in list(hashes) for x in y]
        self.gridSize=unit

    def downsample_weighted(self, sampleSize, alpha=1, replace=False):
        print('alpha is {}'.format(alpha))

        if self.target == 'N':
            self.target = sampleSize
        self.makeHash()
        self.makeFinder()



        grid = self.finder[0]
        full_grid = grid # keeps all squares

        sizes = {square:len(v) for (square,v) in grid.items()} #sizes of grid squares
        #print('square sizes: {}'.format(sizes))

        weights = [1/(np.power(size, alpha)) for size in sizes.values()]
        total = sum(weights)

        weights = [float(w)/total for w in weights] #make them sum to 1
        assert(np.abs(sum(weights) - 1) < .000001)

        available = range(self.numObs)
        included = [True] * self.numObs # all indices available
        square_sampled = [True] * len(grid) # whether square has been sampled
        avail_square = range(len(grid)) # which squares available

        sample = []
        valid_sample=[True] * self.numObs #true if hasn't been sampled
        sample_inds = [] # indices relative to available samples
        count = 0 # how many have been added since reset
        reset = False  # whether we have reset

        if self.keepStats:
            self.lastCounts=[]

        subinds = []

        for n in range(sampleSize):
            grid_cells = list(grid.keys())

            grid_cell = grid_cells[np.random.choice(len(grid_cells),p=weights)]

            samples = list(grid[grid_cell])
            sample = samples[np.random.choice(len(samples))]

            # del grid[grid_cell]

            if not replace:
                #print(grid[grid_cell])
                grid[grid_cell].remove(sample)
                #print(grid[grid_cell])
                sizes[grid_cell] -= 1
                if len(grid[grid_cell]) == 0:
                    del grid[grid_cell]
                    del sizes[grid_cell]


                #update weights
                weights = [1/(np.power(size,alpha)) for size in sizes.values()]

                total = sum(weights)
                weights = [float(w)/total for w in weights] #make them sum to 1
                assert(np.abs(sum(weights) - 1) < .000001)

            #print('appending {}'.format(sample))
            subinds.append(sample)
            #print('samples is now {}'.format(subinds))

        #print('samples: {}'.format(samples))
        return(sorted(subinds))


        # while True:
        #     if len(available) == 0:  # reset available if not enough
        #         reset = True
        #
        #         log("sampled {} out of {} before reset".format(count, sampleSize))
        #         if(self.keepStats):
        #             self.lastCounts.append(count)
        #
        #         if sampleSize == 'auto': #stop sampling when you run out
        #             break
        #
        #
        #         count = 0
        #         if replace:
        #             available = range(self.numObs)
        #         else:
        #             available = list(itertools.compress(range(self.numObs), valid_sample))
        #             #print('available: {}'.format(available))
        #             #available = [x for x in range(self.numObs) if x not in sample]
        #
        #
        #         #reset included so only available indices are true
        #         included = [False]*self.numObs
        #         for i in available:
        #             included[i] = True
        #
        #
        #     next = numpy.random.choice(available)
        #     sample.append(next)
        #     valid_sample[next] = False
        #
        #     if (sampleSize != 'auto') and (len(sample) >= sampleSize):
        #         break
        #
        #     count = count + 1
        #
        #     toRemove = self.findCandidates(next)
        #     for i in toRemove:
        #         included[i]=False
        #
        #     available = list(itertools.compress(range(self.numObs), included))
        #
        #











class cosineLSH(LSH):

    def makeHash(self):
        projector = np.random.randn(self.numFeatures, self.numHashes)
        projection = (np.sign(np.matmul(self.data,projector))+1)/2
        self.hash = projection

class projLSH(LSH):

    def __init__(self, data, numHashes, numBands, bandSize, gridSize=0.1, replace=False, target=10):

        LSH.__init__(self, data, numHashes=numHashes, numBands=numBands, bandSize=bandSize, replace=replace, target=target)

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
    def __init__(self, data, gridSize, numHashes, numBands, bandSize, replace = False, target=10):

        LSH.__init__(self,data, numHashes=numHashes, numBands=numBands, bandSize=bandSize,
        replace=replace, target=target)

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

#how to add two hashers
#can't overload operator due to weird dependencies?
def plus(*args):
    result = sumLSH(args)
    return(result)
        #
        # grid_axes = ortho_group.rvs(self.numFeatures)
        #
        # for i in range(grid_axes.shape[0]):
        #     # project onto ith grid axis
        #     projection = np.matmul

def times(*args):
    result = prodLSH(args)
    return(result)

if __name__ == '__main__':
    gauss2D = gauss_test([10,20,100,200], 2, 4, [0.1, 1, 0.01, 2])
    mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])

    # downsampler = gridLSH(gauss2D,0.1)
    downsampler = projLSH(gauss2D, 10, 2, 5, 0.05)

    subInds = downsampler.downSample(50)

    print(subInds)

    mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')
    mpl.show()
