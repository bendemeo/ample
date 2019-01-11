from __future__ import division
from scipy.stats import ortho_group
import numpy as np
from LSH import *
from test_file import *
from utils import *
from time import time



class gridLSH(LSH):
    """just bins coordinates to make an orthogonal grid"""

    def __init__(self,data,gridSize, replace=False, target=10):
        numBands = 1
        bandSize = 1
        numHashes = 1
        LSH.__init__(self,data, numHashes=numHashes, numBands=numBands, bandSize=bandSize,
        replace=replace, target=target)
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
    def __init__(self, data, target=10, gridSize=None, replace = False, alpha=0.1,
    max_iter = 200, verbose = True, opt_grid=True):

        LSH.__init__(self, data, numHashes=1, numBands=1, bandSize=1, replace=replace,
        target=target)

        self.gridSize=gridSize #starting grid size before optimization

        self.alpha = alpha
        self.target = target # downsampling size you're built for
        self.verbose = verbose
        self.max_iter = max_iter
        self.opt_grid = opt_grid

    def makeHash(self): #re-implementation of gs, formulated as an LSH
        n_samples, n_features = self.data.shape

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
                sys.stderr.write('WARNING: Max iterations reached, try increasing '
                                 ' alpha parameter.\n')
                break
            n_iter += 1



        #enumerate grid squares, and assign each obs to its square index
        keys = list(grid.keys())
        for square in range(len(keys)):
            for idx in grid[keys[square]]:
                hashes[idx,0] = square


        self.hash=hashes
        self.gridSize=unit

    def downsample_weighted(self, sampleSize, alpha=1, replace=False):
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
