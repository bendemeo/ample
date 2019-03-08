from __future__ import division
from scipy.stats import norm, rankdata, ortho_group
from scipy.spatial.distance import squareform
from copy import deepcopy
import numpy as np
from LSH import *
#from test_file import *
from utils import *
from time import time
from sampler import *
import random
import sklearn as sk
from sklearn import manifold
from fbpca import pca
import pandas as pd

class trieNode:
    def __init__(self, children, val='START', parent = None):
        self.parent = parent
        self.val = val
        self.children = children # dict of value:trieNode

    def getParent(self):
        return(self.parent)

    def getValue(self):
        return(self.val)

    def getChildren(self):
        return(self.chidren)

    def tostr(self):
        result = str(self.val)
        for val in self.children.keys():
            result += ", ["
            if self.children[val] == '_end_':
                result += 'END'
            else:
                result += self.children[val].tostr()
            result += ']'
        return(result)

class gridTrie:
    def __init__(self, squares):
        root = trieNode(children = {})

        _end = '_end_'
        for square in squares:
            current_node = root
            for coord in square:
                defaultTrie = trieNode(children={}, parent=current_node, val = coord)

                current_node = current_node.children.setdefault(coord, defaultTrie)
            current_node.children[_end] = _end

        self.trie = root

    def removeNeighbors(self, pos):

        t0 = time()
        current_nodes = [self.trie]
        current_squares = [()] #keep track of paths

        # all_dicts = [] # all dicts at all time points, O(number of neighbors) avg space
        # parent_dicts = [] # parent dicts of all_dicts elements
        #

        for coord in pos:
             new_nodes = []
             new_squares = []

             coords = []


             last_pairs = []
             del_pairs = [] # pairs of (dict, coord) to remove

             for i, d in enumerate(current_nodes):
                 if coord in d.children:

                     new_nodes.append(d.children.get(coord))
                     #parent_dicts.append(d)
                     new_squares.append(current_squares[i]+tuple([coord]))

                 if (coord-1) in d.children:
                     new_nodes.append(d.children.get(coord-1))
                     #parent_dicts.append(d)
                     new_squares.append(current_squares[i]+tuple([coord-1]))

             if len(new_nodes)==0:
                 return(None)
             #all_dicts.append(new_dicts)
             current_nodes = new_nodes
             current_squares = new_squares
        t1 = time()
        print('found {} neighbors in {} seconds'.format(len(current_squares), t1-t0))
        #print([x.parent.tostr() for x in current_nodes])

        ## delete neighbors from tree

        t0 = time()
        current_level = set(current_nodes)
        next_level = current_level

        while len(next_level) > 0:
            next_level = []
            #del_nodes = []
            for node in current_level:
                if node.parent == None:
                    # root node
                    break
                if len(node.parent.children) == 1:
                    next_level.append(node.parent)
                # print(node.parent.children)
                # print(node.val)
                del node.parent.children[node.val]

            current_level = next_level

        t1 = time()
        print('removed them in {} seconds'.format(t1-t0))
                #
                # if len(node.parent.children) == 0:
                #     del_nodes.append(node)
                #     print(node.val)
                #     print(node.parent.children.keys())
                #     del node.parent.children[node.val]

            #current_level = {n.parent for n in del_nodes}


        return(current_squares)


# class gridTrie:
#     def __init__(self, squares):
#         #data structure to allow fast grid neighbor finding
#         ## adapted from https://stackoverflow.com/questions/11015320/how-to-create-a-trie-in-python
#         _end = '_end_'
#         root = dict()
#         for square in squares:
#             current_dict = root
#             for coord in square:
#                 current_dict = current_dict.setdefault(coord, {})
#             current_dict[_end]=_end
#
#
#         self.trie = root #always stores entire trie
#         self.curTrie = root #only non-sampled nodes
#
#
#
#
#     def neighbors(self, pos):
#         # given a grid square, find all occupied squares at most 1 to the left of it
#         #descends grid trie, keeping x and x-1 for each coordinate
#
#         current_dicts = [self.trie]
#         current_squares = [()]
#         all_dicts = [] # all dicts at all time points, O(number of neighbors) avg space
#         parent_dicts = [] # parent dicts of all_dicts elements
#
#
#         for coord in pos:
#              new_dicts = []
#              new_squares = []
#
#
#
#              coords = []
#
#
#              last_pairs = []
#              del_pairs = [] # pairs of (dict, coord) to remove
#
#              for i, d in enumerate(current_dicts):
#                  if coord in d:
#                      new_dicts.append(d.get(coord))
#                      parent_dicts.append(d)
#                      new_squares.append(current_squares[i]+tuple([coord]))
#
#                  if (coord-1) in d:
#                      new_dicts.append(d.get(coord-1))
#                      parent_dicts.append(d)
#                      new_squares.append(current_squares[i]+tuple([coord-1]))
#
#              if len(new_dicts)==0:
#                  return(None)
#              all_dicts.append(new_dicts)
#
#              current_dicts = new_dicts
#              current_squares = new_squares
#         print('found {} neighbors'.format(len(current_squares)))
#         return(current_squares)




if __name__ == '__main__':

    tuples = []
    tuple_len=100
    tuple_max=3
    N=500000
    for i in range(N):
        tuples.append(tuple(np.random.choice(tuple_max, tuple_len)))

    trie = gridTrie(tuples)
    print(trie.trie.tostr())


    randTuple = tuples[np.random.choice(len(tuples))]
    print(randTuple)
    t0 = time()
    print(trie.removeNeighbors(randTuple))
    t1 = time()
    print('it took {} seconds'.format(t1-t0))
    print(trie.trie.tostr())



class softGridSampler(sampler):


    def __init__(self, data, alpha=0.1, gridSize=0.3, opt_grid=False, max_iter=200):
        sampler.__init__(self, data)
        self.gridSize=gridSize
        print('square size: {}'.format(self.gridSize))

        grid = {}

        for sample_idx in range(self.numObs):
            sample = self.data[sample_idx,:]
            grid_cell = tuple(np.floor(sample / self.gridSize).astype(int))

            if grid_cell not in grid:
                grid[grid_cell] = set()
            grid[grid_cell].add(sample_idx)

        self.grid = grid
        print('grid size is {}'.format(len(grid)))
        t0 = time()
        self.trie = gridTrie(grid.keys()) #for fast neighbor computation
        self.curTrie = gridTrie(grid.keys()) # updated as neighbors are removed
        t1 = time()
        print('initialized trie in {} seconds'.format(t1-t0))


    def findCandidates(self, idx):
        "find all points from neighboring squares at nearest junction"

        candidates = []
        sample = self.data[idx, :]
        grid_cell = tuple(np.floor(sample / self.gridSize).astype(int))




        grid_shifts = [(x % self.gridSize > (0.5 * self.gridSize))
                       for x in sample]
        #print(grid_shifts)
        #represents nearest grid intersection
        grid_intersect = [sum(x)for x in zip(grid_cell, grid_shifts)]

        #print(self.curTrie.trie.tostr())
        neighborsquares = self.curTrie.removeNeighbors(grid_intersect)
        #print(neighborsquares)
        candidates = []
        for square in neighborsquares:
            candidates = candidates + list(self.grid[square])
        # grid_shifts = [2 * x - 1 for x in grid_shifts]
        # #print(grid_shifts)
        #
        # for square in list(self.grid.keys()):
        #     neighbor = True
        #     for i, coord in enumerate(square):
        #         if (coord == grid_cell[i]) or (coord == grid_cell[i] + grid_shifts[i]):
        #             continue
        #         else:
        #             neighbor = False
        #             break
        #
        #     if(neighbor):
        #         candidates = candidates + list(self.grid[square])


        return(candidates)





    def downsample(self, sampleSize='auto'):
        available = range(self.numObs)
        included = [True] * self.numObs # all indices available
        sample = []
        valid_sample=[True] * self.numObs #true if hasn't been sampled
        sample_inds = [] # indices relative to available samples
        count = 0 # how many have been added since reset
        reset = False  # whether we have reset

        self.lastCounts = []

        while True:
            if len(available) == 0:  # reset available if not enough
                reset = True

                log("sampled {} out of {} before reset".format(count, sampleSize))
                self.lastCounts.append(count)

                if sampleSize == 'auto': #stop sampling when you run out
                    break


                count = 0

                available = list(itertools.compress(range(self.numObs), valid_sample))
                    #print('available: {}'.format(available))
                    #available = [x for x in range(self.numObs) if x not in sample]

                self.curTrie = self.trie
                #reset included so only available indices are true
                included = [False]*self.numObs
                for i in available:
                    included[i] = True
            #
            # print('available left')
            # print(len(available))
            next = numpy.random.choice(available)
            sample.append(next)
            valid_sample[next] = False

            if (sampleSize != 'auto') and (len(sample) >= sampleSize):
                break

            count = count + 1

            toRemove = self.findCandidates(next)
            for i in toRemove:
                included[i]=False

            available = list(itertools.compress(range(self.numObs), included))


        if not reset:
            self.remnants = len(available)
        else:
            self.remnants = 0

        self.sample = sorted(numpy.unique(sample))
        self.curTrie = deepcopy(self.trie) # done sampling; reset curTrie
        return(self.sample)


class multiscaleSampler(weightedSampler):
    def __init__(self, data, scales=[0.1]):
        weightedSampler.__init__(self, data)

        #self.tests = tests
        self.scales = scales


    def scaleWeights(self, scale):
        X = self.data - self.data.min(0)
        X /= X.max()

        grid = {}

        neighborCounts = [0]*self.numObs #  how many neighbors in grid square
        wts = [0]*self.numObs

        for i in range(self.numObs):
            coords = X[i, :]
            gridsquare = tuple(
                np.floor(coords / float(scale)).astype(int))

            if gridsquare not in grid:
                grid[gridsquare] = set()

            grid[gridsquare].add(i)

        numSquares = len(grid)
        for square in grid:
            size = len(grid[square])
            for i in grid[square]:
                wts[i] = 1./(numSquares * size)

        return(wts)


    def makeWeights(self):
        wtTable = np.empty([self.numObs, len(self.scales)])

        for i,s in enumerate(self.scales):
            print('trying scale {}'.format(s))
            wtTable[:,i] = self.scaleWeights(s)

        print(wtTable)
        newWts = np.mean(wtTable, axis=1).tolist()
        print(newWts)

        print(sum(newWts))
        self.wts = newWts


class sigSampler(sampler):

    def __init__(self, data, bins=4, **kwargs):
        sampler.__init__(self, data)

        self.bins = bins
        self.sample = None
        self.available = list(range(self.numObs))

    def sigTransform(self, bins=4):
        transformed = np.empty(self.data.shape)
        # convert data to bins, ala cut_width
        for i in range(self.numFeatures):
            binVals = pd.cut(self.data[:,i], self.bins,
                             labels=False).tolist()
            proportions = []
            for k in range(self.bins):
                p = float(sum(1 for x in binVals if x == k)) / len(self.available)
                proportions.append(p)
            # print(proportions)
            # print(sum(proportions))
            binProps = [proportions[v] for v in binVals]
            expected = 1./self.bins

            scores = [expected / x for x in binProps]
            transformed[:,i] = scores

        self.data = transformed
        return (transformed)


    def downsample(self, sampleSize):
        self.available = list(range(self.numObs))
        sample = []

        while(len(sample) < sampleSize):

            binData = np.empty(self.data.shape)
            # convert data to bins, ala cut_width
            for i in range(self.numFeatures):
                binVals = pd.cut(self.data[self.available,i], self.bins,
                                 labels=False).tolist()
                proportions = []
                for k in range(self.bins):
                    p = float(sum(1 for x in binVals if x == k)) / len(self.available)
                    proportions.append(p)
                # print(proportions)
                # print(sum(proportions))
                binProps = [proportions[v] for v in binVals]
                expected = 1./self.bins

                scores = [expected / x for x in binProps]




                newInd = binProps.index(min(binProps))
                sample.append(self.available[newInd])
                del self.available[newInd]

        self.sample = sample
        return(sample)


class detSampler(seqSampler):
    """adds point which makes determinant of kernel matrix better"""

    def __init__(self, data, batch=100, replace=False):
        seqSampler.__init__(self, data, replace)
        self.batch = batch
        self.kernel = None
        self.sample = []

    def addSample(self, viz=False, file=None, **kwargs):
        if len(self.sample) == 0:
            new = np.random.choice(self.numObs)
            self.sample.append(new)
            self.kernel = np.matrix(np.matmul(self.data[self.sample, :],
                                              np.transpose(self.data[self.sample, :])))
            return(new)
        else:
            # self.normalized = sk.preprocessing.normalize(self.data, axis=1)
            size = min([self.batch, len(self.avail)])  # how many to check
            candidates = np.random.choice(self.avail, size, replace=False)
            dets = []
            # kernel = np.matrix(np.matmul(self.data[self.sample,:],
            #                    np.transpose(self.data[self.sample,:])))

            for c in candidates:
                newrow = np.matmul(self.data[c, :], np.transpose(
                    self.data[self.sample, :]))

                newcol = list(
                    newrow) + [np.matmul(self.data[c, :], np.transpose(self.data[c, :]))]
                # print(newcol)
                # print(newrow)
                # print(kernel)

                newKernel = np.vstack([self.kernel, np.matrix(newrow)])

                # print(newKernel)
                # print(np.matrix(np.transpose(newcol)))
                newKernel = np.hstack(
                    [newKernel, np.transpose(np.matrix(newcol))])

                t0 = time()
                dets.append(np.linalg.det(newKernel))
                t1 = time()

                # k=min([subset, self.numFeatures])
                # U,s,vt = pca(self.data[subset,:], k=k)
                # print(s)
                # dets.append(np.prod(s))

            #print('det size {} took {}'.format(len(self.sample), t1 - t0))
            # print(self.kernel.shape)

            ind = dets.index(max(dets))
            self.det = max(dets)
            new = candidates[ind]

            newrow = np.matmul(self.data[new, :], np.transpose(
                self.data[self.sample, :]))
            newcol = list(
                newrow) + [np.matmul(self.data[new, :], np.transpose(self.data[new, :]))]
            newKernel = np.vstack([self.kernel, np.matrix(newrow)])
            newKernel = np.hstack([newKernel, np.transpose(np.matrix(newcol))])
            self.kernel = newKernel

            del self.avail[ind]

            self.sample.append(new)
            return(new)


class dpp(sampler):
    """uses an MCMC framework to emulate a DPP"""

    def __init__(self, data, steps=100, normalize=False, **kwargs):
        sampler.__init__(self, data, **kwargs)
        self.sample = None
        self.sampled = [False] * self.numObs
        self.available = range(self.numObs)
        self.steps = steps
        self.det = None
        if(normalize):
            self.normalize()

    def sigTransform(self, bins=4):
        transformed = np.empty(self.data.shape)
        # convert data to bins, ala cut_width
        for i in range(self.numFeatures):
            binVals = pd.cut(self.data[:,i], bins,
                             labels=False).tolist()
            proportions = []
            for k in range(bins):
                p = float(sum(1 for x in binVals if x == k)) / self.numObs
                proportions.append(p)
            # print(proportions)
            # print(sum(proportions))
            binProps = [proportions[v] for v in binVals]
            expected = 1./bins

            scores = [expected / x for x in binProps]
            transformed[:,i] = scores

        self.data = transformed
        return (transformed)

    def step(self):
        c = np.random.choice(self.available, 1)  # new candidate
        s = np.random.choice(self.sample, 1)  # sample to switch out

        # compute kernel matrix if s is swapped for c
        newKernel = deepcopy(self.kernel)
        newSample = deepcopy(self.sample)

        i = self.sample.index(s)
        del newSample[i]
        newKernel = np.delete(newKernel, i, 0)
        newKernel = np.delete(newKernel, i, 1)

        #print(newSample)
        newrow = np.matmul(self.data[c, :], np.transpose(
            self.data[newSample, :]))


        newrow = [x for y in newrow for x in y]
        # print(newrow)

        newcol = newrow + [np.matmul(self.data[c, :], np.transpose(self.data[c, :]))]

        #print(newcol)
        newKernel = np.vstack([newKernel, np.matrix(newrow)])

        # print(np.transpose(np.matrix(newcol)).shape)
        # print(newKernel.shape)
        newKernel = np.hstack(
            [newKernel, np.transpose(np.matrix(np.array(newcol)))])
        newKernel = np.matrix(newKernel, dtype='float')

        newSample = newSample + c.tolist()
        newDet = np.linalg.det(newKernel)

        if newDet > self.det:
            self.sample = newSample
            self.kernel = newKernel
            self.det = newDet
            #print(newDet)

    def downsample(self,sampleSize):
        self.available = range(self.numObs)  # reset

        self.sample = np.random.choice(
            range(self.numObs), sampleSize, replace=False).tolist()

        self.kernel = np.matrix(np.matmul(self.data[self.sample, :],
                                          np.transpose(self.data[self.sample, :])),
                                dtype='float')

        self.det = np.linalg.det(self.kernel)

        for s in self.sample:
            self.sampled[s] = True

        for n in range(self.steps):
            print('sampling {}: step {} / {}'.format(sampleSize, n, self.steps))
            self.step()

        return(self.sample)


class densitySampler(weightedSampler):
    def makeWeights(self):
        dists = sk.metrics.pairwise.euclidean_distances(self.data)
        probs = sk.manifold.t_sne._joint_probabilities(dists, 10, True)
        probs = squareform(probs)
        print(probs)
        wts = np.sum(np.multiply(dists, probs), axis = 1)
        total = sum(wts)
        wts = [x/total for x in wts]
        print(wts)
        print(sum(wts))
        self.wts = wts


class centerSampler(sampler):

    def __init__(self, data, numCenters=10, steps=1000, normalize=False, transformed=False, weighted=False, spherical = False, **kwargs):
        sampler.__init__(self, data, **kwargs)
        self.numCenters = numCenters
        self.steps = steps
        self.sample = None
        self.centers = None
        self.available = list(range(self.numObs))
        self.normalize = normalize
        self.transformed = transformed
        self.weighted = weighted
        self.spherical = spherical
        #self.data -= self.data.min(0) # put in first quadrant

    def downsample(self, sampleSize):
        self.sample = []
        self.available = list(range(self.numObs))
        centerFinder = dpp(self.data, self.steps, normalize=self.normalize)
        if(self.transformed):
            centerFinder.sigTransform()
        self.centers = centerFinder.downsample(self.numCenters)


        # stores each point's distance to each center
        distTable = np.empty([self.numObs, self.numCenters])

        for i, c in enumerate(self.centers):
            dists = []
            for j in range(self.numObs):
                dists.append(np.linalg.norm(self.data[j, :]
                                            - self.data[c, :]))
            distTable[:, i] = dists


        # if(self.weighted):
        #     k = centerFinder.kernel
        #     print('printing kernel')
        #     print(k)
        #     cosines = np.empty(k.shape)
        #
        #     # construct cos(theta) matrix
        #     for i in range(len(self.centers)):
        #         for j in range(len(self.centers)):
        #             s1 = self.centers[i]
        #             s2 = self.centers[j]
        #             cosines[i,j] = float(k[i,j]) / (np.linalg.norm(self.data[s1,:]) * np.linalg.norm(self.data[s2,:]))
        #     print('printing cosines')
        #     print(cosines)
        #     weights = np.sum(cosines, axis=0)
        #     weights = [float(1)/x for x in weights]
        #     # weights = [x**3 for x in weights]
        #     total = sum(weights)
        #     weights = [x/total for x in weights]
        #     print(weights)
        #     print(sum(weights))
        #     self.weights=weights
        if(self.weighted):
            dists = sk.metrics.pairwise.cosine_distances(self.data[self.centers,:])
            probs = sk.manifold.t_sne._joint_probabilities(dists, 10, True)
            probs = squareform(probs)
            print(probs)
            wts = np.sum(np.multiply(dists, probs), axis = 1)
            total = sum(wts)
            wts = [x/total for x in wts]
            print(wts)
            print(sum(wts))
            self.weights = wts




            #
            # cDists = distTable[self.centers,:] #pairwise center dists
            # probs = sk.manifold.t_sne._joint_probabilities(cDists, 10, True)
            # probs = squareform(probs)
            # print(probs)
            # wts = np.sum(np.multiply(cDists, probs), axis = 1)
            # total = sum(wts)
            # wts = [x/total for x in wts]
            # print(wts)
            # print(sum(wts))
            # self.weights = wts




        i = 0
        while(len(self.sample) < sampleSize):
            if(self.weighted):
                c = np.random.choice(self.centers, p=self.weights)
                i = self.centers.index(c)
            elif(self.spherical):
                # randomly sample on unit hypersphere and pick nearest center
                seed = [random.gauss(0,1) for i in range(self.numFeatures)]
                mag = sum(x**2 for x in seed) ** .5
                seed = [x/mag for x in seed]

                cdists = []
                for center in self.centers:
                    normed = np.linalg.norm(self.data[center,:])
                    cdists.append(np.linalg.norm(normed -
                                                 seed))
                i = cdists.index(min(cdists))
                c = self.centers[i]

            else:
                # print(self.centers)
                # print(self.available)
                c = self.centers[i]

            dists = distTable[:,i].tolist()

            # dists.pop(self.available.index(c))
            smallest = min(dists)

            #dists = distTable[:,i].tolist()
            new = dists.index(smallest)

            self.sample.append(self.available[new])
            del self.available[new]

            distTable = np.delete(distTable, new, 0)
            i = (i + 1) % len(self.centers)

        return self.sample

    def vizSample(self, full=False, anno=False, annoMax=100, c='m', cmap=None, **kwargs):
        if(full):
            if self.embedding is None:
                self.embed()
            mpl.scatter(self.embedding[:,0], self.embedding[:,1])
            mpl.scatter(self.embedding[self.sample, 0], self.embedding[self.sample,1], c=c)

            if(self.weighted):
                cols = self.weights
            else:
                cols = None

            mpl.scatter(self.embedding[self.centers, 0], self.embedding[self.centers,1], c='w',
                        edgecolors='g',
                        cmap='hot')

            if(anno):
                for i in range(min([len(self.sample),annoMax])):
                    mpl.annotate(i, (self.embedding[self.sample[i],0], self.embedding[self.sample[i],1]))

        else:
            print('embedding sample only')
            if self.sampleEmbedding is None:
                self.embedSample()
            print(self.sampleEmbedding.shape)
            mpl.scatter(self.sampleEmbedding[:,0], self.sampleEmbedding[:,1], c=c, cmap=cmap)

            if(anno):
                for i in range(min([len(self.sample),annoMax])):
                    mpl.annotate(i, (self.sampleEmbedding[i,0], self.sampleEmbedding[i,1]))


        mpl.legend()

        if file is not None:
            mpl.savefig('{}.png'.format(file))

        mpl.show()
        mpl.close()


class diverseLSH(LSH):
    """uses a DPP-like process to select diverse centers,
     then assigns points to their nearest one"""

    def __init__(self, data, numCenters=10, batch=100, steps=1000, replace=False, **kwargs):
        numBands = 1
        bandSize = 1
        numHashes = 1

        LSH.__init__(self, data, numHashes=numHashes, numBands=numBands, bandSize=bandSize,
                     replace=replace, **kwargs)

        self.numCenters = numCenters
        self.batch = batch
        self.steps = steps
        self.centers = None

    def makeHash(self):
        # print('data at time of hashing:')
        # print(self.data)
        # centerSampler = detSampler(self.data, self.batch, self.replace)
        # self.centers = centerSampler.downsample(self.numCenters)

        centerSampler=dpp(self.data, steps=self.steps)
        self.centers = centerSampler.downsample(self.numCenters)


        hashes = np.empty([self.numObs, 1])

        for i in range(self.numObs):
            centerDists = []
            for c in self.centers:
                centerDists.append(np.linalg.norm(self.data[i, :]
                                                  - self.data[c, :]))
            hashes[i, 0] = self.centers[centerDists.index(min(centerDists))]

        self.hash = hashes

    def vizHash(self, file=None, maxPoints=float("inf"), plotCenters=True, anno=False, **kwargs):
        if self.embedding is None:
            tsne = sk.manifold.TSNE(**kwargs)

            if self.numObs > maxPoints:
                self.embeddingInds = np.random.choice(
                    self.numObs, maxPoints, replace=False)
            else:
                self.embeddingInds = range(self.numObs)

            fit = tsne.fit(self.data[self.embeddingInds, :])
            self.embedding = tsne.embedding_

        else:
            self.embeddingInds = range(self.numObs)
        if self.numHashes > 1:
            log('too many hashes to vizualize; visualiing only first hash')

        cols = self.hash[self.embeddingInds, 0]
        mpl.scatter(self.embedding[:, 0], self.embedding[:, 1], c=cols)

        if plotCenters:
            mpl.scatter(self.embedding[self.centers, 0],
                        self.embedding[self.centers, 1], c='r')

        if(anno):
            for i, h in enumerate(self.hash[self.embeddingInds, 0]):
                mpl.annotate(int(h), (self.embedding[self.embeddingInds[i], 0],
                                      self.embedding[self.embeddingInds[i], 1]))

        if file is not None:
            mpl.savefig('{}.png'.format(file))

        mpl.show()
        mpl.close()


class diverseSampler(seqSampler):
    def __init__(self, data, batch, numCenters, replace=False):
        seqSampler.__init__(self, data, replace)
        self.centerSampler = detSampler(data, batch, replace)
        self.numCenters = numCenters  # before repeat
        self.centers = []
        self.iter = 0  # how many centers we've sampled since last
        self.batch = batch
        self.avail = list(range(self.numObs))

    def addSample(self):
        if self.iter >= self.numCenters:
            # print('resetting')
            # print(self.centerSampler.sample)
            self.centerSampler.sample.sort(reverse=True)
            for new in self.centerSampler.sample:
                del self.avail[new]
            # clean slate on sampler
            # print(self.data[self.avail,:])
            self.centerSampler = detSampler(
                self.data[self.avail, :], self.batch, self.replace)
            self.iter = 0

        new = self.centerSampler.addSample()

        self.sample.append(new)
        # self.centerSampler.sample.append(new)
        self.iter += 1

        return(new)


class rankLSH(diverseLSH):
    """like diverseLSH but uses epsilon-balls around each center"""

    # def __init__(self, data, batch=100, replace=False, eps=0.1):
    #     diverseLSH.__init__(self, data, batch=batch, replace=replace)
    #
    #     self.avail = list(range(self.numObs))
    #     self.sampled [False]*self.numObs

    def makeHash(self):
        centerSampler = detSampler(self.data, self.batch, self.replace)
        self.centers = centerSampler.downsample(self.numCenters)

        hashes = np.empty([self.numObs, 1])

        # stores each point's distance to each center
        distTable = np.empty([self.numObs, self.numCenters])

        for i, c in enumerate(self.centers):
            dists = []
            for j in range(self.numObs):
                dists.append(np.linalg.norm(self.data[j, :]
                                            - self.data[c, :]))
            distTable[:, i] = dists

        # add rows to enumerate
        distTable = np.hstack([distTable,
                               np.transpose(np.matrix(list(range(self.numObs))))])

        while(distTable.shape[0] > 0):
            for i, c in enumerate(self.centers):
                if (distTable.shape[0] == 0):
                    break
                closest = np.argmin(distTable[:, i])
                # distTable[:,i].index(max(distTable[:,i]))
                closestInd = int(distTable[closest, -1])
                print(closestInd)
                hashes[closestInd, 0] = c
                distTable = np.delete(distTable, (closest), axis=0)

        self.hash = hashes


class ballLSH(diverseLSH):
    def __init__(self, data, batch=100, replace=False, epsilon=1, ord=None):
        numBands = 1
        bandSize = 1
        numHashes = 1

        LSH.__init__(self, data, numHashes=numHashes, numBands=numBands, bandSize=bandSize,
                     replace=replace)
        self.epsilon = epsilon
        self.batch = batch
        self.ord = ord

    def makeHash(self):
        centerSampler = detSampler(self.data, self.batch, self.replace)

        # whether each has been covered by a ball
        covered = [False] * self.numObs
        sigs = np.array([]).reshape(self.numObs, 0)

        hashes = np.empty([self.numObs, 1])

        while sum(covered) < self.numObs:
            c = centerSampler.addSample()
            print('{} centers added'.format(len(centerSampler.sample)))
            center = self.data[c, :]
            centerDists = []
            for i in range(self.numObs):
                centerDists.append(np.linalg.norm(
                    center - self.data[i, :], ord=self.ord))

            nearby = [int(x < self.epsilon) for x in centerDists]

            sigs = np.hstack([sigs, np.transpose(np.matrix(nearby))])

            covered[c] += 1
            # set nearby points as covered
            covered = np.maximum(covered, nearby)
            # print(sum(covered))

        # print(sigs)
        # print(sigs.shape)

        # make unique signature for each combo
        hashdict = {}

        for sample_idx in range(self.numObs):
            hashlist = sigs[sample_idx, :].tolist()
            hashlist = [y for x in hashlist for y in x]
            hash_cell = tuple(hashlist)
            if hash_cell not in hashdict:
                hashdict[hash_cell] = []
            hashdict[hash_cell].append(sample_idx)

        keys = list(hashdict.keys())
        for bucket in range(len(keys)):
            for idx in hashdict[keys[bucket]]:
                hashes[idx, 0] = bucket

        self.hash = hashes
        self.occSquares = len(hashdict)
        self.centers = centerSampler.sample
        return(hashes)


class gapLSH(diverseLSH):

    @staticmethod
    def findLeap(dists):
        """find the first big gap in distances"""

    def makeHash(self):
        centerSampler = detSampler(self.data, self.batch, self.replace)
        self.centers = centerSampler.downsample(self.numCenters)

        hashes = np.empty([self.numObs, 1])

        hashtable = np.empty([self.numObs, self.numCenters])

        # stores each point's distance to each center
        distTable = np.empty([self.numObs, self.numCenters])

        for i, c in enumerate(self.centers):
            dists = []
            for j in range(self.numObs):
                dists.append(np.linalg.norm(self.data[j, :]
                                            - self.data[c, :]))
            distTable[:, i] = dists

        # add rows to enumerate
        distTable = np.hstack([distTable,
                               np.transpose(np.matrix(list(range(self.numObs))))])

        while(distTable.shape[0] > 0):
            for i, c in enumerate(self.centers):
                if (distTable.shape[0] == 0):
                    break
                closest = np.argmin(distTable[:, i])
                # distTable[:,i].index(max(distTable[:,i]))
                closestInd = int(distTable[closest, -1])
                print(closestInd)
                hashes[closestInd, 0] = c
                distTable = np.delete(distTable, (closest), axis=0)

        self.hash = hashes


class pRankSampler(rankSampler):

    def rank(self):

        gene_means = np.mean(self.data, axis=0)
        gene_vars = np.var(self.data, axis=0)
        gene_sigs = np.empty(self.data.shape)

        print('means: {}'.format(gene_means))
        print('vars: {}'.format(gene_vars))

        for i in range(self.numObs):
            for j in range(self.numFeatures):

                gene_sigs[i, j] = np.abs(self.data[i, j] - gene_means[j])
                # P = norm.cdf(self.data[i,j], loc = gene_means[j], scale = gene_vars[j])
                # print(P)
                # P = min([P, 1-P])
                #
                # gene_sigs[i,j] = P

        print(gene_sigs)
        cell_sigs = np.max(gene_sigs, axis=1)
        print(min(cell_sigs))
        print(max(cell_sigs))
        print('cell sigs: {}'.format(cell_sigs))
        self.ranking = np.argsort(cell_sigs)


class angleSampler(weightedSampler):
    """weights points by the mean angle with an axis"""

    def __init__(self, data, replace=False, strength=1):
        # translate to first quadrant and normalize
        for i in range(data.shape[1]):
            data[:, i] -= data[:, i].min(0)
        data /= data.max()

        weightedSampler.__init__(self, data, strength, replace)

    def makeWeights(self):
        print('making angle weights')
        wts = [None] * self.numObs
        for i in range(self.numObs):
            mag = sum([x**2 for x in self.data[i, :]])

            # print(mag)

            angles = [None] * self.numFeatures
            for j in range(self.numFeatures):
                x = self.data[i, j]
                if math.sqrt(mag - x**2) == 0:
                    angles[j] = math.pi / 2
                else:
                    angles[j] = math.atan(float(x) / math.sqrt(mag - x**2))

            # print(angles)
            wts[i] = sum([a < (math.pi / 30) for a in angles])
            # wts[i] = sum(angles)/len(angles)
            # wts[i] = min(angles)
            # print(wts[i])

        # wts = [math.pi/2 - w for w in wts]
        #
        # # #normalize to be between 0 and 1
        # lowest = min(wts)
        # wts =[w - lowest for w in wts]
        # highest = max(wts)
        # wts = [w/highest for w in wts]

        print(wts[1:10])
        #
        #
        #
        #
        # wts = [float(1) / (w ** self.strength) if w > 0 else 1000 for w in wts]

        wts = [w**self.strength for w in wts]
        total = sum(wts)
        wts = [float(w) / total for w in wts]
        print(wts[1:10])
        self.wts = wts


class splitLSH(LSH):
    def __init__(self, data, maxSplits=2, minDiam=0, replace=False):
        numBands = 1
        bandSize = 1
        numHashes = 1

        LSH.__init__(self, data, numHashes, numBands, bandSize, replace)
        self.replace = replace
        self.maxSplits = maxSplits
        self.minDiam = minDiam

    @staticmethod
    def splitDim(X, maxSplits, minDiam):
        diam = max(X) - min(X)
        if diam < minDiam:
            return [0] * len(X)
        else:
            print('splitting...')

        Y = sorted(X)  # sort on dimension
        gaps = np.subtract(Y[1:], Y[:-1])  # hopefully vectorized

        ind = gaps.tolist().index(max(gaps))

        split = Y[ind]

        return([int(x > split) for x in X])

    def makeHash(self):
        table = np.empty(self.data.shape)
        for i in range(self.numFeatures):
            table[:, i] = splitLSH.splitDim(
                self.data[:, i], self.maxSplits, self.minDiam)

        hashDict = {}
        for i in range(self.numObs):
            print(i)
            if tuple(table[i, :]) in hashDict:
                hashDict[tuple(table[i, :])].append(i)
            else:
                hashDict[tuple(table[i, :])] = [i]
                print(len(hashDict))

        self.occSquares = len(hashDict)

        keys = list(hashDict.keys())

        result = np.empty([self.numObs, 1])

        for square in range(len(keys)):
            print(square)
            for idx in hashDict[keys[square]]:
                result[idx, 0] = square

        self.hash = result


class treeLSH(LSH):
    """rp-tree like hashing scheme"""

    def __init__(self, data, splitSize, children=2, minPoints=0):
        numBands = 1
        bandSize = 1
        numHashes = 1

        LSH.__init__(self, data, numHashes=numHashes,
                     numBands=numBands, bandSize=bandSize)

        self.data = data
        self.children = children
        self.splitSize = splitSize
        self.minPoints = minPoints

    @staticmethod
    def quantilate(vals, splitSize, children=2):
        """converts arrays of values to which quantile division they belong to"""
        diam = max(vals) - min(vals)

        # print('vals {}'.format(vals))
        # print('diameter {}'.format(diam))
        if diam < splitSize:
            return([0] * len(vals))

        # print('diameter is {}'.format(diam))
        splits = min(np.ceil(diam / float(splitSize)), children)

        return pd.qcut(vals, int(splits), labels=False)

    @staticmethod
    def dimHash(data, splitSize, children, max_splits):
        # print('shape is {}'.format(data.shape))

        # print('data has size {}'.format(data.shape[0]))
        hashes = np.empty([data.shape[0], 1])
        hashes[:, 0] = np.array(treeLSH.quantilate(
            data[:, 0], splitSize, max_splits=max_splits))

        if data.shape[1] == 1:
            return hashes
        if data.shape[0] == 1:
            hashes = np.empty(data.shape)
            hashes[0, :] = [0] * data.shape[1]
            return hashes

        result = np.empty(data.shape)
        result[:, 0] = hashes[:, 0]

        if len(np.unique(hashes)) > 1:
            print('splitting into {}'.format(len(np.unique(hashes))))
        # print('there are {} splits'.format(len(np.unique(hashes))))
        for val in np.unique(hashes):
            inds = [i for i in range(len(hashes)) if hashes[i] == val]
            # print('recursing')
            subframe = treeLSH.dimHash(
                data[inds, 1:], splitSize, children, max_splits)

            result[inds, 1:] = subframe
        # print(result)
        return(result)

    def makeHash(self):
        result = np.empty((self.numObs, 1))
        # #fill first column of table
        # table[:,0] = treeLSH.quantilate(self.data[:,0], self.splitSize, self.children)

        cur_dict = {}
        # start: everything in empty square
        cur_dict[tuple([])] = range(self.numObs)
        for i in range(self.data.shape[1]):
            print('dealing with dimension {}'.format(i))
            new_dict = {}
            for k in cur_dict.keys():
                # print('partition {}'.format(k))
                inds = cur_dict[k]  # which indices have this signature
                if len(inds) <= self.minPoints:
                    # don't partition
                    new_dict[k] = inds
                    continue

                new_keys = treeLSH.quantilate(
                    self.data[inds, i], self.splitSize, self.children)
                if len(np.unique(new_keys)) == 1:
                    new_dict[k] = inds
                    continue  # no partitioning to do, so don't bother
                for nk in np.unique(new_keys):
                    new_dict[k + tuple([nk])] = [inds[j]
                                                 for j in range(len(inds)) if new_keys[j] == nk]
            cur_dict = new_dict
            print('updated dict has length {}'.format(len(cur_dict)))

        self.occSquares = len(cur_dict)

        keys = list(cur_dict.keys())
        for square in range(len(keys)):
            for idx in cur_dict[keys[square]]:
                result[idx, 0] = square

        self.hash = result


class gridLSH(LSH):
    """just bins coordinates to make an orthogonal grid"""

    def __init__(self, data, gridSize, replace=False, target=10, randomize_origin=True, cell_labels=None, cluster_labels=None, record_counts=False):
        numBands = 1
        bandSize = 1
        numHashes = 1
        LSH.__init__(self, data, numHashes=numHashes, numBands=numBands, bandSize=bandSize,
                     replace=replace, target=target)
        self.gridSize = gridSize
        self.randomize_origin = randomize_origin
        self.cell_labels = cell_labels
        self.cluster_labels = cluster_labels
        self.record_counts = record_counts
        self.occSquares = None

    def makeHash(self):

        # make positive and max 1
        X = self.data - self.data.min(0)
        X /= X.max()

        # hashes = np.empty((self.numObs,self.numFeatures))
        hashes = np.empty((self.numObs, 1))

        grid = {}

        if(self.randomize_origin):
            for i in range(self.numFeatures):
                # print(X[:,i])
                # determine how much you can shift without altering no. grid squares
                shift_min = (-1) * X[:, i].min()
                shift_max = (self.gridSize *
                             (np.floor(X[:, i].max() / float(self.gridSize))) + 1) - X[:, i].max()
                # shift_min = X[:,i].max() - (self.gridSize * (np.floor(X[:,i].max()/float(self.gridSize))) + 1)

                # print('data min: {}'.format(X[:,i].min()))
                # print('data max: {}'.format(X[:,i].max()))
                # print('min: {}'.format(shift_min))
                # print('max: {}'.format(shift_max))

                shift = random.uniform(shift_min, shift_max)

                # print('shift: {}'.format(shift))
                for j in range(self.numObs):
                    X[j, i] += shift

                # print('new data min: {}'.format(X[:,i].min()))
                # print('new data max: {}'.format(X[:,i].max()))
                # print('data: {}'.format(X[:,i]))

        # make dict mapping grid squares to points in it
        for i in range(self.numObs):
            coords = X[i, :]
            # if(randomize_origin): #shift data by random vector
            #     for x in coords:
            #         x += random.uniform(0,self.gridSize)

            gridsquare = tuple(
                np.floor(coords / float(self.gridSize)).astype(int))

            if gridsquare not in grid:
                grid[gridsquare] = set()
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
                # print(len(grid.values()))
                # print(grid.values())
                counts = [len([i for i in square if self.cluster_labels[i] == lab])
                          for square in grid.values()]
                # print('counts: {}'.format(counts))
                counts = [count for count in counts if count > 0]

                # print('counts: {}'.format(counts))
                # normalize to percentages
                counts = [float(count) / sum(counts) for count in counts]
                # print('counts: {}'.format(counts))

                counts.sort(reverse=True)
                pct_covered = [sum(counts[:(i + 1)])
                               for i in range(len(counts))]
                print(pct_covered)
                good_inds = [i for i in range(
                    len(pct_covered)) if pct_covered[i] > 0.5]
                score = min(good_inds)

                # score = sum([count for count in counts])/len(counts)
                # score = max(counts)
                scores[lab] = score
            self.clustScores = scores
            print(scores)
        # enumerate grid squares, and assign each obs to its square index
        keys = list(grid.keys())
        for square in range(len(keys)):
            for idx in grid[keys[square]]:
                hashes[idx, 0] = square

        self.hash = hashes


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

    def findCandidates(self, ind):  # OR of all summand candidates
        candidates = []
        for lsh in self.components:
            candidates.append(lsh.findCandidates(ind))

        candidates = set().union(*candidates)

        if self.keepStats:
            self.nbhdSizes.append(len(candidates) + 1)
        return candidates


class prodLSH(multiLSH):
    def __init__(self, factors):
        multiLSH.__init__(self, factors)

    def findCandidates(self, ind):  # AND of all product candidates
        candidates = []
        for lsh in self.components:
            candidates.append(lsh.findCandidates(ind))

        candidates = set().intersection(*candidates)

        if self.keepStats:
            self.nbhdSizes.append(len(candidates) + 1)

        return candidates


class gsLSH(LSH):
    def __init__(self, data, target='auto', gridSize=None, replace=False, alpha=0.1,
                 max_iter=200, verbose=True, opt_grid=True):

        LSH.__init__(self, data, numHashes=1, numBands=1, bandSize=1, replace=replace,
                     target=target)

        self.gridSize = gridSize  # starting grid size before optimization

        self.alpha = alpha
        # self.target = target # downsampling size you're built for
        self.verbose = verbose
        self.max_iter = max_iter
        self.opt_grid = opt_grid
        self.occSquares = None

        print('target: {}'.format(self.target))

    def makeHash(self):  # re-implementation of gs, formulated as an LSH
        n_samples, n_features = self.data.shape

        print('target is {} at time of hashing'.format(self.target))

        X = self.data - self.data.min(0)
        X -= X.max()

        hashes = np.empty((self.numObs, 1))  # table of grid square assignments

        X_ptp = X.ptp(0)  # list of ranges for each feature

        low_unit, high_unit = 0., max(X_ptp)

        if self.gridSize is None:
            self.gridSize = (low_unit + high_unit) / 4

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
                # too many grid cells
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
        self.weights = [0] * self.numObs


        # enumerate grid squares, and assign each obs to its square index
        keys = list(grid.keys())
        for square in range(len(keys)):
            for idx in grid[keys[square]]:
                self.weights[idx] = 1./(len(grid[keys[square]]))
                hashes[idx, 0] = square

        self.hash = hashes
        self.gridLabels = [int(x) for y in list(hashes) for x in y]
        self.gridSize = unit

    def downsample_weighted(self, sampleSize, alpha=1, replace=False):
        print('alpha is {}'.format(alpha))

        if self.target == 'N':
            self.target = sampleSize
        self.makeHash()
        self.makeFinder()

        grid = self.finder[0]
        full_grid = grid  # keeps all squares

        # sizes of grid squares
        sizes = {square: len(v) for (square, v) in grid.items()}
        # print('square sizes: {}'.format(sizes))

        weights = [1 / (np.power(size, alpha)) for size in sizes.values()]
        total = sum(weights)

        weights = [float(w) / total for w in weights]  # make them sum to 1
        assert(np.abs(sum(weights) - 1) < .000001)

        available = range(self.numObs)
        included = [True] * self.numObs  # all indices available
        square_sampled = [True] * len(grid)  # whether square has been sampled
        avail_square = range(len(grid))  # which squares available

        sample = []
        valid_sample = [True] * self.numObs  # true if hasn't been sampled
        sample_inds = []  # indices relative to available samples
        count = 0  # how many have been added since reset
        reset = False  # whether we have reset

        if self.keepStats:
            self.lastCounts = []

        subinds = []

        for n in range(sampleSize):
            grid_cells = list(grid.keys())

            grid_cell = grid_cells[np.random.choice(
                len(grid_cells), p=weights)]

            samples = list(grid[grid_cell])
            sample = samples[np.random.choice(len(samples))]

            # del grid[grid_cell]

            if not replace:
                # print(grid[grid_cell])
                grid[grid_cell].remove(sample)
                # print(grid[grid_cell])
                sizes[grid_cell] -= 1
                if len(grid[grid_cell]) == 0:
                    del grid[grid_cell]
                    del sizes[grid_cell]

                # update weights
                weights = [1 / (np.power(size, alpha))
                           for size in sizes.values()]

                total = sum(weights)
                # make them sum to 1
                weights = [float(w) / total for w in weights]
                assert(np.abs(sum(weights) - 1) < .000001)

            # print('appending {}'.format(sample))
            subinds.append(sample)
            # print('samples is now {}'.format(subinds))

        # print('samples: {}'.format(samples))
        return(sorted(subinds))


class cosineLSH(LSH):

    def makeHash(self):
        projector = np.random.randn(self.numFeatures, self.numHashes)
        projection = (np.sign(np.matmul(self.data, projector)) + 1) / 2
        self.hash = projection


class projLSH(LSH):

    def __init__(self, data, numHashes, numBands, bandSize, gridSize=0.1, replace=False, target=10):

        LSH.__init__(self, data, numHashes=numHashes, numBands=numBands,
                     bandSize=bandSize, replace=replace, target=target)

        self.gridSize = gridSize

    def makeHash(self):
        """
        projects randomly, and bins projection values
        """

        projector = np.random.randn(self.numFeatures, self.numHashes)
        projection = np.matmul(self.data, projector)

        projection = projection - projection.min(0)
        projection = projection / projection.max()

        projection = np.floor(projection / self.gridSize)

        self.hash = projection


class randomGridLSH(LSH):
    """like gridLSH, but grid axes are randomly chosen orthogonal basis"""

    def __init__(self, data, gridSize, numHashes, numBands, bandSize, replace=False, target=10):

        LSH.__init__(self, data, numHashes=numHashes, numBands=numBands, bandSize=bandSize,
                     replace=replace, target=target)

        self.gridSize = gridSize

    def makeHash(self):
        t0 = time()
        hashes = np.empty((self.numObs, self.numHashes))
        t1 = time()

        print('initiating took {} seconds'.format(t1 - t0))

        for hashno in range(self.numHashes):
            t0 = time()
            basis = rvs(dim=self.numFeatures)  # random orthonormal basis

            newData = np.matmul(self.data, basis)
            t1 = time()

            # print('making random basis took {} seconds'.format(t1-t0))

            # do gridLSH in this new basis

            # make positive and max 1
            X = newData - newData.min(0)
            X /= X.max()

            grid = {}

            # make dict mapping grid squares to points in it
            for i in range(self.numObs):
                coords = X[i, :]

                gridsquare = tuple(
                    np.floor(coords / float(self.gridSize)).astype(int))

                if gridsquare not in grid:
                    grid[gridsquare] = set()
                grid[gridsquare].add(i)

            # enumerate grid squares, and assign each obs to its square index
            keys = list(grid.keys())
            for square in range(len(keys)):
                for idx in grid[keys[square]]:
                    hashes[idx, hashno] = square

        self.hash = hashes

# how to add two hashers
# can't overload operator due to weird dependencies?


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



    # gauss2D = gauss_test([10,20,100,200], 2, 4, [0.1, 1, 0.01, 2])
    # mpl.scatter(gauss2D[:, 0], gauss2D[:, 1])
    #
    # # downsampler = gridLSH(gauss2D,0.1)
    # downsampler = projLSH(gauss2D, 10, 2, 5, 0.05)
    #
    # subInds = downsampler.downSample(50)
    #
    # print(subInds)
    #
    # mpl.scatter(gauss2D[subInds, 0], gauss2D[subInds, 1], c='m')
    # mpl.show()

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
