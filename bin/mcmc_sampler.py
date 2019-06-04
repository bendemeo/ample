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
from itertools import *
from sklearn.metrics.pairwise import pairwise_distances
from fbpca import pca
from scanorama import *
from vptree import *
from heapq import heappush, heappop, heappushpop
import pickle

def euclidean(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))



class mcmcSampler(sampler):
    def __init__(self, data, iters, DIMRED=2):
        sampler.__init__(self, data)
        self.iters = iters
        self.DIMRED = DIMRED
        self.treeData = self.data[:,:DIMRED]

    def downsample(self, sampleSize='auto'):
        if sampleSize == 'auto':
            sampleSize = math.sqrt(self.numObs)

        #initialize to random sample
        sampled_inds = np.random.choice(self.numObs, sampleSize, replace=False)
        sampled = [False]*self.numObs
        for i in sampled_inds:
            sampled[i]=True
        unsampled_inds = list(itertools.compress(range(self.numObs),sampled))


        sampleData = self.treeData[sampled_inds,:]

        T = VPTree(self.treeData, euclidean, inds=sampled_inds, PCA=False, DIMRED=self.DIMRED)

        iter = 1
        lastBuild=1

        while(iter < self.iters):
            print('iter number {}/{}'.format(iter, self.iters))
            treeIndPos = np.random.choice(len(sampled_inds))
            treeInd = sampled_inds[treeIndPos]
            treePt = self.data[treeInd,:]
            treeNeighbor = T.get_nearest_neighbor(treePt, full=True)
            treeNeighborInd = treeNeighbor.ind
            treeNeighborPt = self.data[treeNeighborInd,:]
            treeDist = euclidean(treeNeighborPt, treePt)

            newIndPos = np.random.choice(len(unsampled_inds))
            newInd = unsampled_inds[newIndPos]
            newPt = self.data[newInd,:]
            newNeighborInd = T.get_nearest_neighbor(newPt, full=True).ind
            newNeighborPt = self.data[newNeighborInd,:]
            newDist = euclidean(newNeighborPt, newPt)

            if(treeDist < newDist):
                #replace nearest tree neighbor with new index
                sampled[treeNeighborInd]=False
                sampled[newInd] = True

                #update sampled and unsampled indices
                sampled_inds = list(itertools.compress(range(self.numObs),sampled))
                del unsampled_inds[newIndPos]
                unsampled_inds.append(treeNeighborInd)

                #remove tree's neighbor from consideration
                treeNeighbor.active = False

                #add new point to tree
                T.add(newPt, newInd)

                lastBuild += 1
                if lastBuild > sampleSize/2:
                    #freshen up the tree
                    print('rebuilding tree!')
                    sampleData = self.treeData[sampled_inds,:]
                    T = VPTree(sampleData, euclidean, inds=sampled_inds, DIMRED=self.DIMRED)
            iter += 1
        self.sample = sampled_inds
        return(sampled_inds)
