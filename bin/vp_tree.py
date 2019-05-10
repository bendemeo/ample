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
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from fbpca import pca
from scanorama import *


class vpNode:
    def __init__(self, vp=None, rad=None, left=None, right=None, ind=None, parent=None, leaf=False):
        self.vp = vp
        self.ind = ind #index in global data
        self.rad = rad
        self.left = left
        self.right = right
        self.leaf = leaf
        self.parent = parent
        self.visited = False

    def tostr(self):

        if self.leaf:
            return('END')
        else:
            result = str(self.ind) + ', ['
            result += self.left.tostr() + ', '
            result += self.right.tostr() + ']'


        return(result)

    def prune(self):
        if self.leaf:
            print("tried to prune leaf node")
        elif self.left.leaf and self.right.leaf:
            self.leaf = True
            if self.parent is not None and self.parent.visited:
                self.parent.prune()

class vpTree:
    def __init__(self, data, inds=None):
        self.data = data
        if inds is None:
            inds = list(range(self.data.shape[0]))

        t0 = time()
        #print('building tree...')
        self.tree = vpTree.buildTree(data, inds)
        t1 = time()
        #print('built tree in {} seconds'.format(t1-t0))

    @staticmethod
    def buildTree(data, inds):
        """recursive method for building VP-trees"""

        numObs, numFeatures = data.shape
        if numObs == 0:
            return vpNode(leaf=True)
        if numObs == 1:
            return vpNode(data[0,:], rad=None,
                          left=vpNode(leaf=True),
                          right=vpNode(leaf=True),
                          ind=inds[0])
        else:
            vp_ind = np.random.choice(range(numObs), 1)[0]
            # print(vp_ind)


            vp = data[vp_ind,:].reshape(1,numFeatures)
            ind = inds[vp_ind]

            #remove it from data
            data = np.delete(data, (vp_ind), axis=0)
            del inds[vp_ind]

            dists = euclidean_distances(vp, data)[0,:]
            # print(dists)

            rad = np.median(dists)

            split = [x < rad for x in dists]
            # print(split)

            left_positions = list(itertools.compress(list(range(data.shape[0])), split))
            right_positions = list(itertools.compress(list(range(data.shape[0])), [not x for x in split]))

            # print(left_positions)
            # print(right_positions)
            # print(inds)
            #points for left and right children
            left_data = data[left_positions,:]
            right_data = data[right_positions,:]

            left_inds = [inds[x] for x in left_positions]
            right_inds = [inds[y] for y in right_positions]

            left_child = vpTree.buildTree(left_data, left_inds)
            right_child = vpTree.buildTree(right_data, right_inds)

            result = vpNode(vp, rad, left_child, right_child, ind)
            left_child.parent = result
            right_child.parent = result

            return(result)


    def NNSearch(self, query, rad, prune = True, nearest = True):
        query = np.array(query)
        toSearch = [self.tree]

        result = []
        searched = 0

        while(len(toSearch) > 0):
            currentNode = toSearch.pop(0)
            if(currentNode.leaf):
                continue

            searched += 1
            dist = np.linalg.norm(query-currentNode.vp)

            if dist <= rad:
                if not currentNode.visited:
                    result = result + [currentNode.ind]
                if nearest:
                    result = [currentNode.ind]
                    rad = dist



                if prune:
                    currentNode.prune()


            if currentNode.rad is not None:
                if dist <= currentNode.rad + rad:
                    toSearch = toSearch + [currentNode.left]

                if dist > currentNode.rad - rad:
                    toSearch = toSearch + [currentNode.right]

        #print(self.tree.tostr())
        print('{} nodes searched'.format(searched))
        print('{} nodes returned'.format(len(result)))
        return(result)
