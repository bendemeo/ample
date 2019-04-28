# make projection ala mouse brain? Bring to high dimension, see what squares look like

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




def flyTransform(data, inputs=5, out_dim=200):
    # take [out_dim] random groups of [inputs] elements, sum them, and return

    numObs, numFeatures = data.shape

    inputMatrix = np.empty((out_dim, inputs), dtype=int)
    output = np.empty((numObs, out_dim))

    for i in range(out_dim):
        vals = np.random.choice(numFeatures, size = inputs)
        vals = [int(x) for x in vals]
        inputMatrix[i,:] = vals

    for i in range(numObs):
        for j in range(out_dim):
            #print(inputMatrix[j,:])
            output[i,j]=sum(data[i,inputMatrix[j,:]])

    return(output)
