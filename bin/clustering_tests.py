from fbpca import pca
import numpy as np
import os
from scanorama import *
from scipy.sparse import vstack
from sklearn.preprocessing import LabelEncoder, scale

from experiments import *
from process import load_names
from utils import *
from LSH import *
from hashers import *
from fttree import *
import sys
import pickle
import scanpy as sc

##PBMC##
with open('target/experiments/pbmc_ft.txt', 'r') as f:
    order = f.readlines()[0].split('\t')
    order = [int(x) for x in order]

print(order)
