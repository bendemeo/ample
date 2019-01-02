from LSH import *
from hashers import *
from test_file import *

# def optimize_param(lsh, param, N, inverted = False, step=1):
#     cur_val = getattr(lsh, param)
#
#     subsample = lsh.downSample(N)
#
#     counts = lsh.getMeanCounts() #  how many did you sample?
#
#     while counts > 0:
#         cur_val = cur_val + step #increase
#
#         setattr(lsh, param, cur_val)
#         subsample = lsh.downSample(N)
#         counts = lsh.getMeanCounts() #  how many did you sample?
#
#         print(counts)
#         print(cur_val)
#
#
#     while(counts < 0):
#         cur_val = cur_val - step
#         setattr(lsh, param, cur_val)
#         subsample = lsh.downSample(N)
#         counts = lsh.getMeanCounts()
#
#     cur_val = cur_val - 1
#
#     return(lsh)
#

if __name__=='__main__':

    gauss2D = gauss_test([10,20,100,2000], 2, 4, [0.1, 1, 0.01, 2])

    # hasher = cosineLSH(gauss2D, numHashes = 10000, numBands = 5, bandSize=50,
    # replace = False, keepStats=True, allowRepeats=True)


    # hasher = gridLSH(gauss2D, replace = False,
    # gridSize=0.2)
    #
    # hasher = projLSH(gauss2D, numHashes = 100, numBands = 2, bandSize=20, gridSize=0.1,replace = False)

    hasher = randomGridLSH(gauss2D, numHashes=100, numBands=2, bandSize=10, replace=False,
    gridSize=0.1)

    hasher.optimize_param('gridSize',1000,step=0.01, inverted = True)
