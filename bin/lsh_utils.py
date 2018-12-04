from __future__ import division

import numpy as np
import fractions as fr
import math
import scipy as sp


def find_params(thresh, error = 0.01, min_size=2, min_bins=2, max_fpr= 0.05, max_fnr=0.05):
    bins_cur, size_cur = min_bins, min_size
    print(bins_cur)
    print(size_cur)
    break_value = math.pow(1/bins_cur,1/size_cur)


    # break_value = float(fr.Fraction(str(1/bins_cur))**fr.Fraction(str(1/size_cur)))
    print(break_value)


    while(abs(break_value-thresh)>error):
        if break_value < thresh:
            size_cur = size_cur + 1
        elif break_value > thresh:
            bins_cur = bins_cur + 1

        break_value = math.pow(1/bins_cur,1/size_cur)

        print(break_value)




    return(tuple([bins_cur,size_cur]))



if __name__ == '__main__':
    print(find_params(0.99, min_bins=100))
