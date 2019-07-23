import numpy as np


def euclidean(x1, x2):
    return(np.linalg.norm(np.array(x1)-np.array(x2)))

def trunc_euclidean(k=5):
    def F(x,y):
        return np.linalg.norm(np.array(x)[:k]-np.array(y)[:k])

    return(F)

def top_diffs(k=1):
    def F(x,y):
        diffs = np.absolute(np.array(x)-np.array(y))
        #print(diffs)
        topdiffs = nlargest(k, diffs)
        return(np.linalg.norm(topdiffs))
    return(F)

def neg_pearson(x,y): #negative pearson correlation
    return(-1*((pearsonr(np.array(x),np.array(y))[0]+1)/2))
