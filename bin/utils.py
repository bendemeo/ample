import errno
from fbpca import pca
import datetime
import numpy as np
import os
from sklearn.random_projection import SparseRandomProjection as JLSparse
import sys

# Default parameters.
DIMRED = 100

def log(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

def reduce_dimensionality(X, method='svd', dimred=DIMRED, raw=False):
    if method == 'svd':
        k = min((dimred, X.shape[0], X.shape[1]))
        U, s, Vt = pca(X, k=k, raw=raw)
        return U[:, range(k)] * s[range(k)]
    elif method == 'jl_sparse':
        jls = JLSparse(n_components=dimred)
        return jls.fit_transform(X).toarray()
    elif method == 'hvg':
        X = X.tocsc()
        disp = dispersion(X)
        highest_disp_idx = np.argsort(disp)[::-1][:dimred]
        return X[:, highest_disp_idx].toarray()
    else:
        sys.stderr.write('ERROR: Unknown method {}.'.format(svd))
        exit(1)

def dispersion(X, eps=1e-10):
    mean = X.mean(0).A1

    X_nonzero = X[:, mean > eps]
    nonzero_mean = X_nonzero.mean(0).A1
    nonzero_var = (X_nonzero.multiply(X_nonzero)).mean(0).A1
    del X_nonzero

    nonzero_dispersion = (nonzero_var / nonzero_mean)

    dispersion = np.zeros(X.shape[1])
    dispersion[mean > eps] = nonzero_dispersion
    dispersion[mean <= eps] = float('-inf')

    return dispersion

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H

     
# if __name__ == '__main__':
#     print(rvs(3))
