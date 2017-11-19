import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg import cholesky

def randncor(n,N,C):
    try:
        A = cholesky(C)
    except LinAlgError:
        m=0
        print('A is not positive definite')
    m = n
    u = np.random.randn(m,N)
    x = A.conj().transpose().dot(u)
    return x 