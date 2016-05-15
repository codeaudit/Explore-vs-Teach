""" Utilities for e_vs_t """
import numpy as np
import warnings

def normalizeLili(Lili):
    nor = np.sum(flatten(Lili))
    if nor <= 0:
        warnings.warn('All entries should >=0 with at least one >0')
        return Lili
    else:
        norLili = [[item/nor for item in li] for li in Lili]
        # print("Sum of Lili = %s" %(np.sum(flatten(norLili))))
        return norLili

def flatten(L):
    '''Flattens nested lists or tuples with non-string items'''
    for item in L:
        try:
            for i in flatten(item):
                yield i
        except TypeError:
            yield item

def normalizeRow(M):
    nrow = M.shape[0]
    for irow in range(nrow):
        if M[irow,:].sum()>0:
            M[irow,:] = normalize(M[irow,:])
    return M

def normalizeCol(M):
    ncol = M.shape[1]
    for icol in range(ncol):
        if M[:,icol].sum()>0:
            M[:,icol] = normalize(M[:,icol])
    return M

def uniformSampleMaxInd(vec):
    norVec = normalize((vec==vec.max()).astype(float))
    return randDistreteSample(norVec)

def makeZero(vec, inds):
    vec[inds] = 0
    return vec

def normalize(vec):
    vec = np.array(vec)
    nor = vec.sum()
    if nor <= 0:
        # print('input vec = %s' %(vec))
        # raise ValueError('All entries should >=0 with at least one >0')
        warnings.warn('All entries should >=0 with at least one >0')
        return vec
    else:
        return vec/nor

def randomNor(n):
    randVec = np.random.rand(n)
    return normalize(randVec)

def perturbDistr(distr, scale):
    """ perturb distr by adding random noise of magnitude scale """
    n = len(distr)
    noise = scale*np.random.rand(n)
    return normalize(distr + noise)

def randDistreteSample(probVec):
    r = np.random.rand()
    cumProb = np.cumsum(probVec)
    ans = np.where(cumProb > r)
    return ans[0][0]
