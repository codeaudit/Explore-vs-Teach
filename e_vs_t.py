"""Exploration vs Teaching"""
from itertools import chain
from copy import deepcopy
from scipy.stats import entropy
from unique_permutations import unique_permutations as uperm

from utils import normalize
from utils import makeZero
from utils import normalizeRow
from utils import normalizeCol
from utils import randDistreteSample

import numpy as np


"""
2016-02-22
-more generic class for hierarchical configuration?
2016-03-10
-inference methods in the model class could be its own class?
-refactore so that class within class becomes list of lists
-refactored so that no distribution changes inside any method
2016-03-12
-code a test class using unittest?
2016-04-08
-in initHypoProbeMatrix and updateHypoProbeMatrix, should not use
"if postHypo[ihypo] > 0.:" before model.posteriorLabelGivenHypo
to skip hypo, because we eventually want P(x|h).
"""

def defineCompartment(name):
    compk = np.zeros(16)
    if name is '1H':
        # 1 horizontal split
        compk[8:] = 1.
    elif name is '1V':
        # 1 vertical split
        compk[2::4] = 1.
        compk[3::4] = 1.
    elif name is '3H':
        # 3 horizontal splits
        compk[4:8] = 1.
        compk[8:12] = 2.
        compk[12:16] = 3.
    elif name is '3V':
        # 3 vertical splits
        compk[[1, 5, 9, 13]] = 1.
        compk[[2, 6, 10, 14]] = 2.
        compk[[3, 7, 11, 15]] = 3.
    elif name is '4Q':
        # 4-quadrant split
        compk[[2, 3, 6, 7]] = 1.
        compk[[8, 9, 12, 13]] = 2.
        compk[[10, 11, 14, 15]] = 3.
    elif name is 'NR':
        #no repeat
        compk = np.arange(16)
    return compk

def genBasePerm(compk):
    """ genearte a base label configuration from compartment label compk """
    nk = len(set(compk))

    if (nk % 2):
        raise ValueError('number of compartment should be even.')
    else:
        base = np.append(np.zeros(nk/2), np.ones(nk/2)).tolist()
    return base

def genAllPerm(nx, compk, base):
    """ define all label configurations given a base label configuration """
    nbase = len(base)
    basePerm = [list(item) for item in uperm(base)]
    nperm = len(basePerm)
    perm = []
    for indp in range(nperm):
        perm.append(np.zeros(nx))
        for indb in range(nbase):
            perm[indp][np.where(compk == indb)[0]] = basePerm[indp][indb]
    return perm, nperm

def NRConfiguration():
    compk = defineCompartment('NR')
    base = genBasePerm(compk)
    MC, nMC = genAllPerm(16, compk, base)
    return MC, nMC

def findIndexPerm(allItem, target):
    for ind, item in enumerate(allItem):
        if np.array_equiv(item, target):
            return ind

def arrNotInArrList(arr, arrList):
    """ is an np array in a list of np arrays """
    a = np.array(arr)
    for item in arrList:
        item = np.array(item)
        if np.array_equiv(item, a):
            return False
    return True

def permSet(permLL):
    # permLL should be a list of list
    uniquePerm = []
    for permL in permLL:
        for perm in permL:
            if arrNotInArrList(perm, uniquePerm):
                uniquePerm.append(perm)
    return uniquePerm

def genMasterPermSet():
    compk = [defineCompartment('1H'),
             defineCompartment('1V'),
             defineCompartment('4Q'),
             defineCompartment('3H'),
             defineCompartment('3V')]
    n = len(compk)
    permLL = [0]*n
    for i in range(n):
        base = genBasePerm(compk[i])
        permL, _ = genAllPerm(16, compk[i], base)
        permLL[i] = permL
    return permSet(permLL)

class model:
    """ define a person's hypothesis space and hierarchical prior probability """
    nhypo = 3 #4
    x = np.arange(16)
    nx = len(x)
    allPermSet = genMasterPermSet()

    def __init__(self):
        """ manually initialize pattern """
        # bad way initization
        self.perm = [0]*model.nhypo
        self.perm[0] = model.allPermSet[0:2]
        self.perm[1] = model.allPermSet[2:4]
        self.perm[2] = model.allPermSet[0:6]
        # self.perm[3] = model.allPermSet[0:4] + model.allPermSet[6:14]
        self.nperm = [len(p) for p in self.perm]

        model.initPriorHypo(self)
        model.initPriorLabelGivenHypo(self) #usage:[ihypo][iconfig]
        model.initPosteriorJoint(self) #usage:[ihypo][iconfig]

        # hard-wired special case for NR at last hypo
        # self.uniPerm = self.perm[3]
        # self.nUniPerm = self.nperm[3]
        # self.permId = model.idPerm4NR(self)

        # for case without NR
        self.uniPerm = model.allPermSet
        self.nUniPerm = len(self.uniPerm)
        self.permId = model.idPerm(self) #usage:[ihypo][iconfig]

    def idPerm(self):
        """ assign an id to each permutation relative to unique perm"""
        allPerm = self.uniPerm
        permId = [[findIndexPerm(allPerm, self.perm[ihypo][iconfig])
            for iconfig in range(self.nperm[ihypo])]
                for ihypo in range(model.nhypo)]
        return permId

    def idPerm4NR(self):
        """ use if perm contains the No Repeat type at last hypo"""
        # hard-wired
        allPerm, _ = NRConfiguration()
        permId = [[findIndexPerm(allPerm, self.perm[ihypo][iconfig])
            for iconfig in range(self.nperm[ihypo])]
                for ihypo in range(3)]
        permId.append(np.arange(self.nperm[3]))
        return permId

    def getPossPostVals(self):
        """ get possible posterior values for full observations """
        nperm = [self.nperm[ihypo] for ihypo in range(model.nhypo)]
        prob = np.multiply(np.divide(np.ones(model.nhypo), nperm), self.priorHypo)
        post = [normalize(makeZero(prob, [0])),
                normalize(makeZero(prob, [1])),
                normalize(makeZero(prob, [0, 1])),
                normalize(makeZero(prob, [0, 1, 2]))]
        return set(chain.from_iterable(post))

    def initPriorHypo(self):
        """ initialize prior over hypothesis """
        self.priorHypo = np.ones(model.nhypo)/model.nhypo

    def initPriorLabelGivenHypo(self):
        """ initialize P(label onfiguration|hypo) """
        self.priorLabelGivenHypo = [
            np.ones(self.nperm[ihypo])/self.nperm[ihypo]
            for ihypo in range(model.nhypo)]

    def initPosteriorJoint(self):
        """ initialize posterior joint to prior """
        self.postJoint = [
            [self.priorLabelGivenHypo[ihypo][iconfig]*self.priorHypo[ihypo]
            for iconfig in range(self.nperm[ihypo])]
            for ihypo in range(model.nhypo)]

    @staticmethod
    def gety(perm, x):
        return perm[x]

    @staticmethod
    def likelihood(perm, X, Y):
        """ return likelihood of observing X and Y (vectos) given a label configuration """
        nobs = len(X)
        for i in range(nobs):
            if Y[i] != model.gety(perm, X[i]):
                return 0.
        return 1.

    def posteriorJoint(self, X, Y):
        """ compute posterior of label, hypo jointly given observations X and Y.
            P(h,f|D) = prod_h prod_f P(y|x,f)P(f|h)P(h) """
        # Normalized? No. Does it matter? No, because postHypo and postLabel are
        postJoint = [
            [model.likelihood(self.perm[ihypo][iconfig], X, Y)
            *self.priorLabelGivenHypo[ihypo][iconfig]
            *self.priorHypo[ihypo]
            for iconfig in range(self.nperm[ihypo])]
            for ihypo in range(model.nhypo)]
        return postJoint
        # return normalizeLili(postJoint)

    def updatePosteriorJoint(self, x, y, postJoint):
        """ update joint with one new observation pair, unnormalized.
            These update functions could be the computational bottle neck. """
        update = [
            [postJoint[ihypo][iconfig]
            *model.likelihood(self.perm[ihypo][iconfig], x, y)
            for iconfig in range(self.nperm[ihypo])]
            for ihypo in range(model.nhypo)]
        return update
        # return normalizeLili(update)

    def updatePosteriorJointWithTeacher(self, x, y, postJoint, probXHypo):
        """ update joint with one new observation pair & teacher's choice prob, unnormalized """
        update = [
            [postJoint[ihypo][iconfig]
            *probXHypo[ihypo]
            *model.likelihood(self.perm[ihypo][iconfig], x, y)
            for iconfig in range(self.nperm[ihypo])]
            for ihypo in range(model.nhypo)]
        return update
        # return normalizeLili(update)

    @staticmethod
    def posteriorHypo(postJoint):
        """ compute posterior of hypo.
            P(h|D) = 1/Z sum_f P(f,h|D) """
        postHypo = np.zeros(model.nhypo)
        for ihypo in range(model.nhypo):
            nperm = len(postJoint[ihypo])
            for iconfig in range(nperm):
                postHypo[ihypo] += postJoint[ihypo][iconfig]
        postHypo = normalize(postHypo)
        return postHypo

    def posteriorLabel(self, postJoint):
        """ compute posterior of label (or configuration).
            (f|D) = 1/Z sum_h P(f,h|D) """
        postLabel = np.zeros(self.nUniPerm)
        for ihypo in range(model.nhypo):
            for iconfig in range(self.nperm[ihypo]):
                idp = self.permId[ihypo][iconfig]
                postLabel[idp] += postJoint[ihypo][iconfig]
        postLabel = normalize(postLabel)
        return postLabel

    def posteriorLabelGivenHypo(self, postJoint, ihypo):
        """ P(f|h,D): pick out from P(f,h|D) """
        postLabel = np.zeros(self.nUniPerm)
        for iconfig in range(self.nperm[ihypo]):
            idp = self.permId[ihypo][iconfig]
            postLabel[idp] += postJoint[ihypo][iconfig]
        postLabel = normalize(postLabel)
        return postLabel

    @staticmethod
    def predicty(uniPerm, postLabel, x):
        """ compute posterior predictive distriubtion of y for one probe x.
            P(y|x,D) = sum_f P(y|x,f)P(f|D).
            Checked: yis0 + yis1 = 1, even with posteriorLabelGivenHypo. """
        yis0 = np.zeros(1)
        yis1 = np.zeros(1)
        for iconfig, config in enumerate(uniPerm):
            y = model.gety(config, x)
            if y == 0:
                yis0 += postLabel[iconfig]
            elif y == 1:
                yis1 += postLabel[iconfig]
        #print("yis0 + yis1 = %s" %(yis0 + yis1))
        return yis0, yis1

    @staticmethod
    def predictY(uniPerm, postLabel, X):
        """ loop over predicty for multiple probes X """
        probYis0 = np.zeros(model.nx)
        probYis1 = np.zeros(model.nx)
        for x in X:
            yis0, yis1 = model.predicty(uniPerm, postLabel, x)
            probYis0[x] = yis0
            probYis1[x] = yis1
        return probYis0, probYis1

    def explore(self, postJoint, probeX, mode):
        """ choose probe x via active learning given postJoint
            currently, probeX cannot repeat obsX """
        postLabel = model.posteriorLabel(self, postJoint)
        oldPostHypo = model.posteriorHypo(postJoint)
        score = np.zeros(model.nx)
        probeY = [0., 1.]
        for probex in probeX:
            yis0, yis1 = model.predicty(self.uniPerm, postLabel, probex)
            for probey in probeY:
                if probey==0:
                    predPy = yis0
                elif probey==1:
                    predPy = yis1
                newJoint = model.updatePosteriorJoint(self, [probex], [probey], postJoint)
                newPostHypo = model.posteriorHypo(newJoint)
                score[probex] += predPy*model.objective(oldPostHypo, newPostHypo, mode)
        return score

    @staticmethod
    def objective(oldPost, newPost, mode):
        if mode is 'prob_gain':
            return np.absolute(oldPost-newPost).max()
        elif mode is 'prob_total_change':
            return np.absolute(oldPost-newPost).sum()
        elif mode is 'prob_max':
            return newPost.max()
        elif mode is 'info_max':
            return entropy(oldPost) - entropy(newPost)

    def initHypoProbeMatrix(self, postJoint, probeX):
        """ initialize hypothesis-probe matrix with expected updated hyothesis posterior """
        # explore, this, and updatHypoProbeMatrix share repeated bits...
        hypoProbeM = np.zeros([model.nhypo, model.nx])
        probeY = [0., 1.]
        #postHypo = model.posteriorHypo(postJoint)
        for ihypo in range(model.nhypo):
            postLabel = model.posteriorLabelGivenHypo(self, postJoint, ihypo)
            for probex in probeX:
                yis0, yis1 = model.predicty(self.uniPerm, postLabel, probex)
                for probey in probeY:
                    if probey==0:
                        predPy = yis0
                    elif probey==1:
                        predPy = yis1
                    newJoint = model.updatePosteriorJoint(self,
                               [probex], [probey], postJoint)
                    newPostHypo = model.posteriorHypo(newJoint)
                    hypoProbeM[ihypo,probex] += predPy*newPostHypo[ihypo]
        #hypo distr not normalized
        #print("Sum hypoProbeM over hypo: %s" %(np.sum(hypoProbeM, axis=0)))
        return hypoProbeM

    def updateHypoProbeMatrix(self, postJoint, hypoProbeM, probeX):
        newM = np.zeros([model.nhypo, model.nx])
        probeY = [0., 1.]
        # postHypo = model.posteriorHypo(postJoint)
        for ihypo in range(model.nhypo):
            postLabel = model.posteriorLabelGivenHypo(self, postJoint, ihypo)
            for probex in probeX:
                yis0, yis1 = model.predicty(self.uniPerm, postLabel, probex)
                for probey in probeY:
                    if probey==0:
                        predPy = yis0
                    elif probey==1:
                        predPy = yis1
                    update = model.updatePosteriorJointWithTeacher(self,
                             [probex], [probey], postJoint, hypoProbeM[:,probex])
                    newPostHypo = model.posteriorHypo(update)
                    newM[ihypo,probex] += predPy*newPostHypo[ihypo]
        return newM

    # refactor: the 4 functions below repeat each other a lot
    @staticmethod
    def iterateNorSimple(hypoProbeM, postHypo, alpha):
        M = deepcopy(hypoProbeM)
        M = np.power(M, alpha)
        M = normalizeRow(M) # teacher's normalization
        M = M * postHypo[:, np.newaxis]
        M = normalizeCol(M)
        return M, np.array_equal(hypoProbeM, M)

    @staticmethod
    def iterSimpleTilConverge(hypoProbeM, postHypo, alpha):
        maxIter = 14
        count = 0
        stopFlag = False
        hypoProbeM, stopFlag = model.iterateNorSimple(
            hypoProbeM, postHypo, alpha)
        while (not stopFlag):
            hypoProbeM, stopFlag = model.iterateNorSimple(
                hypoProbeM, postHypo, alpha)
            count += 1
            print('Iter at step %s' %(count))
            if count==maxIter:
                print('maxIter reached but not converged yet')
                break
        return hypoProbeM

    def iterateNor(self, postJoint, hypoProbeM, probeX, alpha):
        M = deepcopy(hypoProbeM)
        M = np.power(M, alpha)
        M = normalizeRow(M) # teacher's normalization
        M = model.updateHypoProbeMatrix(self, postJoint, M, probeX)
        M = normalizeCol(M)
        return M, np.array_equal(hypoProbeM, M)

    def iterTilConverge(self, postJoint, hypoProbeM, probeX, alpha):
        maxIter = 7
        count = 0
        stopFlag = False
        hypoProbeM, stopFlag = model.iterateNor(self,
            postJoint, hypoProbeM, probeX, alpha)
        while (not stopFlag):
            hypoProbeM, stopFlag = model.iterateNor(self,
                postJoint, hypoProbeM, probeX, alpha)
            count += 1
            print('Iter at step %s' %(count))
            if count==maxIter:
                print('maxIter reached but not converged yet')
                break
        return hypoProbeM

    @staticmethod
    def teachingChoice(hypoProbeM, ihypo):
        x = randDistreteSample(normalize(hypoProbeM[ihypo,:]))
        probXHypo = hypoProbeM[:,x]
        return x, probXHypo
