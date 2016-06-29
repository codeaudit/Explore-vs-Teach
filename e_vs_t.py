"""Exploration vs Teaching"""
from itertools import chain
from copy import deepcopy
from scipy.stats import entropy

from utils import normalize
from utils import flatten
from utils import makeZero
from utils import normalizeRow
from utils import normalizeCol
from utils import max_thresh_row
from utils import randDiscreteSample

from utils_pattern import NRConfiguration
from utils_pattern import findIndexPerm
from utils_pattern import permSet
from utils_pattern import genMasterPermSet

import numpy as np

"""
2016-03-10
-refactore so that class within class becomes list of lists
-refactored so that no distribution changes inside any method
2016-04-08
-in initHypoProbeMatrix and updateHypoProbeMatrix, should not use
"if postHypo[ihypo] > 0.:" before model.posteriorLabelGivenHypo
to skip hypo, because we eventually want P(x|h).
"""

class model:
    """ define a person's hypothesis space and hierarchical prior probability """
    nhypo = 0 #4
    x = np.arange(16)
    nx = len(x)

    def __init__(self, perm):
        """ Input should have the form: perm[ihypo][iconfig] """
        self.max_mode = "hardmax"
        self.alpha = 10
        self.look_ahead = "one-step"

        self.perm = perm
        model.nhypo = len(perm)
        self.nperm = [len(p) for p in self.perm]

        model.initPriorHypo(self)
        model.initPriorLabelGivenHypo(self) #usage:[ihypo][iconfig]
        model.initPosteriorJoint(self) #usage:[ihypo][iconfig]

        self.uniPerm = permSet(perm)
        self.nUniPerm = len(self.uniPerm)
        self.permId = model.idPerm(self) #usage:[ihypo][iconfig]

    def change_mode(self, max_mode, alpha, look_ahead):
        self.max_mode = max_mode
        self.alpha = alpha
        self.look_ahead = look_ahead

    @staticmethod
    def check_perm_length(perm):
        if len(perm) != model.nhypo:
            raise ValueError("length of perm must be", model.nhypo)

    def idPerm(self):
        """ assign an id to each permutation relative to unique perm"""
        allPerm = self.uniPerm
        permId = [[findIndexPerm(allPerm, self.perm[ihypo][iconfig])
            for iconfig in range(self.nperm[ihypo])]
                for ihypo in range(model.nhypo)]
        return permId

    def getPossPostVals(self):
        """ get possible posterior values for full observations """
        nperm = [self.nperm[ihypo] for ihypo in range(model.nhypo)]
        prob = np.multiply(np.divide(np.ones(model.nhypo), nperm), self.priorHypo)
        post = []
        for ihypo in range(model.nhypo):
            inds = list(np.arange(ihypo))
            post.append(normalize(makeZero(prob, inds)))
        possVals = flatten(post)
        return set(possVals)

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


    def get_hypoProbeMatrix(self, postJoint, probeX):
        hypoProbeM = model.initHypoProbeMatrix(self, postJoint, probeX)
        hypoProbeM = model.iterate_til_converge(self, postJoint, hypoProbeM, probeX)
        return hypoProbeM


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


    def iterate_once(self, postJoint, hypoProbeM, probeX):
        M = deepcopy(hypoProbeM)

        if (self.max_mode == "softmax"):
            M = np.power(M, self.alpha)
        elif (self.max_mode == "hardmax"):
            M = max_thresh_row(M)

        M = normalizeRow(M) # teacher's normalization along probe

        if (self.look_ahead == "one-step"):
            M = model.updateHypoProbeMatrix(self, postJoint, M, probeX)
        elif (self.look_ahead == "zero-step"):
            postHypo = model.posteriorHypo(postJoint)
            M = M*postHypo[:, np.newaxis]

        M = normalizeCol(M) # normalization along hypo

        return M, np.array_equal(hypoProbeM, M)


    def iterate_til_converge(self, postJoint, hypoProbeM, probeX):
        if (self.max_mode == "hardmax"):
            maxIter = 3
        elif (self.max_mode == "softmax"):
            if (self.look_ahead == "one-step"):
                maxIter = 15
            elif (self.look_ahead =="zero-step"):
                maxIter = 30
        count = 0
        stopFlag = False
        while (not stopFlag):
            hypoProbeM, stopFlag = model.iterate_once(self, postJoint, hypoProbeM, probeX)
            count += 1
            # print('Iter at step %s' %(count))
            if count==maxIter:
                print('maxIter reached but not converged yet')
                break
        return hypoProbeM


    @staticmethod
    def teachingChoice(hypoProbeM, ihypo, probeX):
        M = deepcopy(hypoProbeM)
        M[:,probeX] += 1e-6 # inject reserve probability to avoid contradiction with observations
        x = randDiscreteSample(normalize(M[ihypo,:]))
        probXHypo = normalize(M[:,x])
        return x, probXHypo
