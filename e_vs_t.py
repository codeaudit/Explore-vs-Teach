"""Exploration vs Teaching"""
from itertools import chain
from copy import deepcopy
from scipy.stats import entropy

from utils import normalize
from utils import flatten
from utils import makeZero
from utils import normalizeRow
from utils import normalizeCol
from utils import normalizeRowin3D
from utils import max_thresh_row
from utils import randDiscreteSample
from utils import uniformSampleMaxInd

from utils_pattern import findIndexPerm
from utils_pattern import permSet

import numpy as np
import warnings


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
    nx = 0 # used

    def __init__(self, perm):
        """ Input should have the form: perm[ihypo][iconfig] """
        self.max_mode = "hardmax"
        self.alpha = 10
        self.look_ahead = "one-step"

        self.perm = perm
        model.nhypo = len(perm)
        model.nx = len(perm[0][0])
        model.obsY = [0., 1.] # hard-wired binary setting
        model.ny = len(model.obsY)
        self.nperm = [len(p) for p in self.perm]

        model.initialize(self)

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

    def initialize(self):
        model.initPriorHypo(self)
        model.initPriorLabelGivenHypo(self) #usage:[ihypo][iconfig]
        model.initPosteriorJoint(self) #usage:[ihypo][iconfig]

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
        # Normalized? No. Does it matter? No, because postHypo and postLabel are.
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
                if probey == 0:
                    predPy = yis0
                elif probey == 1:
                    predPy = yis1
                newJoint = model.updatePosteriorJoint(self, [probex], [probey], postJoint)
                newPostHypo = model.posteriorHypo(newJoint)
                score[probex] += predPy*model.objective(oldPostHypo, newPostHypo, mode)
        return score


    @staticmethod
    def objective(oldPost, newPost, mode):
        if mode is 'prob_gain':
            return np.absolute(oldPost-newPost).max()
        elif mode is 'nelsos_prob_gain':
            # not robust because can be negative
            return newPost.max() - oldPost.max()
        elif mode is 'prob_total_change':
            return np.absolute(oldPost-newPost).sum()
        elif mode is 'prob_max':
            return newPost.max()
        elif mode is 'info_max':
            # FIXME: infs and negatives
            value = entropy(oldPost) - entropy(newPost)
            if value < 0: # this happens when XXX
                value = 0.
            if np.isinf(value): # happens for counterfactual Post =[0,0,0,0]
                value = 0.
            # print("value =", value)
            return value


    @staticmethod
    def explore_choice(score, probeX):
        """ choose unvisted x with the highest score """
        new_score = np.zeros_like(score)
        new_score[probeX] = score[probeX]
        if np.isclose(np.sum(new_score), 0):
            new_score[probeX] += 1
        # print("new_score =", new_score)
        x = uniformSampleMaxInd(new_score)
        return x

    # ###########################################################################
    # This section follows e-vs-t-v1 (4)-(6), which may not be perfectly right
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
                    if probey == 0:
                        predPy = yis0
                    elif probey == 1:
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
                    if probey == 0:
                        predPy = yis0
                    elif probey == 1:
                        predPy = yis1
                    update = model.updatePosteriorJointWithTeacher(self,
                             [probex], [probey], postJoint, hypoProbeM[:,probex])
                    newPostHypo = model.posteriorHypo(update)
                    newM[ihypo,probex] += predPy*newPostHypo[ihypo]
        return newM


    def iterate_once(self, postJoint, hypoProbeM, probeX):

        M = deepcopy(hypoProbeM)

        # TODO: understand why different normalization order give different result!
        M = normalizeCol(M) # normalization along hypo

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

        # flag_same = np.array_equal(hypoProbeM, M) #bad!
        flag_same = np.allclose(hypoProbeM, M)

        return M, flag_same


    def iterate_til_converge(self, postJoint, hypoProbeM, probeX):
        if (self.max_mode == "hardmax"):
            maxIter = 3
        elif (self.max_mode == "softmax"):
            if (self.look_ahead == "one-step"):
                maxIter = 50
            elif (self.look_ahead =="zero-step"):
                maxIter = 30
        count = 0
        stopFlag = False
        while (not stopFlag):
            hypoProbeM, stopFlag = model.iterate_once(self, postJoint, hypoProbeM, probeX)
            count += 1
            # print('Iter at step %s' %(count))
            if count == maxIter:
                print('maxIter reached but not converged yet')
                break
        return hypoProbeM
    # ###########################################################################

    # ###########################################################################
    # # This section tries a new formulation, but does not produce good results
    # # The new formulation may be the one in the comments in the e-vs-t paper
    # def get_hypoProbeObsMatrix(self, postJoint, probeX):
    #     hypoProbeObsM = model.initHypoProbeObsMatrix(self, postJoint, probeX)
    #     hypoProbeObsM = model.iterate_til_converge(self, postJoint, hypoProbeObsM, probeX)
    #     return hypoProbeObsM
    #
    # def initHypoProbeObsMatrix(self, postJoint, probeX):
    #     """ initialize hypo-probex-obsy matrix """
    #     hypoProbeObsM = np.zeros([model.nhypo, model.nx, model.ny])
    #     for probex in probeX:
    #         for indy, obsy in enumerate(model.obsY):
    #             newJoint = model.updatePosteriorJoint(self,
    #                                                 [probex], [obsy], postJoint)
    #             newPostHypo = model.posteriorHypo(newJoint)
    #             for ihypo in range(model.nhypo):
    #                 hypoProbeObsM[ihypo, probex, indy] = newPostHypo[ihypo]
    #     return hypoProbeObsM # this is normalize along hypo becaue of posteriorHypo
    #
    # def updateHypoProbeObsMatrix(self, postJoint, hypoProbeM, probeX):
    #     """ update hypo-probex-obsy matrix with teacher's likelihood hypoProbeM """
    #     hypoProbeObsM = np.zeros([model.nhypo, model.nx, model.ny])
    #     for probex in probeX:
    #         for indy, obsy in enumerate(model.obsY):
    #             update = model.updatePosteriorJointWithTeacher(self,
    #                      [probex], [obsy], postJoint, hypoProbeM[:,probex])
    #             newPostHypo = model.posteriorHypo(update)
    #             for ihypo in range(model.nhypo):
    #                 hypoProbeObsM[ihypo, probex, indy] = newPostHypo[ihypo]
    #     return hypoProbeObsM # this is normalize along hypo becaue of posteriorHypo
    #
    # def predMargY(self, postJoint, hypoProbeObsM, probeX):
    #     """ marginalize over obsy in hypo-probex-obsy matrix
    #         using the predictive distribution of obsy """
    #     hypoProbeM = np.zeros([model.nhypo, model.nx])
    #     for ihypo in range(model.nhypo):
    #         postLabel = model.posteriorLabelGivenHypo(self, postJoint, ihypo)
    #         for probex in probeX:
    #             yis0, yis1 = model.predicty(self.uniPerm, postLabel, probex)
    #             for indy, obsy in enumerate(model.obsY):
    #                 if obsy == 0:
    #                     predPy = yis0
    #                 elif obsy == 1:
    #                     predPy = yis1
    #                 hypoProbeM[ihypo, probex] += predPy*hypoProbeObsM[ihypo, probex, indy]
    #     return hypoProbeM
    #
    # def iterate_teacher(self, postJoint, M_3, probeX):
    #     """ M_3 is 3-d array (hypoProbeObsM); M_2 is 2-d array (hypoProbeM)
    #         input M_3, outputs M_2 """
    #     M_3 = deepcopy(M_3)
    #     M_3 = normalizeRowin3D(M_3) # teacher's normalization along probe before marginalization
    #     M_2 = model.predMargY(self, postJoint, M_3, probeX)
    #     if (self.max_mode == "softmax"):
    #         M_2 = np.power(M_2, self.alpha)
    #     elif (self.max_mode == "hardmax"):
    #         M_2 = max_thresh_row(M_2)
    #     M_2 = normalizeRow(M_2) # normalize along probe again after predMargy
    #     return M_2
    #
    # def iterate_learner(self, postJoint, M_2, probeX):
    #     """ M_3 is 3-d array (hypoProbeObsM); M_2 is 2-d array (hypoProbeM)
    #         input M_2, outputs M_3 """
    #     # TODO: Does not work with look_ahead = zero-step!
    #     if (self.look_ahead == "one-step"):
    #         M_3 = model.updateHypoProbeObsMatrix(self, postJoint, M_2, probeX) # this function already produces hypo-normalized array
    #     elif (self.look_ahead == "zero-step"):
    #         warnings.warn('Cannot use look_ahead: zero-step!')
    #     return M_3
    #
    # def iterate_once(self, postJoint, hypoProbeObsM, probeX):
    #     """ M_3 is 3-d array (hypoProbeObsM); M_2 is 2-d array (hypoProbeM) """
    #     # TODO: Does not work with look_ahead = zero-step!
    #     M_3 = deepcopy(hypoProbeObsM)
    #     M_2 = model.iterate_teacher(self, postJoint, M_3, probeX)
    #     M_3 = model.iterate_learner(self, postJoint, M_2, probeX)
    #     flag_same = np.allclose(hypoProbeObsM, M_3)
    #     return M_3, flag_same
    #
    # def iterate_til_converge(self, postJoint, hypoProbeObsM, probeX):
    #     if (self.max_mode == "hardmax"):
    #         maxIter = 5 # 3
    #     elif (self.max_mode == "softmax"):
    #         if (self.look_ahead == "one-step"):
    #             maxIter = 50
    #         elif (self.look_ahead =="zero-step"):
    #             maxIter = 30
    #     count = 0
    #     stopFlag = False
    #     while (not stopFlag):
    #         hypoProbeObsM, stopFlag = model.iterate_once(self, postJoint, hypoProbeObsM, probeX)
    #         count += 1
    #         # print('Iter at step %s' %(count))
    #         if count == maxIter:
    #             print('maxIter reached but not converged yet')
    #             break
    #     return hypoProbeObsM
    # ###########################################################################

    @staticmethod
    def teachingChoice(hypoProbeM, ihypo, probeX):
        """ 2nd line injects reserve probability to
            i) avoid contradiction with observations
            ii) avoid revisiting when all of M = 0 """
        M = deepcopy(hypoProbeM)
        M[:,probeX] += 1e-6
        # x = randDiscreteSample(normalize(M[ihypo,:])) # expect only one kind of values
        x = uniformSampleMaxInd(M[ihypo,:])
        probXHypo = normalize(M[:,x])
        return x, probXHypo
