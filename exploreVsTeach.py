# Exploration vs teaching codes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import seaborn as sns
from unique_permutations import unique_permutations as uperm
from itertools import chain
from copy import deepcopy
from time import time
from scipy.stats import entropy

"""
2016-02-22 possible refactoring for later on:
-more generic class for hierarchical configuration
-all permulations/configurations in one function, and just assign perm to hypo from it
-posteriors: should them all take input (self, postJoint) instead of (self)?
-test functions will need to be updated
-the input and output of the methods in model are messy
"""

class compartmet_config:
    """ define a compartment configuration via position x and compartment label k.
        Can extend to multiple configurations: x, nk -> many k's. """
    def __init__(self, x, k, prob):
        self.x = x
        self.k = k
        self.nk = len(set(k))
        self.prob = prob

        if (self.nk % 2):
            raise ValueError('number of compartment should be even.')
        else:
            self.base = np.append(np.zeros(self.nk/2), np.ones(self.nk/2)).tolist()


class label_config:
    """ define all label configurations given a base label configuration """
    def __init__(self, x, k, base):
        self.basePerm = [list(item) for item in uperm(base)]
        self.nperm = len(self.basePerm)
        self.prob = np.ones(self.nperm)/self.nperm

        nbase = len(base)
        nx = len(x)
        self.perm = []
        for indp in range(self.nperm):
            self.perm.append(np.zeros(nx))
            for indb in range(nbase):
                self.perm[indp][np.where(k==indb)[0]] = self.basePerm[indp][indb]

    def gety(self, x, iconfig):
        return self.perm[iconfig][x]

    def likelihood(self, X, Y, iconfig):
        """ return likelihood of observing X and Y (vectos) given a label configuration """
        nobs = len(X)
        for i in range(nobs):
            if Y[i] != label_config.gety(self, X[i], iconfig):
                return 0.
        return 1.


class model:
    """ define a person's hypothesis space and hierarchical prior probability """
    def __init__(self):
        """ manual initialization of compartment """
        self.nhypo = 4
        self.prob = np.ones(self.nhypo)/self.nhypo

        self.x = np.arange(16)
        self.nx = len(self.x)

        self.compk = [0,0,0,0]

        self.compk[0] = np.zeros(16)
        self.compk[0][8:] = 1.

        self.compk[1] = np.zeros(16)
        self.compk[1][2::4] = 1.
        self.compk[1][3::4] = 1.

        self.compk[2] = np.zeros(16)
        self.compk[2][[2,3,6,7]] = 1.
        self.compk[2][[8,9,12,13]] = 2.
        self.compk[2][[10,11,14,15]] = 3.

        self.compk[3] = self.x

        self.comp = [0,0,0,0]
        self.label = [0,0,0,0]
        for ind in range(self.nhypo):
            self.comp[ind] = compartmet_config(self.x, self.compk[ind], self.prob[ind])
            self.label[ind] = label_config(self.x, self.compk[ind], self.comp[ind].base)
        self.ntotPerm = self.label[3].nperm

        model.initPosteriorJoint(self)
        model.idPerm(self)

    def idPerm(self):
        """ assign an id to each permutation """
        # sort of hard-wired
        allPerm = self.label[3].perm
        self.label[3].permId = np.arange(self.label[3].nperm)
        for ihypo in range(3):
            self.label[ihypo].permId = -np.ones(self.label[ihypo].nperm)
            for iconfig in range(self.label[ihypo].nperm):
                self.label[ihypo].permId[iconfig] = (
                  model.findIndexPerm(allPerm, self.label[ihypo].perm[iconfig]))

    def findIndexPerm(allItem, target):
        for ind,item in enumerate(allItem):
            if np.array_equiv(item, target):
                return ind

    def getPossPostVals(self):
        """ get possible posterior values for full observations """
        nperm = [self.label[ind].nperm for ind in range(self.nhypo)]
        prob = np.multiply(np.divide(np.ones(self.nhypo), nperm), self.prob)
        post = [normalize(makeZero(prob, [0])),
                normalize(makeZero(prob, [1])),
                normalize(makeZero(prob, [0,1])),
                normalize(makeZero(prob, [0,1,2]))]
        return set(chain.from_iterable(post))

    def initPosteriorJoint(self):
        self.postJoint = [
            [self.label[ihypo].prob[ilabel]*self.prob[ihypo]
            for ilabel in range(self.label[ihypo].nperm)]
            for ihypo in range(self.nhypo)]

    def posteriorJoint(self, X, Y):
        """ compute posterior of label, hypo jointly given observations X and Y """
        """ Normalized? No. Does it matter? Probably not because postHypo and postLabel are"""
        self.postJoint = [
            [self.label[ihypo].likelihood(X, Y, ilabel)
            *self.label[ihypo].prob[ilabel]
            *self.prob[ihypo]
            for ilabel in range(self.label[ihypo].nperm)]
            for ihypo in range(self.nhypo)]
        #print("Sum of postJoint = %s" %(np.sum(flatten(self.postJoint))))

    def updatePosteriorJoint(self, x, y):
        """ update joint with one new observation pair, unnormalized """
        """ these functions that update the joint should be the computational bottle neck """
        update = deepcopy(self.postJoint)
        for ihypo in range(self.nhypo):
            for ilabel in range(self.label[ihypo].nperm):
                update[ihypo][ilabel] *= self.label[ihypo].likelihood(x, y, ilabel)
        return update

    def updatePosteriorJointWithTeacher(self, x, y, probXHypo):
        """ update joint with one new observation pair & teacher's choice prob, unnormalized """
        update = deepcopy(self.postJoint)
        for ihypo in range(self.nhypo):
            for ilabel in range(self.label[ihypo].nperm):
                update[ihypo][ilabel] *= (
                  self.label[ihypo].likelihood(x, y, ilabel)*probXHypo[ihypo])
        return update

    def posteriorHypo(self, postJoint):
        """ compute posterior of hypo given observations X and Y """
        self.postHypo = np.zeros(self.nhypo)
        for ihypo in range(self.nhypo):
            for iconfig in range(self.label[ihypo].nperm):
                self.postHypo[ihypo] += postJoint[ihypo][iconfig]
        # short-hand that does the same thing - not correct now
        #self.postHypo = [np.sum(self.postJoint[ihypo]) for ihypo in range(self.nhypo)]
        self.postHypo = normalize(self.postHypo)
        return self.postHypo

    def posteriorLabel(self):
        self.postLabel = np.zeros(self.ntotPerm)
        for ihypo in range(self.nhypo):
            for iconfig in range(self.label[ihypo].nperm):
                idp = self.label[ihypo].permId[iconfig]
                self.postLabel[idp] += self.postJoint[ihypo][iconfig]
        self.postLabel = normalize(self.postLabel)

    def posteriorLabelGivenHypo(self, ihypo):
        self.postLabel = np.zeros(self.ntotPerm)
        for iconfig in range(self.label[ihypo].nperm):
            idp = self.label[ihypo].permId[iconfig]
            self.postLabel[idp] += self.postJoint[ihypo][iconfig]
        self.postLabel = normalize(self.postLabel)

    def predicty(self, x):
        """ compute predictive distriubtion of y for one probe x """
        """ checked: yis0 + yis1 = 1, even with posteriorLabelGivenHypo """
        yis0 = np.zeros(1)
        yis1 = np.zeros(1)
        for iconfig in range(self.ntotPerm):
            y = self.label[3].gety(x, iconfig) #hard-wired
            if y == 0:
                yis0 += self.postLabel[iconfig]
            elif y == 1:
                yis1 += self.postLabel[iconfig]
        #print("yis0 + yis1 = %s" %(yis0 + yis1))
        return yis0, yis1

    def predictY(self, X):
        """ loop over predicty for multiple probes X """
        """ to get predictY given hypo, run posteriorLabelGivenHypo then predictY """
        self.probYis0 = np.zeros(self.nx)
        self.probYis1 = np.zeros(self.nx)
        for x in X:
            yis0, yis1 = model.predicty(self, x)
            self.probYis0[x] = yis0
            self.probYis1[x] = yis1

    def explore(self, postJoint, probeX, mode):
        """ choose probe x via active learning given postJoint """
        """ currently, probeX cannot repeat obsX """
        self.postJoint = postJoint
        model.posteriorLabel(self)
        oldPostHypo = model.posteriorHypo(self, self.postJoint)
        score = np.zeros(self.nx)
        probeY = [0., 1.]
        for probex in probeX:
            yis0, yis1 = model.predicty(self, probex)
            for probey in probeY:
                if probey==0:
                    predPy = yis0
                elif probey==1:
                    predPy = yis1
                newJoint = model.updatePosteriorJoint(self, [probex], [probey])
                newPostHypo = model.posteriorHypo(self, newJoint)
                score[probex] += predPy*model.objective(oldPostHypo, newPostHypo, mode)
        return score

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
        # explore, this, and updatHypoProbeMatrix repeats a lot of each other...
        self.postJoint = postJoint
        hypoProbeM = np.zeros([self.nhypo, self.nx])
        probeY = [0., 1.]
        for ihypo in range(self.nhypo):
            # modifiy this to not go through ihypo with P(h|ihypo) = 0!
            model.posteriorLabelGivenHypo(self, ihypo)            
            for probex in probeX:
                yis0, yis1 = model.predicty(self, probex)
                for probey in probeY:
                    if probey==0:
                        predPy = yis0
                    elif probey==1:
                        predPy = yis1
                    newJoint = model.updatePosteriorJoint(self, [probex], [probey])
                    newPostHypo = model.posteriorHypo(self, newJoint)
                    hypoProbeM[ihypo,probex] += predPy*newPostHypo[ihypo]
        #print("Sum hypoProbeM over hypo: %s" %(np.sum(hypoProbeM, axis=0))) #hypo distr not normalized
        return hypoProbeM

    def updateHypoProbeMatrix(self, hypoProbeM, probeX):
        # annoying: check self.postJoint is never updated after initHypoProbeMatrix
        newM = np.zeros([self.nhypo, self.nx])
        probeY = [0., 1.]
        for ihypo in range(self.nhypo):
            model.posteriorLabelGivenHypo(self, ihypo)
            for probex in probeX:
                yis0, yis1 = model.predicty(self, probex)
                for probey in probeY:
                    if probey==0:
                        predPy = yis0
                    elif probey==1:
                        predPy = yis1
                    update = model.updatePosteriorJointWithTeacher(
                                self, [probex], [probey], hypoProbeM[:,probex])
                    newPostHypo = model.posteriorHypo(self, update)
                    newM[ihypo,probex] += predPy*newPostHypo[ihypo]
        return newM

    def iterateNorSimple(self, hypoProbeM, priorHypo, alpha):
        # Bad: repeating itenateNor
        M = deepcopy(hypoProbeM)
        M = np.power(M, alpha)
        M = normalizeRow(M) # teacher's normalization
        M = M * priorHypo[:, np.newaxis]
        M = normalizeCol(M)
        return M, np.array_equal(hypoProbeM, M)

    def iterateNor(self, hypoProbeM, probeX, alpha):
        M = deepcopy(hypoProbeM)
        M = np.power(M, alpha)
        M = normalizeRow(M) # teacher's normalization
        M = model.updateHypoProbeMatrix(self, M, probeX)
        M = normalizeCol(M)
        return M, np.array_equal(hypoProbeM, M)

    def iterTilConverge(self, hypoProbeM, vec, alpha):
        maxIter = 7
        count = 0
        hypoProbeM, stopFlag = model.iterateNor(self, hypoProbeM, vec, alpha)
        while (not stopFlag):
            hypoProbeM, stopFlag = model.iterateNor(self, hypoProbeM, vec, alpha)
            count += 1
            print('Iter at step %s.' %(count))
            if count==maxIter:
                print('maxIter reached but not converged yet.')
                break

        return hypoProbeM

    def teachingChoice(self, hypoProbeM, ihypo):
        x = randDistreteSample(normalize(hypoProbeM[ihypo,:]))
        probXHypo = hypoProbeM[:,x]
        return x, probXHypo


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
        print('input vec = %s' %(vec))
        raise ValueError('All entries should >=0 with at least one >0')
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

def plotHorLines(vec, lim):
    plt.hold(True)
    for el in vec:
        plt.plot(lim, [el, el], 'k:')
    plt.hold(False)

def plot4x4(vec):
    config = np.array(vec).reshape(4,4)
    plt.imshow(config, interpolation="nearest", cmap="gray")

def overlayX(X):
    plt.hold(True)
    for x in X:
        ind2d = np.unravel_index(x, (4,4))
        plt.plot(ind2d[1], ind2d[0], 'ro')
    plt.hold(False)


def vis_iterateNor(person, hypoProbeM, probeX, alpha):
    nTrans = 5

    M = deepcopy(hypoProbeM)
    ax =plt.subplot(1,nTrans,1)
    plt.imshow(M.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('input M')

    M = np.power(M, alpha)
    ax = plt.subplot(1,nTrans,2)
    M1 = deepcopy(M)
    plt.imshow(M1.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('softmax M')

    M = normalizeRow(M) # teacher's normalization
    ax = plt.subplot(1,nTrans,3)
    M2 = deepcopy(M)
    plt.imshow(M2.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('nor probe')

    #M = M * priorHypo[:, np.newaxis]
    M = person.updateHypoProbeMatrix(M, probeX)
    ax = plt.subplot(1,nTrans,4)
    M3 = deepcopy(M)
    plt.imshow(M3.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('M*prior')

    M = normalizeCol(M)
    ax = plt.subplot(1,nTrans,5)
    M4 = deepcopy(M)
    plt.imshow(M4.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('nor hypo')

    if np.array_equal(hypoProbeM, M):
        print('Converged!')
    else:
        print('Not converging yet.')
        #print('input M = %s' %(hypoProbeM))
        #print('output M = %s' %(M))

    return M


def model_test_config():
    teacher = model()
    print(teacher.compk[3])
    print(teacher.label[3].perm[12869])

    for ihypo in range(teacher.nhypo-1):
        for iconfig in range(teacher.label[ihypo].nperm):
            ax = plt.subplot(4,6, ihypo*6 + iconfig + 1)
            plot4x4(teacher.label[ihypo].perm[iconfig])
            ax.set_xticks([])
            ax.set_yticks([])

    for iconfig in range(6):
        ax = plt.subplot(4,6, 3*6 + iconfig + 1)
        r = randDistreteSample(teacher.label[3].prob)
        plot4x4(teacher.label[3].perm[r])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def model_test_hypo():
    teacher = model()
    for trial in range(10):
        ihypo = randDistreteSample(teacher.prob)
        iconfig = randDistreteSample(teacher.label[ihypo].prob)
        print(ihypo)
        print(iconfig)

        X = np.random.permutation(np.arange(16))
        Y = [teacher.label[ihypo].gety(X[i],iconfig) for i in range(len(X))]
        print(X)
        print(Y)

        teacher.posteriorJoint(X,Y)
        teacher.posteriorHypo(teacher.postJoint)
        print(teacher.postHypo)

        vals = teacher.getPossPostVals()
        print(vals)
        plt.bar(np.arange(4), teacher.postHypo)
        plotHorLines(vals)
        plt.show()


def model_test_predict():
    learner = model()
    print("done model assignment")

    ihypo = 2
    iconfig = 2
    X = np.arange(16)
    Y = [learner.label[ihypo].gety(X[i],iconfig) for i in range(len(X))]
    print(Y)

    Xd = np.random.permutation(X)[:3]
    Yd = [learner.label[ihypo].gety(Xd[i],iconfig) for i in range(len(Xd))]
    learner.posteriorJoint(Xd, Yd)
    print(Xd)
    print("done joint posterior")

    learner.posteriorHypo(learner.postJoint)
    print("done hypo posterior")

    learner.predictY(X)
    print(learner.probYis0)
    print(learner.probYis1)
    print("done predicting Y")

    ax = plt.subplot(2,3,1)
    plot4x4(Y)
    overlayX(Xd)
    ax.set_title('Truth & X')

    ax = plt.subplot(2,3,2)
    plot4x4(learner.probYis1)
    ax.set_title('Predictive P(y=1)')

    #plt.subplot(2,3,3)

    ax = plt.subplot(2,3,4)
    plt.bar(np.arange(4), learner.postHypo)
    ax.set_title('Posterior P(h|X,Y)')

    plt.show()


def model_test_hypoProbeMatrix():
    teacher = model()
    Xfull = np.arange(16)
    for trial in range(1):
        ihypo = randDistreteSample(teacher.prob)
        iconfig = randDistreteSample(teacher.label[ihypo].prob)
        print('hypo=%s, iconfig=%s' %(ihypo, iconfig))
        X = np.random.permutation(Xfull)
        Y = [teacher.label[ihypo].gety(X[i],iconfig) for i in range(len(X))]
        Xd = X[:1]
        Yd = Y[:1]
        probeX = np.delete(Xfull, Xd)

        teacher.posteriorJoint(Xd, Yd)
        postHypo = teacher.posteriorHypo(teacher.postJoint)
        hypoProbeM = teacher.initHypoProbeMatrix(teacher.postJoint, probeX)

        # M = teacher.iterTilConverge(hypoProbeM, postHypo, 10)
        # plt.imshow(M.transpose(), interpolation="nearest", cmap="gray")
        # plt.show()

        # ax = plt.subplot(1,2,1)
        # Yfull = [teacher.label[ihypo].gety(Xfull[i],iconfig) for i in range(len(Xfull))]
        # plot4x4(Yfull)
        # overlayX(Xd)
        # ax.set_title('Truth (%s-%s) & X' %(ihypo, iconfig))
        #
        # ax = plt.subplot(1,2,2)
        # probeX = np.delete(Xfull, Xd)
        # plt.imshow(hypoProbeM.transpose(), interpolation="nearest", cmap="gray")
        #
        # plt.show()

        for i in range(10):
            print("interation %s" %(i))
            hypoProbeM = vis_iterateNor(teacher, hypoProbeM, probeX, 10)
            plt.show()


def model_simulate_explore(learner, ihypo, iconfig, showFlag):
    Xfull = learner.x
    Xd = []
    Yd = []
    perf = np.empty(learner.nx+1)
    perf[:] = np.NAN
    terminalVals = learner.getPossPostVals()
    for step in range(len(perf)):
        """ Simulation sometimes stop at step 12. Erroring on normalizing
        postHypo with prePostJoint. This suggests that the input y is not a
        feasible configuration. Another problem is all scores being 0 with
        most config in ihypo=4. A hacky fix is to terminate the process early
        when no more change in posteriror is possible. """
        learner.posteriorHypo(learner.postJoint)
        perf[step] = learner.postHypo[ihypo]
        if  learner.postHypo[ihypo] in terminalVals:
            perf[step+1:] = learner.postHypo[ihypo]
            print("perf is %s" %(perf))
            break

        probeX = np.delete(Xfull, Xd)
        score = learner.explore(learner.postJoint, probeX, 'prob_gain')
        xnext = uniformSampleMaxInd(score)
        ynext = learner.label[ihypo].gety(xnext,iconfig)
        Xd.append(xnext)
        Yd.append(ynext)

        """ using update gives error, probably b/c explore() already calls it.
        Should refract this so that explore doesn't change things? """
        learner.posteriorJoint(Xd, Yd)

        if showFlag:
            print("step %s" %(step))
            print("P(h|D) = %s" %(learner.postHypo))
            print("Terminal Values = %s" %(terminalVals))
            print("score = %s" %(score))

            ax = plt.subplot(1,3,1)
            Yfull = [learner.label[ihypo].gety(Xfull[i],iconfig) for i in range(len(Xfull))]
            plot4x4(Yfull)
            overlayX(Xd[:step])
            ax.set_title('Truth (%s-%s) & X' %(ihypo, iconfig))

            ax = plt.subplot(1,3,2)
            plot4x4(score)
            overlayX(Xd)
            ax.set_title('Score')

            ax = plt.subplot(1,3,3)
            plt.bar(np.arange(learner.nhypo), learner.postHypo)
            plotHorLines(terminalVals, [0,3])

            plt.show()

    return perf


def model_simulate_teach(pair, ihypo, iconfig, showFlag):
    Xfull = pair.x
    alpha = 10
    Xd = []
    Yd = []
    perf = np.empty(pair.nx+1)
    perf[:] = np.NAN
    maxStep = 3 #len(perf)
    for step in range(maxStep):
        postHypo = pair.posteriorHypo(pair.postJoint)
        perf[step] = postHypo[ihypo]
        if  postHypo[ihypo] == 1.:
            perf[step+1:] = postHypo[ihypo]
            print("Perf = %s" %(perf))
            break

        probeX = np.delete(Xfull, Xd)
        vec =  probeX #postHypo
        hypoProbeM = pair.initHypoProbeMatrix(pair.postJoint, probeX)
        hypoProbeM = pair.iterTilConverge(hypoProbeM, vec, alpha)
        xnext, probXHypo = pair.teachingChoice(hypoProbeM, ihypo)
        ynext = pair.label[ihypo].gety(xnext, iconfig)
        #pair.postJoint = pair.updatePosteriorJointWithTeacher([xnext], [ynext], probXHypo)

        Xd.append(xnext)
        Yd.append(ynext)
        pair.posteriorJoint(Xd, Yd)
        print(pair.postJoint)

        if showFlag:
            print("step %s" %(step))
            print("P_T(x|h) = %s" %(probXHypo))
            print("P(h|D) = %s" %(postHypo))
            print("score = %s" %(hypoProbeM[ihypo]))

            ax = plt.subplot(1,3,1)
            Yfull = [pair.label[ihypo].gety(Xfull[i],iconfig) for i in range(len(Xfull))]
            plot4x4(Yfull)
            overlayX(Xd[:step])
            ax.set_title('Truth (%s-%s) & X' %(ihypo, iconfig))

            ax = plt.subplot(1,3,2)
            plot4x4(hypoProbeM[ihypo])
            overlayX(Xd)
            ax.set_title('Score & Next x')

            ax = plt.subplot(1,3,3)
            plt.bar(np.arange(pair.nhypo), postHypo)

            plt.show()

    return perf


def model_simulate_trial(person, Xobs, Yobs, ihypo, iconfig, showFlag):
    perf = np.empty(16)
    perf[:] = np.NAN
    perf[0] = person.prob[ihypo]
    person.posteriorJoint([Xobs[0]], [Yobs[0]])

    for step in range(1,16):
        updateJoint = person.updatePosteriorJoint([Xobs[step]],[Yobs[step]])
        person.postJoint = deepcopy(updateJoint)
        postHypo = person.posteriorHypo(person.postJoint)
        perf[step] = postHypo[ihypo]

        if showFlag:
            print(postHypo)
            person.posteriorJoint(Xobs[:step+1], Yobs[:step+1])
            checkHypo = person.posteriorHypo(person.postJoint)
            print(checkHypo)

            X = np.arange(16)
            Y = [person.label[ihypo].gety(X[i],iconfig) for i in range(len(X))]

            ax = plt.subplot(1,2,1)
            plot4x4(Y)
            overlayX(Xobs[:step+1])
            ax.set_title('Truth (%s-%s) & X' %(ihypo, iconfig))

            ax = plt.subplot(1,2,2)
            plt.plot(X, perf)
            ax.set_ylim([-0.1,1.1])
            ax.set_xlim([0,15])
            ax.set_title('Performance')
            plotHorLines(person.getPossPostVals(),[0,15])

            plt.show()

    return perf


def model_test_simulate():
    person = model()
    #ihypo = randDistreteSample(person.prob)
    #iconfig = randDistreteSample(person.label[ihypo].prob)
    ihypo = 0
    iconfig = 0
    X = np.random.permutation(person.x)
    Y = [person.label[ihypo].gety(X[i],iconfig) for i in range(len(X))]

    #person.prob = normalize(np.array([2.,1.,2.,20.]))
    #person.prob = perturbDistr(person.prob, 0.)
    for i in range(person.nhypo):
        person.label[i].prob = perturbDistr(person.label[i].prob, 0.01)
    person.initPosteriorJoint()

    print('hypo=%s; config=%s' %(ihypo, iconfig))
    #model_simulate_trial(person, X, Y, ihypo, iconfig, True)
    #model_simulate_explore(person, ihypo, iconfig, True)
    model_simulate_teach(person, ihypo, iconfig, True)


def model_simulate_trials(ntr):
    person = model()

    #Xbench = np.arange(16)
    #Xbench = np.array([5,6,10,9,8,4,0,1,2,3,7,11,15,14,13,12])
    Xbench = np.array([5,6,10,9,0,12,15,3,2,11,13,4,8,1,7,14])

    perfRand = np.empty([ntr, 16])
    perfRand[:] = np.NAN
    perfBench = deepcopy(perfRand)
    perfExplore = deepcopy(perfRand)
    perfTeach = deepcopy(perfRand)
    for i in range(ntr):
        ihypo = randDistreteSample(person.prob)
        iconfig = randDistreteSample(person.label[ihypo].prob)

        Xrand = np.random.permutation(np.arange(16))
        Yrand = [person.label[ihypo].gety(Xrand[i],iconfig) for i in range(len(Xrand))]
        perfRand[i][:] = model_simulate_trial(person, Xrand, Yrand, ihypo, iconfig, False)

        Ybench = [person.label[ihypo].gety(Xbench[i],iconfig) for i in range(len(Xbench))]
        perfBench[i][:] = model_simulate_trial(person, Xbench, Ybench, ihypo, iconfig, False)

        perfExplore[i][:] = model_simulate_explore(person, Xbench[0], Ybench[0], ihypo, iconfig, False)

        perfTeach[i][:] = model_simulate_teach(person, Xbench[0], Ybench[0], ihypo, iconfig, False)

        print(i)

    ax = plt.subplot(1,1,1)
    plt.hold(True)
    x = np.arange(16)+1
    simuRand = plt.errorbar(x, np.mean(perfRand, axis=0),
            yerr = np.std(perfRand, axis=0)/ntr, color ='b', label='random x')
    simuBench = plt.errorbar(x, np.mean(perfBench, axis=0),
            yerr = np.std(perfBench, axis=0)/ntr, color = 'k', label='bench-mark x')
    simuExplore = plt.errorbar(x, np.mean(perfExplore, axis=0),
            yerr = np.std(perfExplore, axis=0)/ntr, color = 'r', label='explore')
    simuTeach = plt.errorbar(x, np.mean(perfTeach, axis=0),
            yerr = np.std(perfTeach, axis=0)/ntr, color = 'g', label='teach')
    plt.hold(False)

    ax.set_xlabel('number of observations')
    ax.set_ylabel('P(correct hypothesis | data)')
    ax.set_xlim([0, 17])
    ax.set_ylim([0, 1.02])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.legend(loc='lower right')

    plt.show()


if __name__ == '__main__':
    #model_test_config()
    #model_test_hypo()
    #model_test_predict()
    #model_test_hypoProbeMatrix()
    model_test_simulate()
    #model_simulate_trials(100)
