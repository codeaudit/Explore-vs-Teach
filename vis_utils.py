""" Visualization utilities for e_vs_t """
from e_vs_t import model
from copy import deepcopy
from utils import normalizeRow
from utils import normalizeCol

import numpy as np
import matplotlib.pyplot as plt


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


def vis_config():
    teacher = model()
    nhypo = teacher.nhypo
    nperm = np.array([teacher.nperm[i] for i in range(nhypo)])

    for ihypo in range(nhypo):
        for iconfig in range(nperm[ihypo]):
            if (nperm.max() > 12):
                raise ValueError("Too many configuration to plot")
            ax = plt.subplot(nhypo, nperm.max(),
                ihypo*nperm.max() + iconfig + 1)
            plot4x4(teacher.perm[ihypo][iconfig])
            ax.set_xticks([])
            ax.set_yticks([])
            print('hypo=%s, config=%s, pattern=%s'
                %(ihypo, iconfig, teacher.perm[ihypo][iconfig]))

    # with NR at last hypo
    # print("Hypo 3, config 12869, label =", teacher.perm[3][12869])
    # for iconfig in range(6):
    #     ax = plt.subplot(4,6, 3*6 + iconfig + 1)
    #     r = randDistreteSample(teacher.priorLabelGivenHypo[3])
    #     plot4x4(teacher.perm[3][r])
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    plt.show()


def vis_hypo(ihypo, iconfig):
    teacher = model()
    # ihypo = randDistreteSample(teacher.priorHypo)
    # iconfig = randDistreteSample(teacher.priorLabelGivenHypo[ihypo])
    print('hypo=%s, config=%s' %(ihypo, iconfig))

    X = np.random.permutation(np.arange(16))
    Y = [teacher.gety(teacher.perm[ihypo][iconfig], X[i])
         for i in range(len(X))]
    # print("Probed locations:", X)
    # print("Probed labels:", Y)

    teacher.postJoint = teacher.posteriorJoint(X,Y)
    teacher.postHypo = teacher.posteriorHypo(teacher.postJoint)
    print("P(h|D) = ", teacher.postHypo)

    vals = teacher.getPossPostVals()
    print("Possible values for P(h|D):", vals)
    plt.bar(np.arange(teacher.nhypo), teacher.postHypo)
    plotHorLines(vals, [0,teacher.nhypo])
    plt.show()


def vis_predict(ihypo, iconfig):
    learner = model()
    print("Done model assignment.")

    X = np.arange(16)
    Y = [learner.gety(learner.perm[ihypo][iconfig], X[i])
         for i in range(len(X))]

    Xd = np.random.permutation(X)[:3]
    Yd = [learner.gety(learner.perm[ihypo][iconfig], Xd[i])
          for i in range(len(Xd))]
    learner.postJoint = learner.posteriorJoint(Xd, Yd)
    print("Probed locations:", Xd)
    print("Probed labels:", Yd)
    # print("Done joint posterior.")

    learner.postHypo = learner.posteriorHypo(learner.postJoint)
    # print("Done hypo posterior.")

    learner.postLabel = learner.posteriorLabel(learner.postJoint)
    probYis0, probYis1 = learner.predictY(learner.uniPerm, learner.postLabel, X)
    print("P(Y=0|D) =", probYis0)
    print("P(Y=1|D) =", probYis1)
    # print("Done predicting Y.")

    ax = plt.subplot(1,3,1)
    plot4x4(Y)
    overlayX(Xd)
    ax.set_title('Truth & X')

    ax = plt.subplot(1,3,2)
    plot4x4(probYis1)
    ax.set_title('Predictive P(y=1)')

    ax = plt.subplot(1,3,3)
    plt.bar(np.arange(learner.nhypo), learner.postHypo)
    ax.set_title('Posterior P(h|X,Y)')

    plt.show()


def vis_hypoProbeMatrix(ihypo, iconfig):
    teacher = model()
    Xfull = np.arange(16)
    print('hypo=%s, iconfig=%s' %(ihypo, iconfig))
    X = np.random.permutation(Xfull)
    Y = [teacher.gety(teacher.perm[ihypo][iconfig], X[i])
         for i in range(len(X))]
    Xd = X[:1]
    Yd = Y[:1]
    print('probe location = %s' %(Xd))
    probeX = np.delete(Xfull, Xd)

    teacher.postJoint = teacher.posteriorJoint(Xd, Yd)
    hypoProbeM = teacher.initHypoProbeMatrix(teacher.postJoint, probeX)

    for i in range(5):
        print("iteration %s" %(i))
        hypoProbeM = vis_iterateNor(teacher, hypoProbeM, probeX, 10)
        plt.show()


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
    M = person.updateHypoProbeMatrix(person.postJoint, M, probeX)
    ax = plt.subplot(1,nTrans,4)
    M3 = deepcopy(M)
    plt.imshow(M3.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('update M')

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
