""" Visualization utilities for e_vs_t """
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from utils import normalizeRow
from utils import normalizeCol
from utils import max_thresh_row
from utils import normalizeRowin3D

def plotHorLines(vec, lim):
    plt.hold(True)
    for el in vec:
        plt.plot(lim, [el, el], 'k:')
    plt.hold(False)

def plot_nbyn(vec):
    n = int(np.sqrt(len(vec)))
    config = np.array(vec).reshape(n,n)
    plt.imshow(config, interpolation="nearest", cmap="gray")

def overlayX(X, n):
    """ X is the data point,
        n defines the space to be drawn on, ie. n-by-n """
    plt.hold(True)
    for x in X:
        ind2d = np.unravel_index(x, (n,n))
        plt.plot(ind2d[1], ind2d[0], 'ro')
    plt.hold(False)

def vis_perm_set(perm_set):
    ncols = 6
    n = len(perm_set)
    nrows = np.ceil(n/ncols)
    for i in range(n):
        ax = plt.subplot(nrows, ncols, i+1)
        plot_nbyn(perm_set[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def vis_config(person):
    max_num_shown = 12
    nhypo = person.nhypo
    nperm = np.array([person.nperm[i] for i in range(nhypo)])
    for ihypo in range(nhypo):
        for iconfig in range(nperm[ihypo]):
            if (nperm.max() > max_num_shown):
                raise ValueError("Too many configuration to plot")
            ax = plt.subplot(nhypo, nperm.max(), ihypo*nperm.max() + iconfig + 1)
            plot_nbyn(person.perm[ihypo][iconfig])
            ax.set_xticks([])
            ax.set_yticks([])
            # print('hypo=%s, config=%s, pattern=%s'
            #     %(ihypo, iconfig, person.perm[ihypo][iconfig]))
    plt.show()


def vis_hypo(person, ihypo, iconfig):

    print('hypo=%s, config=%s' %(ihypo, iconfig))

    X = np.random.permutation(np.arange(person.nx))
    Y = [person.gety(person.perm[ihypo][iconfig], X[i])
         for i in range(len(X))]
    # print("Probed locations:", X)
    # print("Probed labels:", Y)

    person.postJoint = person.posteriorJoint(X,Y)
    person.postHypo = person.posteriorHypo(person.postJoint)
    print("P(h|D) = ", person.postHypo)

    vals = person.getPossPostVals()
    print("Possible values for P(h|D):", vals)
    plt.bar(np.arange(person.nhypo), person.postHypo)
    plotHorLines(vals, [0,person.nhypo])
    plt.show()


def vis_predict(learner, ihypo, iconfig):
    n_obs = 3
    X = np.arange(learner.nx)
    Y = [learner.gety(learner.perm[ihypo][iconfig], X[i])
         for i in range(len(X))]
    Xd = np.random.permutation(X)[:n_obs]
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
    plot_nbyn(Y)
    overlayX(Xd, int(np.sqrt(learner.nx)))
    ax.set_title('Truth & X')

    ax = plt.subplot(1,3,2)
    plot_nbyn(probYis1)
    ax.set_title('Predictive P(y=1)')

    ax = plt.subplot(1,3,3)
    plt.bar(np.arange(learner.nhypo), learner.postHypo)
    ax.set_title('Posterior P(h|X,Y)')

    plt.show()

# ###########################################################################
# # This section follows e-vs-t-v1 (4)-(6), which may not be perfectly right

def vis_hypoProbeMatrix(person, Xd, Yd):

    Xfull = np.arange(person.nx)
    probeX = np.delete(Xfull, Xd)
    person.postJoint = person.posteriorJoint(Xd, Yd)
    hypoProbeM = person.initHypoProbeMatrix(person.postJoint, probeX)

    n_step = 20
    for i in range(n_step):
        print("iteration %s" %(i))
        hypoProbeM = vis_iterateNor(person, hypoProbeM, probeX)
        plt.show()


def vis_iterateNor(person, hypoProbeM, probeX):
    nTrans = 5

    M = deepcopy(hypoProbeM)
    ax = plt.subplot(1,nTrans,1)
    M0 = deepcopy(M)
    plt.imshow(M0.transpose(), interpolation="none", cmap="gray")
    ax.set_title('input M')

    M = normalizeCol(M) # learner's normalization
    ax = plt.subplot(1,nTrans,2)
    M4 = deepcopy(M)
    plt.imshow(M4.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('nor hypo')

    alpha = 20
    M = np.power(M, alpha)
    # M = max_thresh_row(M)
    ax = plt.subplot(1,nTrans,3)
    M1 = deepcopy(M)
    plt.imshow(M1.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('softmax M')

    M = normalizeRow(M) # teacher's normalization
    ax = plt.subplot(1,nTrans,4)
    M2 = deepcopy(M)
    plt.imshow(M2.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('nor probe')

    #M = M * priorHypo[:, np.newaxis]
    M = person.updateHypoProbeMatrix(person.postJoint, M, probeX)
    ax = plt.subplot(1,nTrans,5)
    M3 = deepcopy(M)
    plt.imshow(M3.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('update M')

    print("input M:", M0)
    print("nor hypo:", M4)
    print("max thresh:", M1)
    print("nor probe:", M2)
    print("update M:", M3)

    if np.array_equal(hypoProbeM, M):
        print('Converged!')
    else:
        print('Not converging yet.')
        #print('input M = %s' %(hypoProbeM))
        #print('output M = %s' %(M))

    return M
# ###########################################################################

# ###########################################################################
# # This section tries a new formulation, but does not produce good results
# # The new formulation may be the one in the comments in the e-vs-t paper
# def vis_hypoProbeObsMatrix(person, Xd, Yd):
#
#     Xfull = np.arange(person.nx)
#     probeX = np.delete(Xfull, Xd)
#     person.postJoint = person.posteriorJoint(Xd, Yd)
#     hypoProbeObsM = person.initHypoProbeObsMatrix(person.postJoint, probeX)
#
#     n_step = 20
#     for i in range(n_step):
#         print("iteration %s" %(i))
#         hypoProbeObsM = vis_iterateNor_v2(person, hypoProbeObsM, probeX)
#         plt.show()
#
#
# def vis_iterateNor_v2(person, hypoProbeObsM, probeX):
#     nTrans = 5
#
#     # the overall sequence:
#     # M_3 = normalizeRowin3D(M_3) # teacher's normalization along probe before marginalization
#     # M_2 = model.predMargY(self, postJoint, M_3)
#     # M_2 = max_thresh_row(M_2)
#     # M_2 = normalizeRow(M_2) # normalize along probe again after predMargy
#     # M_3 = model.updateHypoProbeObsMatrix(self, postJoint, M_2, probeX) # this function already produces hypo-normalized array
#
#     M = deepcopy(hypoProbeObsM)
#     ax = plt.subplot(1,nTrans,1)
#     M_plot1 = deepcopy(np.sum(M, axis=2))
#     plt.imshow(M_plot1.transpose(), interpolation="none", cmap="gray")
#     ax.set_title('input M summed over y') # should look normalized along hypo
#
#     M = normalizeRowin3D(M) # teacher's normalization along probe
#     M = person.predMargY(person.postJoint, M, probeX)
#     ax = plt.subplot(1,nTrans,2)
#     M_plot2 = deepcopy(M)
#     plt.imshow(M_plot2.transpose(), interpolation="nearest", cmap="gray")
#     ax.set_title('predictive marginalization of y')
#
#     M = max_thresh_row(M)
#     ax = plt.subplot(1,nTrans,3)
#     M_plot3 = deepcopy(M)
#     plt.imshow(M_plot3.transpose(), interpolation="nearest", cmap="gray")
#     ax.set_title('max thresh M')
#
#     M = normalizeRow(M) # normalize along probe again after predMargy
#     ax = plt.subplot(1,nTrans,4)
#     M_plot4 = deepcopy(M)
#     plt.imshow(M_plot4.transpose(), interpolation="nearest", cmap="gray")
#     ax.set_title('normalize probe after max')
#
#     M = person.updateHypoProbeObsMatrix(person.postJoint, M, probeX)
#     ax = plt.subplot(1,nTrans,5)
#     M_plot5 = deepcopy(M.sum(axis=2))
#     plt.imshow(M_plot5.transpose(), interpolation="nearest", cmap="gray")
#     ax.set_title('update M')
#
#     print("input M summed over y:", M_plot1)
#     print("predictive marginalization of y:", M_plot2)
#     print("max thresh:", M_plot3)
#     print("normalize probe after max:", M_plot4)
#     print("update M:", M_plot5)
#     print("input M:", hypoProbeObsM)
#     print("output M:", M)
#
#     if np.allclose(hypoProbeObsM, M):
#         print('Converged!')
#     else:
#         print('Not converging yet.')
#         #print('input M = %s' %(hypoProbeM))
#         #print('output M = %s' %(M))
#
#     return M
# ###########################################################################
