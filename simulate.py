""" Simulating exploration and teaching in pattern-concept learning """
from copy import deepcopy

from utils import normalize
from utils import uniformSampleMaxInd
from utils import randDiscreteSample
from utils import perturbDistr

from vis_utils import plot4x4
from vis_utils import overlayX
from vis_utils import plotHorLines

import matplotlib.pyplot as plt
import numpy as np

def explore(learner, ihypo, iconfig, showFlag):
    Xfull = learner.x
    Xd = []
    Yd = []
    perf = np.empty(learner.nx+1)
    perf[:] = np.NAN
    terminalVals = learner.getPossPostVals()

    for step in range(len(perf)):

        learner.postHypo = learner.posteriorHypo(learner.postJoint)
        perf[step] = learner.postHypo[ihypo]

        # terminate early to avoid error in what?
        if  learner.postHypo[ihypo] in terminalVals:
            perf[step+1:] = learner.postHypo[ihypo]
            print("perf is %s" %(perf))
            break

        probeX = np.delete(Xfull, Xd)
        score = learner.explore(learner.postJoint, probeX, 'prob_gain')
        xnext = uniformSampleMaxInd(score)
        ynext = learner.gety(learner.perm[ihypo][iconfig], xnext)
        Xd.append(xnext)
        Yd.append(ynext)

        learner.postJoint = learner.posteriorJoint(Xd, Yd)

        if showFlag:
            print("step %s" %(step))
            print("P(h|D) = %s" %(learner.postHypo))
            print("Terminal Values = %s" %(terminalVals))
            print("score = %s" %(score))

            ax = plt.subplot(1, 3, 1)
            Yfull = [learner.gety(learner.perm[ihypo][iconfig], Xfull[i])
                     for i in range(len(Xfull))]
            plot4x4(Yfull)
            overlayX(Xd[:step])
            ax.set_title('Truth (%s-%s) & X' %(ihypo, iconfig))

            ax = plt.subplot(1, 3, 2)
            plot4x4(score)
            overlayX(Xd)
            ax.set_title('Score')

            ax = plt.subplot(1, 3, 3)
            plt.bar(np.arange(learner.nhypo), learner.postHypo)
            plotHorLines(terminalVals, [0, 3])

            plt.show()

    return perf


def initialize_simulation():
    Xd = []
    Yd = []
    perf = []
    Xfull = np.arange(16)
    return Xd, Yd, perf, Xfull


def reached_1_or_0(value):
    eps = 1e-6
    if ((1 - value) < eps) or (value < eps):
        return True
    else:
        return False


def teach(pair, ihypo, iconfig, showFlag):

    max_step = 5
    Xd, Yd, perf, Xfull = initialize_simulation()
    postJoint = pair.postJoint

    for step in range(max_step):

        postHypo = pair.posteriorHypo(postJoint)
        perf.append(postHypo[ihypo])
        if reached_1_or_0(postHypo[ihypo]):
            break

        probeX = np.delete(Xfull, Xd)
        hypoProbeM = pair.get_hypoProbeMatrix(postJoint, probeX)
        xnext, probXHypo = pair.teachingChoice(hypoProbeM, ihypo)
        ynext = pair.gety(pair.perm[ihypo][iconfig], xnext)
        postJoint = pair.updatePosteriorJointWithTeacher(
            [xnext], [ynext], postJoint, probXHypo)

        Xd.append(xnext)
        Yd.append(ynext)

        if showFlag:
            print("step %s" %(step))
            print("P_T(x|h) = %s" %(probXHypo))
            print("Before update P(h|D) = %s" %(postHypo))
            print("After update P(h|D) = %s" %(pair.posteriorHypo(postJoint)))
            print("score = %s" %(hypoProbeM[ihypo]))

            ax = plt.subplot(1, 3, 1)
            Yfull = [pair.gety(pair.perm[ihypo][iconfig], Xfull[i])
                     for i in range(len(Xfull))]
            plot4x4(Yfull)
            overlayX(Xd[:step])
            ax.set_title('Truth (%s-%s) & X' %(ihypo, iconfig))

            ax = plt.subplot(1, 3, 2)
            plot4x4(hypoProbeM[ihypo])
            overlayX(Xd)
            ax.set_title('Score & Next x')

            ax = plt.subplot(1, 3, 3)
            plt.bar(np.arange(pair.nhypo), postHypo)

            plt.show()

    print("Perf = %s" %(perf))
    return perf


def teacher_learner_interaction(learner, ihypo, iconfig, teacher, showFlag):

    max_step = 10

    Xd, Yd, perf, Xfull = initialize_simulation()
    postJoint = learner.postJoint
    _postJoint = teacher.postJoint #leading underscore means the teacher

    for step in range(max_step):

        postHypo = learner.posteriorHypo(postJoint)
        perf.append(postHypo[ihypo])
        if reached_1_or_0(postHypo[ihypo]):
            break

        probeX = np.delete(Xfull, Xd)
        # teacher chooses xnext
        _hypoProbeM = teacher.get_hypoProbeMatrix(_postJoint, probeX)
        xnext, _probXHypo = teacher.teachingChoice(_hypoProbeM, ihypo)
        # learner learns
        hypoProbeM = learner.get_hypoProbeMatrix(postJoint, probeX)
        probXHypo = normalize(hypoProbeM[:, xnext] + 1e-6)
        ynext = learner.gety(learner.perm[ihypo][iconfig], xnext)
        postJoint = learner.updatePosteriorJointWithTeacher(
            [xnext], [ynext], postJoint, probXHypo)
        #teacher updates learner
        _postJoint = teacher.updatePosteriorJointWithTeacher(
            [xnext], [ynext], _postJoint, _probXHypo)

        Xd.append(xnext)
        Yd.append(ynext)

        if showFlag:
            print("step %s" %(step))
            print("P_T(x|h) = %s" %(probXHypo))
            print("Before update P(h|D) = %s" %(postHypo))
            print("After update P(h|D) = %s" %(learner.posteriorHypo(postJoint)))
            print("score = %s" %(hypoProbeM[ihypo]))

            ax = plt.subplot(1, 3, 1)
            Yfull = [learner.gety(learner.perm[ihypo][iconfig], Xfull[i])
                     for i in range(len(Xfull))]
            plot4x4(Yfull)
            overlayX(Xd[:step])
            ax.set_title('Truth (%s-%s) & X' %(ihypo, iconfig))

            ax = plt.subplot(1, 3, 2)
            plot4x4(hypoProbeM[ihypo])
            overlayX(Xd)
            ax.set_title('Score & Next x')

            ax = plt.subplot(1, 3, 3)
            plt.bar(np.arange(learner.nhypo), postHypo)

            plt.show()

    print("Perf = %s" %(perf))
    return perf


def prior_learning(person):
    x = np.arange(16)
    expFreq = []
    expOptPerf = []
    expProbMatchPerf = []
    for ihypo in range(person.nhypo):
        for iconfig in range(person.nperm[ihypo]):
            perm = person.perm[ihypo][iconfig]
            y = [person.gety(perm, x[i]) for i in range(len(x))]
            postJoint = person.posteriorJoint(x, y)
            postHypo = person.posteriorHypo(postJoint)
            expFreq.append(person.priorHypo[ihypo]*
                person.priorLabelGivenHypo[ihypo][iconfig])
            if np.argmax(postHypo) == ihypo:
                expOptPerf.append(1.)
            else:
                expOptPerf.append(0.)
            expProbMatchPerf.append(postHypo[ihypo])
    expFreq = np.array(expFreq)
    expOptPerf = np.array(expOptPerf)
    expProbMatchPerf = np.array(expProbMatchPerf)
    print("Expected performance for perfect teacher-learner interaction = %s"
        %(np.sum(expFreq)))
    print("Expected performance for optimal decision = %s"
        %(np.sum(expFreq*expOptPerf)))
    print("Expected performance for prob-matching decision = %s"
        %(np.sum(expFreq*expProbMatchPerf)))
    print("Expected performance for random decision = 1/3") #hard-wired


def trial(person, Xobs, Yobs, ihypo, iconfig, showFlag):
    perf = np.empty(16)
    perf[:] = np.NAN
    perf[0] = person.priorHypo[ihypo]
    person.postJoint = person.posteriorJoint([Xobs[0]], [Yobs[0]])

    for step in range(1,16):
        updateJoint = person.updatePosteriorJoint(
                      [Xobs[step]],[Yobs[step]], person.postJoint)
        person.postJoint = deepcopy(updateJoint)
        postHypo = person.posteriorHypo(person.postJoint)
        perf[step] = postHypo[ihypo]

        if showFlag:
            print(postHypo)
            person.postJoint = person.posteriorJoint(Xobs[:step+1], Yobs[:step+1])
            checkHypo = person.posteriorHypo(person.postJoint)
            print(checkHypo)

            X = np.arange(16)
            Y = [person.gety(person.perm[ihypo][iconfig], X[i])
                 for i in range(len(X))]

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


def trials(person, ntr):

    #Xbench = np.arange(16)
    #Xbench = np.array([5,6,10,9,8,4,0,1,2,3,7,11,15,14,13,12])
    Xbench = np.array([5,6,10,9,0,12,15,3,2,11,13,4,8,1,7,14])

    perfRand = np.empty([ntr, 16])
    perfRand[:] = np.NAN
    perfBench = deepcopy(perfRand)
    perfExplore = deepcopy(perfRand)
    perfTeach = deepcopy(perfRand)
    for i in range(ntr):
        ihypo = randDistreteSample(person.priorHypo)
        iconfig = randDistreteSample(person.priorLabelGivenHypo[ihypo])

        Xrand = np.random.permutation(np.arange(16))
        Yrand = [person.gety(person.perm[ihypo][iconfig], Xrand[i])
                 for i in range(len(Xrand))]
        perfRand[i][:] = trial(person, Xrand, Yrand, ihypo, iconfig, False)

        Ybench = [person.gety(person.perm[ihypo][iconfig], Xbench[i])
                  for i in range(len(Xbench))]
        perfBench[i][:] = trial(person, Xbench, Ybench, ihypo, iconfig, False)

        perfExplore[i][:] = explore(person, ihypo, iconfig, False)

        perfTeach[i][:] = teach(person, ihypo, iconfig, False)

        print(i)

    ax = plt.subplot(1,1,1)
    plt.hold(True)
    x = np.arange(16)+1
    plt.errorbar(x, np.mean(perfRand, axis=0),
                 yerr = np.std(perfRand, axis=0)/ntr, color ='b', label='random x')
    plt.errorbar(x, np.mean(perfBench, axis=0),
                 yerr = np.std(perfBench, axis=0)/ntr, color = 'k', label='bench-mark x')
    plt.errorbar(x, np.mean(perfExplore, axis=0),
                 yerr = np.std(perfExplore, axis=0)/ntr, color = 'r', label='explore')
    plt.errorbar(x, np.mean(perfTeach, axis=0),
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


def simulate(person, ihypo, iconfig):
    X = np.random.permutation(person.x)
    # Y = [person.gety(person.perm[ihypo][iconfig], X[i]) for i in range(len(X))]

    #person.priorHypo = normalize(np.array([2.,1.,2.,20.]))
    #person.priorHypo = perturbDistr(person.priorHypo, 0.)
    for i in range(person.nhypo):
        person.priorLabelGivenHypo[i] = perturbDistr(person.priorLabelGivenHypo[i], 0)
    person.initPosteriorJoint()

    print('hypo=%s; config=%s' %(ihypo, iconfig))
    #simulate_trial(person, X, Y, ihypo, iconfig, True)
    #simulate_explore(person, ihypo, iconfig, True)
    teach(person, ihypo, iconfig, True)
