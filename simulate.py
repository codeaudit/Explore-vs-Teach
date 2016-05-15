""" Simulating exploration and teaching in pattern-concept learning """
from copy import deepcopy

from e_vs_t import model

from utils import uniformSampleMaxInd
from utils import randDistreteSample
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
        """ Simulation sometimes stop at step 12. Erroring on normalizing
        postHypo with prePostJoint. This suggests that the input y is not a
        feasible configuration. Another problem is all scores being 0 with
        most config in ihypo=4. A hacky fix is to terminate the process early
        when no more change in posteriror is possible. """
        learner.postHypo = learner.posteriorHypo(learner.postJoint)
        perf[step] = learner.postHypo[ihypo]
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

        """ using update gives error, probably b/c explore() already calls it.
        Should refract this so that explore doesn't change things? """
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


def teach(pair, ihypo, iconfig, showFlag, updateRule="blind"):
    alpha = 10
    maxStep = 3
    Xd = []
    Yd = []
    perf = np.empty(pair.nx+1)
    perf[:] = np.NAN

    Xfull = pair.x
    postJoint = pair.postJoint
    for step in range(maxStep):
        postHypo = pair.posteriorHypo(postJoint)
        perf[step] = postHypo[ihypo]

        if  postHypo[ihypo] == 1.:
            perf[step+1:] = postHypo[ihypo]
            print("Perf = %s" %(perf))
            break

        probeX = np.delete(Xfull, Xd)
        hypoProbeM = pair.initHypoProbeMatrix(postJoint, probeX)
        if updateRule == "simple":
            hypoProbeM = pair.iterSimpleTilConverge(hypoProbeM, postHypo, alpha)
        elif updateRule == "blind":
            hypoProbeM = pair.iterTilConverge(postJoint, hypoProbeM, probeX, alpha)
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

    return perf


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


def trials(ntr):
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


def prior_learning():
    person = model()
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

def simulate(ihypo, iconfig):
    person = model()
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
