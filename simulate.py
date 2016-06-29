""" Simulating exploration and teaching in pattern-concept learning """
from copy import deepcopy

from utils import normalize
from utils import uniformSampleMaxInd
from utils import perturbDistr
from utils import is_alist_in_lili

from vis_utils import plot4x4
from vis_utils import overlayX
from vis_utils import plotHorLines
from vis_utils import vis_config
from vis_utils import vis_perm_set

from utils_pattern import genMasterPermSet
from utils_pattern import check_complementarily_paired
from utils_pattern import iter_hypo_indicator
from utils_pattern import build_hypo
from utils_pattern import hypo2perm

from e_vs_t import model

import matplotlib.pyplot as plt
import numpy as np

def example_model(mode="simple"):
    perm_set = genMasterPermSet()
    perm = [0]*3
    if (mode == "simple"):
        perm[0] = perm_set[0:2]
        perm[1] = perm_set[2:4]
        perm[2] = perm_set[0:6]
    elif (mode == "full"):
        perm[0] = perm_set[0:2] + perm_set[6:10]
        perm[1] = perm_set[2:4] + perm_set[10:14]
        perm[2] = perm_set[0:6]
    return model(perm)

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
    print("Expected performance for random decision =", 1/person.nhypo)


def explore(learner, ihypo, iconfig, max_step, showFlag):

    Xd, Yd, perf, Xfull = initialize_simulation()
    terminalVals = learner.getPossPostVals()
    postJoint = learner.postJoint

    for step in range(max_step):

        postHypo = learner.posteriorHypo(postJoint)
        perf.append(postHypo[ihypo])

        # terminate early to avoid error in what?
        # if  learner.postHypo[ihypo] in terminalVals:
        #     break

        probeX = np.delete(Xfull, Xd)
        score = learner.explore(postJoint, probeX, 'prob_gain')
        xnext = uniformSampleMaxInd(score)
        ynext = learner.gety(learner.perm[ihypo][iconfig], xnext)
        Xd.append(xnext)
        Yd.append(ynext)

        postJoint = learner.posteriorJoint(Xd, Yd)

        if showFlag:
            print("step %s" %(step))
            print("P(h|D) = %s" %(postHypo))
            # print("Terminal Values = %s" %(terminalVals))
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
            plt.bar(np.arange(learner.nhypo), postHypo)
            plotHorLines(terminalVals, [0, 3])

            plt.show()

    return perf


def teach(pair, ihypo, iconfig, max_step, showFlag):

    Xd, Yd, perf, Xfull = initialize_simulation()
    postJoint = pair.postJoint

    for step in range(max_step):

        postHypo = pair.posteriorHypo(postJoint)
        perf.append(postHypo[ihypo])
        # if reached_1_or_0(postHypo[ihypo]):
        #     break

        probeX = np.delete(Xfull, Xd)
        hypoProbeM = pair.get_hypoProbeMatrix(postJoint, probeX)
        xnext, probXHypo = pair.teachingChoice(hypoProbeM, ihypo, probeX)
        ynext = pair.gety(pair.perm[ihypo][iconfig], xnext)
        postJoint = pair.updatePosteriorJointWithTeacher(
            [xnext], [ynext], postJoint, probXHypo)

        Xd.append(xnext)
        Yd.append(ynext)

        if showFlag:
            show_teach(step, ihypo, icongfig, Xd, Xfull, probeX,
                       pair, hypoProbeM, probXHypo, postHypo, postJoint)
    # print("Perf = %s" %(perf))
    return perf


def show_teach(step, ihypo, icongfig, Xd, Xfull, probeX,
               pair, hypoProbeM, probXHypo, postHypo, postJoint):
    plot_M = deepcopy(hypoProbeM)
    plot_M[:,probeX] += 1e-6
    print("step %s" %(step))
    print("P_T(x|h) = %s" %(probXHypo))
    print("Before update P(h|D) = %s" %(postHypo))
    print("After update P(h|D) = %s" %(pair.posteriorHypo(postJoint)))
    print("score = %s" %(normalize(plot_M[ihypo] + 1e-6)))

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
    # plt.bar(np.arange(pair.nhypo), postHypo)
    plt.imshow(plot_M.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('update M')

    plt.show()


def teacher_learner_interaction(learner, ihypo, iconfig, teacher, max_step, showFlag):

    Xd, Yd, perf, Xfull = initialize_simulation()
    postJoint = learner.postJoint
    _postJoint = teacher.postJoint #leading underscore means the teacher

    for step in range(max_step):

        postHypo = learner.posteriorHypo(postJoint)
        perf.append(postHypo[ihypo])
        # if reached_1_or_0(postHypo[ihypo]):
        #     break

        probeX = np.delete(Xfull, Xd)
        # teacher chooses xnext
        _hypoProbeM = teacher.get_hypoProbeMatrix(_postJoint, probeX)
        xnext, _probXHypo = teacher.teachingChoice(_hypoProbeM, ihypo, probeX)
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
            show_teach(step, ihypo, icongfig, Xd, Xfull, probeX,
                       learner, hypoProbeM, probXHypo, postHypo, postJoint)

    # print("Perf = %s" %(perf))
    return perf


def perf_all_configs(person, max_step, mode):
    nhypo = person.nhypo
    perf = []
    for ihypo in range(nhypo):
        for iconfig in range(person.nperm[ihypo]):
            if (mode == "explore"):
                perf.append(explore(person, ihypo, iconfig, max_step, False))
            elif (mode == "teach"):
                perf.append(teach(person, ihypo, iconfig, max_step, False))
    # print("Done simulation.")
    return perf


def perf_all_learner_configs(learner, teacher, max_step):
    nhypo = learner.nhypo
    perf = []
    for ihypo in range(nhypo):
        for iconfig in range(learner.nperm[ihypo]):
            perf.append(teacher_learner_interaction(learner,
                            ihypo, iconfig, teacher, max_step, False))
    # print("Done simulation.")
    return perf


def perf_space(master_set, nhypo, n_overlap, max_step, mode):
    plt.hold(True)
    # perf_data = []
    # hypo_data = []
    perf_set = []
    n_pattern = int(len(master_set)/2)
    bag = [[]]
    for count, hypo_indicator in enumerate(iter_hypo_indicator(nhypo, n_pattern, n_overlap)):
        hypo = build_hypo(hypo_indicator, nhypo)
        perm = hypo2perm(hypo, master_set)
        person = model(perm)
        perf = perf_all_configs(person, max_step, mode)
        # hypo_data.append(hypo)
        # perf_data.append(perf)
        print("Going through hypo %d: %s" %(count, hypo))
        for li in perf:
            if not is_alist_in_lili(li, perf_set):
                perf_set.append(li)
                plt.plot(np.arange(max_step), li, 'r-')
    plt.hold(False)
    plt.xlabel("Number of openings")
    plt.ylabel("Performance (prob-matching)")
    plt.show()
    # np.save("perf", perf)


if __name__ == '__main__':
    import seaborn as sns

    master_set = genMasterPermSet()
    del master_set[4:6] # delete diagonal complementary pattern
    # vis_perm_set(master_set)
    complement_flag = check_complementarily_paired(master_set)
    print("Are patterns complementary?", complement_flag)
    perf_space(master_set, 2, 0, 5, "explore")
