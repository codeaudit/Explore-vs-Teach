""" Simulating exploration and teaching in pattern-concept learning """
from copy import deepcopy

from utils import flatten
from utils import normalize
from utils import uniformSampleMaxInd
from utils import perturbDistr
from utils import is_alist_in_lili

from vis_utils import plot_nbyn
from vis_utils import overlayX
from vis_utils import plotHorLines
from vis_utils import vis_config
from vis_utils import vis_perm_set

from utils_pattern import genMasterPermSet4x4
from utils_pattern import genMasterPermSet2x2
from utils_pattern import iter_hypo_indicator
from utils_pattern import iter_all_hypo_isomorphic
from utils_pattern import build_hypo
from utils_pattern import hypo2perm
from utils_pattern import is_lili_subset
from utils_pattern import set_lili_diff

from e_vs_t import model

import matplotlib.pyplot as plt
import numpy as np
import warnings

def example_model4x4(mode="simple"):
    perm_set = genMasterPermSet4x4()
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


def prior_learning(person):
    x = np.arange(person.nx)
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


def initialize_simulation(person):
    Xd = []
    Yd = []
    perf = []
    Xfull = np.arange(person.nx)
    return Xd, Yd, perf, Xfull


def reached_1_or_0(value):
    eps = 1e-6
    if ((1 - value) < eps) or (value < eps):
        return True
    else:
        return False


def explore(learner, ihypo, pattern, max_step, showFlag=False):
    """ Level 1: simuate ideal exploration given hypo and pattern """
    Xd, Yd, perf, Xfull = initialize_simulation(learner)
    learner.initialize()
    terminalVals = learner.getPossPostVals()
    postJoint = learner.postJoint

    for step in range(max_step):

        postHypo = learner.posteriorHypo(postJoint)
        perf.append(postHypo[ihypo])

        # print("step %s" %(step))
        # print("Before update P(h,f|D) = %s" %(postJoint)) #why does this make a difference?
        # print("Before update P(h|D) = %s" %(postHypo))

        # terminate early to avoid error in what?
        # if  learner.postHypo[ihypo] in terminalVals:
        #     break

        probeX = np.delete(Xfull, Xd)
        score = learner.explore(postJoint, probeX, 'prob_gain')
        xnext = learner.explore_choice(score, probeX)
        ynext = learner.gety(pattern, xnext)
        Xd.append(xnext)
        Yd.append(ynext)

        postJoint = learner.posteriorJoint(Xd, Yd)

        # print("After update P(h,f|D) = %s" %(postJoint)) #why does this make a difference?
        # print("After update P(h|D) = %s" %(learner.posteriorHypo(postJoint)))

        if showFlag:
            print("step %s" %(step))
            print("P(h|D) = %s" %(postHypo))
            print("P(h,f|D) = %s" %(postJoint))
            # print("Terminal Values = %s" %(terminalVals))
            print("score = %s" %(score))
            print("data = %s; %s" %(Xd, Yd))

            ax = plt.subplot(1, 3, 1)
            Yfull = [learner.gety(pattern, Xfull[i])
                     for i in range(len(Xfull))]
            plot_nbyn(Yfull)
            overlayX(Xd[:step], int(np.sqrt(learner.nx)))
            ax.set_title('Truth (%s) & X' %(ihypo))

            ax = plt.subplot(1, 3, 2)
            plot_nbyn(score)
            overlayX(Xd, int(np.sqrt(learner.nx)))
            ax.set_title('Score')

            ax = plt.subplot(1, 3, 3)
            plt.bar(np.arange(learner.nhypo), postHypo)
            plotHorLines(terminalVals, [0, learner.nhypo])

            plt.show()

    return perf


def teach(pair, ihypo, pattern, max_step, showFlag=False):
    """ Level 1: simuate ideal teaching given hypo and pattern """
    Xd, Yd, perf, Xfull = initialize_simulation(pair)
    pair.initialize()
    postJoint = pair.postJoint

    for step in range(max_step):

        postHypo = pair.posteriorHypo(postJoint)
        perf.append(postHypo[ihypo])

        # if reached_1_or_0(postHypo[ihypo]):
        #     break

        # print("step %s" %(step))
        # print("Before update P(h,f|D) = %s" %(postJoint)) #why does this make a difference?
        # print("Before update P(h|D) = %s" %(postHypo))

        probeX = np.delete(Xfull, Xd)
        hypoProbeM = pair.get_hypoProbeMatrix(postJoint, probeX)
        xnext, probXHypo = pair.teachingChoice(hypoProbeM, ihypo, probeX)
        ynext = pair.gety(pattern, xnext)
        postJoint = pair.updatePosteriorJointWithTeacher(
            [xnext], [ynext], postJoint, probXHypo)

        Xd.append(xnext)
        Yd.append(ynext)

        # print("P_T(x|h) = %s" %(probXHypo))
        # print("After update P(h,f|D) = %s" %(postJoint)) #why does this make a difference?
        # print("After update P(h|D) = %s" %(pair.posteriorHypo(postJoint)))

        if showFlag:
            show_teach(step, ihypo, pattern, Xd, Yd, Xfull, probeX,
                       pair, hypoProbeM, probXHypo, postHypo, postJoint)
    # print("Perf = %s" %(perf))
    return perf


def interact(teacher, ihypo, pattern, learner, max_step, showFlag=False):
    """ Level 1: simuate conceptual misaligned teaching
        given true hypo and pattern, which the teacher has """
    Xd, Yd, perf, Xfull = initialize_simulation(teacher)
    learner.initialize()
    teacher.initialize()
    postJoint = learner.postJoint
    _postJoint = teacher.postJoint #leading underscore means the teacher

    for step in range(max_step):

        postHypo = learner.posteriorHypo(postJoint)
        perf.append(postHypo[ihypo])
        # if reached_1_or_0(postHypo[ihypo]):
        #     break

        probeX = np.delete(Xfull, Xd)
        # teacher chooses xnext and updates belief
        _hypoProbeM = teacher.get_hypoProbeMatrix(_postJoint, probeX)
        xnext, _probXHypo = teacher.teachingChoice(_hypoProbeM, ihypo, probeX)
        ynext = learner.gety(pattern, xnext)
        _postJoint = teacher.updatePosteriorJointWithTeacher(
                        [xnext], [ynext], _postJoint, _probXHypo)
        # learner learns
        hypoProbeM = learner.get_hypoProbeMatrix(postJoint, probeX)
        probXHypo = normalize(hypoProbeM[:, xnext] + 1e-6)
        postJoint = learner.updatePosteriorJointWithTeacher(
                        [xnext], [ynext], postJoint, probXHypo)

        Xd.append(xnext)
        Yd.append(ynext)

        if showFlag:
            show_teach(step, ihypo, pattern, Xd, Yd, Xfull, probeX,
                       learner, hypoProbeM, probXHypo, postHypo, postJoint)

    return perf


def show_teach(step, ihypo, pattern, Xd, Yd, Xfull, probeX,
               pair, hypoProbeM, probXHypo, postHypo, postJoint):
    """ plotting function for level 1 teach() and interact() """
    plot_M = deepcopy(hypoProbeM)
    plot_M[:,probeX] += 1e-6
    print("step %s" %(step))
    print("P_T(x|h) = %s" %(probXHypo))
    print("P(h,f|D) = %s" %(postJoint)) #why does this make a difference?
    print("Before update P(h|D) = %s" %(postHypo))
    print("After update P(h|D) = %s" %(pair.posteriorHypo(postJoint)))
    print("score = %s" %(normalize(plot_M[ihypo] + 1e-6)))
    print("data = %s; %s" %(Xd, Yd))

    ax = plt.subplot(1, 3, 1)
    Yfull = [pair.gety(pattern, Xfull[i])
             for i in range(len(Xfull))]
    plot_nbyn(Yfull)
    overlayX(Xd[:step], int(np.sqrt(pair.nx)))
    ax.set_title('Truth (%s) & X' %(ihypo))

    ax = plt.subplot(1, 3, 2)
    plot_nbyn(hypoProbeM[ihypo])
    overlayX(Xd, int(np.sqrt(pair.nx)))
    ax.set_title('Score & Next x')

    ax = plt.subplot(1, 3, 3)
    # plt.bar(np.arange(pair.nhypo), postHypo)
    plt.imshow(plot_M.transpose(), interpolation="nearest", cmap="gray")
    ax.set_title('update M')

    plt.show()


def perf_all_configs(person, max_step, mode):
    """ Level 2:
        loop of explore() or teach(),
        loop through hypo and patteren. """
    nhypo = person.nhypo
    perf_list = []
    perf_avg = np.zeros(max_step)
    for ihypo in range(nhypo):
        for iconfig in range(person.nperm[ihypo]):
            pattern = person.perm[ihypo][iconfig]
            if (mode == "explore"):
                perf = explore(person, ihypo, pattern, max_step)
            elif (mode == "teach"):
                perf = teach(person, ihypo, pattern, max_step)
            perf_list.append(perf)
            factor = person.priorHypo[ihypo]*person.priorLabelGivenHypo[ihypo][iconfig]
            perf_avg += np.multiply(perf, factor)
    # print("Done simulation.")
    return perf_list, perf_avg


def perf_space(master_set, nhypo, n_overlap, max_step, mode):
    """ Level 3:
        loop of perf_all_configs(),
        loop through different concept-pattern spaces. """
    perf_data = []
    perf_avg_data = []
    n_pattern = int(len(master_set)) # or for paired: = int(len(master_set)/2)
    for count, hypo_indicator in enumerate(iter_hypo_indicator(nhypo, n_pattern, n_overlap)):
        hypo = build_hypo(hypo_indicator, nhypo)
        perm = hypo2perm(hypo, master_set)
        person = model(perm)
        # person.change_mode("softmax", 20, "one-step")
        perfs, perf_avg = perf_all_configs(person, max_step, mode)
        perf_data.append(perfs)
        perf_avg_data.append(perf_avg)
        print("Going through space %d: %s" %(count, hypo))
    # np.save("perf", perf)
    return perf_data, perf_avg_data


def perf_shared_configs(learner, teacher, max_step):
    """ Level 2:
        loop of misexplore() and interact(),
        loop through shared patterens. """
    learner_perms = []
    for ihypo in range(learner.nhypo):
        for iconfig in range(learner.nperm[ihypo]):
            learner_perms.append(list(learner.perm[ihypo][iconfig]))
    perfs_explo = []
    perfs_inter = []
    factors = []
    for ihypo in range(teacher.nhypo):
        for iconfig in range(teacher.nperm[ihypo]):
            pattern = list(teacher.perm[ihypo][iconfig])
            factor = teacher.priorHypo[ihypo]*teacher.priorLabelGivenHypo[ihypo][iconfig]
            if pattern in learner_perms:
                factors.append(factor)
                perf = explore(learner, ihypo, pattern, max_step)
                perfs_explo.append(perf)
                perf = interact(teacher, ihypo, pattern, learner, max_step)
                perfs_inter.append(perf)
    print("Number of shared pattern:", len(perfs_inter))
    if abs(sum(factors) - 1) > 1e-6:
        print("Factors sumed to", sum(factors))
        warnings.warn("Not everything is shared!")
    return perfs_explo, perfs_inter, factors


def perf_space_product(master_set, nhypo, n_overlap, max_step):
    """ Level 3:
        loop of perf_shared_configs(),
        loop through product of different concept-pattern spaces. """
    perf_explo_data = []
    perf_inter_data = []
    factor_data = []
    num_swap = []
    count = 0
    n_pattern = int(len(master_set)) # or for paired: = int(len(master_set)/2)
    for i, hypo_indicator_i in enumerate(iter_hypo_indicator(nhypo, n_pattern, n_overlap)):
        for j, hypo_indicator_j in enumerate(iter_hypo_indicator(nhypo, n_pattern, n_overlap)):
            if j >= i:
                # build learner
                hypo_i = build_hypo(hypo_indicator_i, nhypo)
                perm_i = hypo2perm(hypo_i, master_set)
                learner = model(perm_i)
                # learner.change_mode("softmax", 20, "one-step")
                # build teacher
                hypo_j = build_hypo(hypo_indicator_j, nhypo)
                perm_j = hypo2perm(hypo_j, master_set)
                teacher = model(perm_j)
                # teacher.change_mode("softmax", 20, "one-step")
                # calculate and record interaction
                # n_swap = len(set_lili_diff(hypo_indicator_i, hypo_indicator_j))
                n_swap = len(set_lili_diff(hypo_i, hypo_j))
                num_swap.append(n_swap)
                perfs_explo, perfs_inter, factors = perf_shared_configs(learner, teacher, max_step)
                perf_explo_data.append(perfs_explo)
                perf_inter_data.append(perfs_inter)
                factor_data.append(factors)
                print("Interacting %d-%d-%d: %s x %s. Number of swaps = %d." %(i, j, count, hypo_i, hypo_j, n_swap))
                count += 1
    return perf_explo_data, perf_inter_data, factor_data, num_swap


def plot_unique_perf(x, perf_data, style):
    """ plotting function for level 3 perf_space() """
    plt.figure(figsize=(5,5))
    plt.hold(True)
    perf_set = []
    for ispace, space in enumerate(perf_data):
        for ind, perf in enumerate(space):
            if not is_alist_in_lili(perf, perf_set):
                print("Space %s-Pattern %s: %s" %(ispace, ind, perf))
                perf_set.append(perf)
                plt.plot(x, perf, style, linewidth=1)
    plt.hold(False)
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Number of openings")
    plt.ylabel("Performance (prob-matching)")
    plt.savefig('result.pdf')  # or just plt.show()


def simulate_overlap():
    """ Level 4:
        loop of perf_space(),
        loop through parameters that define spaces. """
    master_set = genMasterPermSet2x2()
    nhypo = 2
    max_step = 4
    x = np.arange(max_step)
    plt.figure(figsize=(6,6))
    plt.hold(True)
    for n_overlap in range(7):
        e_data, e_avgs = perf_space(master_set, nhypo, n_overlap, max_step, "explore")
        t_data, t_avgs = perf_space(master_set, nhypo, n_overlap, max_step, "teach")
        # for ispace, t_space in enumerate(t_data):
        #     for iconfig, t_perf in enumerate(t_space):
        #         e_perf = e_data[ispace][iconfig]
        #         if smaller_anywhere(e_perf, t_perf):
        #             print("space %d - pattern %d" %(ispace, iconfig))
        #         plt.plot(e_perf, t_perf, 'r-', linewidth=1, alpha=.01)
        #         plt.plot(e_perf[-1], t_perf[-1], 'r.', alpha=.1)
        #     plt.plot(e_avgs[ispace], t_avgs[ispace], 'b-', linewidth=1, alpha=.3)
        plt.plot(np.mean(e_avgs, axis=0), np.mean(t_avgs, axis=0), linewidth=2)
    plt.plot([0,1], [0,1], 'k:', alpha=.5)
    plt.hold(False)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Exploration performance", fontsize=20)
    plt.ylabel("Teaching performance", fontsize=20)
    plt.savefig('result.pdf')  # or just plt.show()


def smaller_anywhere(a, b):
    for i in range(len(a)):
        if b[i] - a[i] > 1e-10:
            return True
    return False

def simulate_nhypo():
    """ Level 4:
        loop of perf_space(),
        loop through parameters that define spaces. """
    master_set = genMasterPermSet2x2()
    n_overlap = 0
    max_step = 4
    x = np.arange(max_step)
    plt.figure(figsize=(6,6))
    plt.hold(True)
    for nhypo in range(2,7):
        e_data, e_avgs = perf_space(master_set, nhypo, n_overlap, max_step, "explore")
        t_data, t_avgs = perf_space(master_set, nhypo, n_overlap, max_step, "teach")
        # for ispace, t_space in enumerate(t_data):
        #     for iconfig, t_perf in enumerate(t_space):
        #         e_perf = e_data[ispace][iconfig]
        #         if smaller_anywhere(t_perf, e_perf):
        #             print("space %d - pattern %d" %(ispace, iconfig))
        #         plt.plot(e_perf, t_perf, 'r-', linewidth=1, alpha=.01)
        #         plt.plot(e_perf[-1], t_perf[-1], 'r.', alpha=.1)
        #     plt.plot(e_avgs[ispace], t_avgs[ispace], 'b-', linewidth=1, alpha=.3)
        plt.plot(np.mean(e_avgs, axis=0), np.mean(t_avgs, axis=0), linewidth=1)
    plt.plot([0,1], [0,1], 'k:', alpha=.5)
    plt.hold(False)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Exploration performance", fontsize=20)
    plt.ylabel("Teaching performance", fontsize=20)
    plt.savefig('result.pdf')  # or just plt.show()


def avg_over_space(perf_data, factor_data, max_step):
    """ input data has 3 layers [ispace][iconfig][step]
        output data has 2 layers [ispace][step] """
    perfs_avg = []
    for ispace, perfs in enumerate(perf_data):
        perf_avg = np.zeros(max_step)
        for iconfig, perf in enumerate(perfs):
            perf_avg += np.multiply(perf, factor_data[ispace][iconfig])
        perfs_avg.append(perf_avg)
    return perfs_avg


def simulate_swap():
    master_set = genMasterPermSet2x2()
    nhypo = 2
    # n_overlap = 1
    max_step = 4
    explo_data, inter_data, factors, num_swap = perf_space_product(master_set, nhypo, n_overlap, max_step)
    explo_avgs = avg_over_space(explo_data, factors, max_step)
    inter_avgs = avg_over_space(inter_data, factors, max_step)
    x = np.arange(max_step)

    plt.figure(figsize=(6,6))
    plt.hold(True)

    for i_swap in range(max(num_swap)+1):
        count = 0
        explo_avg_swap = np.zeros(max_step)
        inter_avg_swap = np.zeros(max_step)
        for ind, swap_value in enumerate(num_swap):
            if swap_value == i_swap:
                count += 1
                explo_avg_swap += explo_avgs[ind]
                inter_avg_swap += inter_avgs[ind]
        plt.plot(explo_avg_swap/count, inter_avg_swap/count, linewidth=2)

    end_point_set = []
    for ispace, t_space in enumerate(inter_data):
        for iconfig, t_perf in enumerate(t_space):
            e_perf = explo_data[ispace][iconfig]
            end_point = [e_perf[-1], t_perf[-1]]
            if not is_alist_in_lili(end_point, end_point_set):
                end_point_set.append(end_point)
                plt.plot(e_perf[-1], t_perf[-1], 'ro', alpha=1)
            # if smaller_anywhere(t_perf, e_perf):
            #     print("space %d - pattern %d" %(ispace, iconfig))
            # if t_perf[-1] < 0.1 and e_perf[-1] > 0.8:
            #     print("space %d - pattern %d" %(ispace, iconfig))
            #     plt.plot(e_perf, t_perf, 'r-', linewidth=1, alpha=.01)
        # plt.plot(e_perf, t_perf, 'r-', linewidth=1, alpha=.01)
        # plt.plot(e_avgs[ispace], t_avgs[ispace], 'b-', linewidth=1, alpha=.3)

    plt.plot([0,1], [0,1], 'k:', alpha=.5)
    plt.hold(False)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Exploration performance", fontsize=20)
    plt.ylabel("Teaching performance", fontsize=20)
    plt.savefig('result.pdf')  # or just plt.show()


# def perf_space_size_mismatch(small_hypo_indicator,
#                              master_set, nhypo, n_overlap, max_step):
#     n_pattern = int(len(master_set)/2)
#     if len(small_hypo_indicator) != n_pattern or \
#        max(flatten(small_hypo_indicator)) > nhypo:
#         warnings.warn("Inputs have the wrong size(s).")
#
#     plt.figure(figsize=(5,5))
#     plt.hold(True)
#
#     perf_unique = []
#     hypo = build_hypo(small_hypo_indicator, nhypo)
#     perm = hypo2perm(hypo, master_set)
#     person = model(perm)
#     perfs, perf_avg = perf_all_configs(person, max_step, "explore")
#     perf_unique = plot_unique_perf(np.arange(max_step), perfs, perf_unique, 'b:')
#
#     perf_unique = []
#     hypo = build_hypo(small_hypo_indicator, nhypo)
#     perm = hypo2perm(hypo, master_set)
#     person = model(perm)
#     perfs, perf_avg = perf_all_configs(person, max_step, "teach")
#     perf_unique = plot_unique_perf(np.arange(max_step), perfs, perf_unique, 'k:')
#
#     perf_unique = []
#     for count, full in enumerate(iter_hypo_indicator(nhypo, n_pattern, n_overlap)):
#         for small in iter_all_hypo_isomorphic(small_hypo_indicator, nhypo):
#             if is_lili_subset(small, full):
#                 full_hypo = build_hypo(full, nhypo)
#                 full_perm = hypo2perm(full_hypo, master_set)
#                 full_guy = model(perm)
#                 print("Going through hypo %d: %s" %(count, full_hypo))
#                 small_hypo = build_hypo(small, nhypo)
#                 small_perm = hypo2perm(small_hypo, master_set)
#                 small_guy = model(small_perm)
#                 perfs = perf_shared_configs(small_guy, full_guy, max_step)
#                 perf_unique = plot_unique_perf(np.arange(max_step), perfs, perf_unique, 'r-')
#
#     plt.hold(False)
#     plt.ylim([0,1])
#     plt.xlabel("Number of openings")
#     plt.ylabel("Performance (prob-matching)")
#     plt.savefig('result.png')  # or just plt.show()


if __name__ == '__main__':
    import seaborn as sns
    # master_set = genMasterPermSet2x2()
    # vis_perm_set(master_set)

    # master_set = genMasterPermSet2x2()
    # nhypo = 2
    # n_overlap = 4
    # max_step = 4
    # e_data, _ = perf_space(master_set, nhypo, n_overlap, max_step, "explore")
    # plot_unique_perf(np.arange(max_step), e_data, 'r-')

    simulate_overlap()

    # simulate_nhypo()

    # simulate_swap()
