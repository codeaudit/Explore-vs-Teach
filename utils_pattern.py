"""
2016-06-29
There should be more efficient way to build the concept-pattern
space, using powerset-like rules.
"""

from unique_permutations import unique_permutations as uperm
from scipy.special import comb
from copy import copy, deepcopy

from itertools import chain
from itertools import combinations

from utils import flatten

import numpy as np
import itertools
import warnings

def defineCompartment2x2():
    compk = np.arange(4)
    return compk

def defineCompartment4x4(name):
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
        nk_half = int(nk/2)
        base = np.append(np.zeros(nk_half), np.ones(nk_half)).tolist()
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

def NRConfiguration4x4():
    compk = defineCompartment4x4('NR')
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

# Old version
# def set_lili_diff(lili1, lili2):
#     if len(lili1) != len(lili2):
#         warnings.warn("Inputs shoud have the same first dimension")
#     diff = []
#     for i in range(len(lili1)):
#         diff += list(set(lili1[i]) - set(lili2[i]))
#     return diff

def set_lili_diff(lili1, lili2):
    if len(lili1) != len(lili2):
        warnings.warn("Inputs shoud have the same first dimension")
    diff = []
    for i in range(len(lili1)):
        set1 = set(lili1[i])
        set2 = set(lili2[i])
        diff += list((set1 - set2).union(set2 - set1))
    return list(set(diff))


def genMasterPermSet2x2():
    compk = defineCompartment2x2()
    base = genBasePerm(compk)
    permL, _ = genAllPerm(4, compk, base)
    return permL

def genMasterPermSet4x4():
    compk = [defineCompartment4x4('1H'),
             defineCompartment4x4('1V'),
             defineCompartment4x4('4Q'),
             defineCompartment4x4('3H'),
             defineCompartment4x4('3V')]
    n = len(compk)
    permLL = [0]*n
    for i in range(n):
        base = genBasePerm(compk[i])
        permL, _ = genAllPerm(16, compk[i], base)
        permLL[i] = permL
    perm_set =  permSet(permLL)
    # manual swap to make complementary pairs
    perm_set = handmake_complementary4x4(perm_set)
    return perm_set

def handmake_complementary4x4(perm_set):
    perm_set[7], perm_set[9] = perm_set[9], perm_set[7]
    perm_set[11], perm_set[13] = perm_set[13], perm_set[11]
    return perm_set

def check_complementarily_paired4x4(perm_set):
    comp_pair = np.ones(16)
    n_pair = int(len(perm_set)/2)
    for i_pair in range(n_pair):
        input_pair = np.add(perm_set[2*i_pair], perm_set[2*i_pair + 1])
        if not np.array_equiv(input_pair, comp_pair):
            return False
    return True


def iter_hypo_indicator(nhypo, n_pattern, n_overlap):
    """ yields all non-hypo-isomorphic hypo_indicators, each of which
        is a list of lists, where val = [ihypos], ind = ipatt. """
    # for i, x in enumerate(iter_hypo_indicator(2,6,5)):
    #     print(i, x)
    base_bag = [[]]
    base_count = 0
    additional_bag =[[]]
    additional_count = 0
    for hypo_base in pattern_hypo_product_space(nhypo, n_pattern):
        if hypo_indicator_filter(hypo_base, nhypo, base_bag):
            base_bag.append([])
            base_count += 1
            base_bag[base_count] = hypo_base
            # print(base_bag)
            for hypo_overlap in pattern_powerhypo_product_space(nhypo-1, n_pattern):
                if overlap_filter(hypo_overlap, n_overlap):
                    hypo_overlap = remap_overlap_indicator(hypo_overlap, hypo_base, nhypo)
                    hypo_indicator = concatenate_hypo_indicators(hypo_base, hypo_overlap)
                    if not is_hypobag_isomorphic(additional_bag, hypo_indicator, nhypo):
                        additional_bag.append([])
                        additional_count += 1
                        additional_bag[additional_count] = hypo_indicator
                        # print(additional_bag)
                        yield hypo_indicator


def pattern_hypo_product_space(nhypo, n_pattern):
    """ ouput: hypo_indicator: val = [ihypos], ind = ipatt.
        No empty values. """
    hypo_ind = [[i] for i in range(nhypo)]
    return itertools.product(hypo_ind, repeat=n_pattern)

def pattern_powerhypo_product_space(nhypo, n_pattern):
    """ For generating pattern overlap in hypo:
        ouput: hypo_indicator: val = [ihypos], ind = ipatt,
        where val can be any element in the powerset. """
    power_hypo_ind = list(powerset(range(nhypo)))
    return itertools.product(power_hypo_ind, repeat=n_pattern)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def remap_overlap_indicator(overlap_indicator, hypo_base, nhypo):
    """ For generating pattern overlap in hypo:
        remap values in overlap_indicator to ensure overlaps are added """
    # it is possible to achieve the same goal by just filtering, but
    # it may be inefficient because most of the combinations are filtered out
    n_pattern = len(overlap_indicator)
    new_overlap_indictor = [[] for i in range(n_pattern)]
    for ipatt in range(n_pattern):
        for hypo_ind in overlap_indicator[ipatt]:
            inactive = [i for i in range(nhypo)]
            active = hypo_base[ipatt][0]
            inactive.remove(active)
            new_overlap_indictor[ipatt].append(inactive[hypo_ind])
    return new_overlap_indictor

def concatenate_hypo_indicators(lili1, lili2):
    n = len(lili1)
    m = len(lili2)
    if n != m:
        return False
    new_lili = [[] for i in range(n)]
    for i in range(n):
        # a saver one is new_lili[i] += list(set(lili1[i] + lili2[i]))
        # but this may hide other bugs
        new_lili[i] = lili1[i] + lili2[i]
        new_lili[i].sort()
    return new_lili

def build_hypo(hypo_indicator, nhypo):
    """ inpiut: hypo_indicator: val = [ihypo]; ind = ipatt
        output: hypo (nested list): ind = ihypo, val = patterns """
    n_pattern = len(hypo_indicator)
    hypo = [[] for i in range(nhypo)]
    for ipatt in range(n_pattern):
        for hypo_ind in hypo_indicator[ipatt]:
            hypo[hypo_ind].append(ipatt)
    return hypo

def hypo2perm(hypo, master_set, pair_flag=False):
    if pair_flag:
        bundle = 2
    else:
        bundle = 1
    nhypo = len(hypo)
    perm = [[] for i in range(nhypo)]
    for ihypo in range(nhypo):
        patt_inds = hypo[ihypo]
        for patt_ind in patt_inds:
            for ind in range(bundle*patt_ind, bundle*(patt_ind+1)):
                perm[ihypo].append(master_set[ind])
    return perm

def hypo_indicator_filter(hypo_indicator, nhypo, bag):
    if len(set(flatten(hypo_indicator))) == nhypo and \
       not is_hypobag_isomorphic(bag, hypo_indicator, nhypo):
        return True
    else:
        return False

def overlap_filter(overlap_indicator, n_overlap):
    # print(list(flatten(overlap_indicator)))
    if len(list(flatten(overlap_indicator))) == n_overlap:
        return True
    else:
        return False

def is_hypobag_isomorphic(bag, lili, nhypo):
    if len(bag) == 0:
        return False
    recoded_list = extended_hypo_ind(nhypo, lili)
    for ele in bag:
        # print("element:", ele)
        # print("input list:", lili)
        recoded_ele = extended_hypo_ind(nhypo, ele)
        if is_hypo_isomorphic(recoded_ele, recoded_list):
            return True
    return False

def extended_hypo_ind(nhypo, lili):
    new_map = list(powerset(range(nhypo)))
    new_list = []
    for li in lili:
        new_list.append(which_ind_in_lili(new_map, li))
    return new_list

def which_ind_in_lili(lili, li):
    for ind, ele in enumerate(lili):
        if list(ele) == li:
            return ind
    warnings.warn('List is not in map of lists!')

def is_hypo_isomorphic(list1, list2):
    """ Based on http://www.geeksforgeeks.org/check-if-two-given-strings-are-isomorphic-to-each-other/ """
    if len(list1) != len(list2):
        return False
    n = len(list2)
    max_ind = max(list1 + list2) + 1
    marked = [False]*max_ind
    mapped = [-1]*max_ind
    for i in range(n):
        if mapped[list1[i]] == -1:
            if marked[list2[i]] == True:
                return False
            marked[list2[i]] = True
            mapped[list1[i]] = list2[i]
        elif mapped[list1[i]] != list2[i]:
            return False
    return True

def is_lili_subset(sub_lili, full_lili):
    """ sub_indicator and full_indicator should pertain to
        the same master pattern set. """
    if len(sub_lili) != len(full_lili):
        warnings.warn("Inputs should have same length")
    for i, li in enumerate(sub_lili):
        if len(li) > 0 and not set(li).issubset(set(full_lili[i])):
            return False
    return True

def iter_all_hypo_isomorphic(hypo_indicator, nhypo):
    """ Yields all hypo isomorphic indicators of input.
        Can deal with empty values. """
    hypo_ind = [i for i in range(nhypo)]
    for permuted in uperm(hypo_ind):
        perm_hypo_indicator = []
        for li in hypo_indicator:
            if len(li) >= 1:
                perm_li = [permuted[v] for v in li]
                perm_hypo_indicator.append(sorted(perm_li))
            elif len(li) == 0:
                perm_hypo_indicator.append(li)
        yield perm_hypo_indicator


def print_iterator(iterator, nhypo):
    for x in iterator:
        if len(set(flatten(x))) == nhypo:
            print(x)

def print_filtered_iterator(iterator, nhypo):
    bag = [[]]
    count = 0
    for x in iterator:
        if hypo_indicator_filter(x, nhypo, bag[:-1]):
                bag.append([])
                bag[count] = x
                count += 1
                print(x)

# Informal tests during code:
    # iterator = pattern_hypo_product_space(2, 6)
    # print_iterator(iterator, 2)
    # print("switch")
    # iterator = pattern_hypo_product_space(2, 6)
    # print_filtered_iterator(iterator, 2)

    # a = extended_hypo_ind(3,([0,1],[0],[1],[2]))
    # b = extended_hypo_ind(3,([0,1],[0],[2],[1]))
    # flag = is_hypo_isomorphic(a,b)
    # print(a)
    # print(b)
    # print(flag)

    # nhypo = 3
    # hypo_indicator_base = [[0],[1],[0],[1],[0],[2]]
    # count = 0
    # for x in pattern_powerhypo_product_space(6,2):
    #     if overlap_filter(x,3):
    #         y = remap_overlap_indicator(x, hypo_indicator_base, nhypo)
    #         z = concatenate_hypo_indicators(hypo_indicator_base, y)
    #         print(count, z)
    #         count += 1

    # for x in iter_all_hypo_isomorphic([[0],[1],[2],[],[0,1],[1,2],[0,1,2]], 3):
    #     print(x)

    # a = [[0],[1],[2],[],[0,1],[1,2],[0,1,2]]
    # hypo = build_hypo(a, 3)
    # print(hypo)

    # a = [[0],[1],[2],[],[0,1],[],[0,1,2]]
    # b = [[0],[1],[2],[2],[0,1],[1,2],[0,1,2]]
    # print(is_lili_subset(a,b))

if __name__ == '__main__':
    a = [[0],[1],[2],[],[0,1],[],[0,1,2]]
    b = [[0],[1],[2],[2],[0,1],[1,2],[0,1,2]]
    print(is_lili_subset(a,b))
