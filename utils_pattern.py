from unique_permutations import unique_permutations as uperm
from scipy.special import comb
from copy import copy, deepcopy

from itertools import chain
from itertools import combinations

from utils import flatten

import numpy as np
import itertools
import warnings

def defineCompartment(name):
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

def NRConfiguration():
    compk = defineCompartment('NR')
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

def handmake_complementary(perm_set):
    perm_set[7], perm_set[9] = perm_set[9], perm_set[7]
    perm_set[11], perm_set[13] = perm_set[13], perm_set[11]
    return perm_set

def genMasterPermSet():
    compk = [defineCompartment('1H'),
             defineCompartment('1V'),
             defineCompartment('4Q'),
             defineCompartment('3H'),
             defineCompartment('3V')]
    n = len(compk)
    permLL = [0]*n
    for i in range(n):
        base = genBasePerm(compk[i])
        permL, _ = genAllPerm(16, compk[i], base)
        permLL[i] = permL
    perm_set =  permSet(permLL)
    # manual swap to make complementary pairs
    perm_set = handmake_complementary(perm_set)
    return perm_set

def check_complementarily_paired(perm_set):
    comp_pair = np.ones(16)
    n_pair = int(len(perm_set)/2)
    for i_pair in range(n_pair):
        input_pair = np.add(perm_set[2*i_pair], perm_set[2*i_pair + 1])
        if not np.array_equiv(input_pair, comp_pair):
            return False
    return True

# def whos_active(hypo):
#     p = flatten(hypo)
#     return list(set(p))

# def whos_inactive(n_pattern, active_patterns):
#     inactive = list(np.arange(n_pattern))
#     for active in active_patterns:
#         inactive.remove(active)
#     return inactive

# def concatenate_hypos(hypo1, hypo2):
#     nhypo = len(hypo1)
#     out_hypo = [[] for i in range(nhypo)]
#     for ihypo in range(nhypo):
#         out_hypo[ihypo] = hypo1[ihypo] + hypo2[ihypo]
#     return out_hypo

# def map_ipatt_to_pattern(hypo, dest_patterns):
#     nhypo = len(hypo)
#     out_hypo = [[] for i in range(nhypo)]
#     for ihypo, vals in enumerate(hypo):
#         for ipatt in vals:
#             out_hypo[ihypo].append(dest_patterns[ipatt])
#     return out_hypo

# def iterator_base_hypo(n_pattern, nhypo):
#     """ ouputs iterator for base_hypo: val = ipatt, ind = ihypo """
#     pattren_ind = list(np.arange(n_pattern))
#     yield itertools.combinations(pattren_ind, nhypo)

# def example_nth_perm(master_set, base_hypo, n):
#     master_set = genMasterPermSet()
#     n_pattern = int(len(master_set)/2)
#     nhypo = len(base_hypo)
#     # find active and inactive pattern set
#     active_set = whos_active(base_hypo)
#     inactive_set = whos_inactive(n_pattern, active_set)
#     # define set to add relative to master set
#     set_to_add = inactive_set
#     # make hypo indicator to add relative to the re-indexed set
#     n_add = len(set_to_add)
#     iter_to_add = pattern_hypo_product_space(n_add, nhypo)
#     hypo_indicator_to_add = next_n(iter_to_add, n)
#     # build hypo to be added
#     hypo_to_add = build_hypo(hypo_indicator_to_add)
#     # map hypo to be relative to master pattern set
#     hypo_to_add = map_ipatt_to_pattern(hypo_to_add, set_to_add)
#     # make concatenated hypo
#     hypo = concatenate_hypos(base_hypo, hypo_to_add)
#     # make perm by putting actual patterns into hypo
#     perm = hypo2perm(hypo, master_set)
#     return perm

def iter_hypo_indicator(nhypo, n_pattern, n_overlap):
    """ yields all non-hypo-isomorphic hypo_indicators
        hypo_indicator: val = [ihypos], ind = ipatt """
    base_bag = [[]]
    base_count = 0
    for hypo_base in pattern_hypo_product_space(nhypo, n_pattern):
        if hypo_indicator_filter(hypo_base, nhypo, base_bag[:-1]):
            base_bag.append([])
            base_count += 1
            bag[base_count] = hypo_base
            for additional in pattern_powerhypo_product_space(nhypo-1, n_pattern):
                if overlap_filter(additional, n_overlap):
                    additional = remap_hypo_indicator(additional, hypo_base, nhypo)
                    hypo_indicator = concatenate_hypo_indicators(hypo_base, additional)
                    yield hypo_indicator

def pattern_hypo_product_space(nhypo, n_pattern):
    """ ouput: hypo_indicator: val = [ihypos], ind = ipatt
        relative to add_patterns"""
    hypo_ind = [[i] for i in range(nhypo)]
    return itertools.product(hypo_ind, repeat=n_pattern)

def pattern_powerhypo_product_space(nhypo, n_pattern):
    power_hypo_ind = list(powerset(range(nhypo)))
    return itertools.product(power_hypo_ind, repeat=n_pattern)

def remap_hypo_indicator(hypo_indicator, hypo_base, nhypo):
    # it is possible to achieve the same goal by just filtering, but
    # it may be inefficient because most of the combinations are filtered out
    n_pattern = len(hypo_indicator)
    new_hypo_indictor = [[] for i in range(n_pattern)]
    for ipatt in range(n_pattern):
        for hypo_ind in hypo_indicator[ipatt]:
            inactive = [i for i in range(nhypo)]
            active = hypo_base[ipatt][0]
            inactive.remove(active)
            new_hypo_indictor[ipatt].append(inactive[hypo_ind])
    return new_hypo_indictor

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

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def build_hypo(hypo_indicator, nhypo):
    """ inpiut: hypo_indicator: val = ihypo; ind = ipatt
        output: hypo (nested list): ind = ihypo, val = patterns """
    n_pattern = len(hypo_indicator)
    hypo = [[] for i in range(nhypo)]
    for ipatt in range(n_pattern):
        for hypo_ind in hypo_indicator[ipatt]:
            hypo[hypo_ind].append(ipatt)
    return hypo

def hypo2perm(hypo, master_set):
    nhypo = len(hypo)
    perm = [[] for i in range(nhypo)]
    for ihypo in range(nhypo):
        patt_inds = hypo[ihypo]
        for patt_ind in patt_inds:
            for ind in range(2*patt_ind, 2*(patt_ind+1)):
                perm[ihypo].append(master_set[ind])
    return perm

def hypo_indicator_filter(hypo_indicator, nhypo, bag):
    if len(set(flatten(hypo_indicator))) == nhypo and \
       not is_hypobag_isomorphic(bag, hypo_indicator, nhypo):
        return True
    else:
        return False

def overlap_filter(hypo_indicator, n_overlap):
    # print(list(flatten(hypo_indicator)))
    if len(list(flatten(hypo_indicator))) == n_overlap:
        return True
    else:
        return False

def is_hypobag_isomorphic(bag, lili, nhypo):
    if len(bag) == 0:
        return False
    flat_list = extended_hypo_ind(nhypo, lili)
    for ele in bag:
        # print("element:", ele)
        # print("input list:", lili)
        flat_ele = extended_hypo_ind(nhypo, ele)
        if is_hypo_isomorphic(flat_ele, flat_list):
            return True
    return False

def extended_hypo_ind(nhypo, lili):
    new_map = list(powerset(range(nhypo)))
    new_list = []
    for li in lili:
        new_list.append(which_ind_in_mapli(new_map, li))
    return new_list

def which_ind_in_mapli(mapli, li):
    for ind, ele in enumerate(mapli):
        if list(ele) == li:
            return ind
    warnings.warn('List is not in map!')

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

if __name__ == '__main__':
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
    #         y = remap_hypo_indicator(x, hypo_indicator_base, nhypo)
    #         z = concatenate_hypo_indicators(hypo_indicator_base, y)
    #         print(count, z)
    #         count += 1

    for i, x in enumerate(iter_hypo_indicator(2,6,0)):
        print(i, x)
