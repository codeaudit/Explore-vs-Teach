""" unit test """

from utils_pattern import genMasterPermSet

from e_vs_t import model

from simulate import example_model
from simulate import perf_all_configs
from simulate import perf_all_learner_configs

def setup_test_hypo(guy):
    """ test P(h|X,Y) for initialize_model("full") """
    possible_val = guy.getPossPostVals()
    x_full = np.arange(guy.nx)
    for ihypo in range(guy.nhypo):
        for iconfig in range(guy.nperm[ihypo]):
            perm = guy.perm[ihypo][iconfig]
            y_full = [guy.gety(perm, x) for x in x_full]
            post_joint = guy.posteriorJoint(x_full, y_full)
            post_hypo = guy.posteriorHypo(post_joint)
            for prob in post_hypo:
                assert prob in possible_val

def test_2hypos():
    perm_set = genMasterPermSet()
    perm = [0]*2
    perm[0] = perm_set[0:2] + perm_set[6:10]
    perm[1] = perm_set[2:4] + perm_set[10:14]
    guy = model(perm)
    setup_test_hypo(guy)

def test_3hypo():
    guy = example_model("full")
    setup_test_hypo(guy)

def test_smoke_perf_all():
    max_step = 5
    person = example_model("full")
    perf_all_configs(person, max_step, "explore")
    learner = example_model("simple")
    teacher = example_model("full")
    perf_all_learner_configs(learner, teacher, max_step)
    assert True

def test_is_isomorphic():
    a = extended_hypo_ind(3,([0],[0],[1],[2]))
    b = extended_hypo_ind(3,([0],[0],[2],[1]))
    flag = is_hypo_isomorphic(a,b)
    # print(a)
    # print(b)
    # print(flag)
    assert flag == True
