""" unit test """
from  e_vs_t import model

def test_hypo():
    """ test P(h|X,Y) """
    ihypo = 1
    iconfig = 1
    guy = model()
    possible_val = guy.getPossPostVals()
    x_full = guy.x
    for ihypo in range(guy.nhypo):
        for iconfig in range(guy.nperm[ihypo]):
            perm = guy.perm[ihypo][iconfig]
            y_full = [guy.gety(perm, x) for x in x_full]
            post_joint = guy.posteriorJoint(x_full, y_full)
            post_hypo = guy.posteriorHypo(post_joint)
            for prob in post_hypo:
                assert prob in possible_val
