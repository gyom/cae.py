
import numpy as np

def one_step_update(U, grad_U, epsilon, L, current_q, verbose=False):
    # Code inspired from
    # "MCMC using Hamiltonian Dynamics" by Radford Neal.
    #
    # U : potential function of the states.
    #     takes numpy arrays of shape (d,) and yields a value of type double
    #
    # grad_U : gradient of U
    #          takes numpy arrays of shape (d,) and yields arrays of shape (d,)
    # epsilon : size of the leapfrog jumps. type double
    #
    # L : number of leapfrog jumps to perform. type int
    #
    # current_q : starting state from which the proposals are generated.
    #             numpy array of shape (d,)
    #
    # Note that the data points are probably "baked" into the functions U and grad_U.
    # That is, it's actually U(q | data) instead of just U(q) and we
    # are interested in sampling from the posterior.
    #
    # This function uses the traditional momentum from physics
    # as potential K(p) = (0.5 * p**2 / m).sum() .
    # Here the mass is always 1.0 .

    q = current_q
    p = np.random.normal(size = q.shape)
    current_p = p

    # half update for momentum at beginning
    p = p - epsilon * grad_U(q)/2

    for i in np.arange(1,L+1):
        q = q + epsilon * p
        # full update for momentum except at last iteration
        if i != L:
            p = p - epsilon * grad_U(q)

    # half update for momentum at the end
    p = p - epsilon * grad_U(q)/2

    # negating momentum (I don't understand why)
    p = -p

    current_U = U(current_q)
    current_K = (current_p ** 2).sum() / 2
    proposed_U = U(q)
    proposed_K = (p ** 2).sum() / 2

    acceptance_ratio = np.exp(current_U - proposed_U + current_K - proposed_K)
    if verbose:
        print "Acceptance odds for proposal is %0.2f" % acceptance_ratio
        print "We would go from "
        print "    H(current_q, current_p) = %0.6f" % (current_U + current_K,)
        print "to"
        print "    H(q, p) = %0.6f" % (proposed_U + proposed_K,)
        print ""

    if ((acceptance_ratio >= 1.0) or
        acceptance_ratio > np.random.random()):
        if verbose:
            print "Accepted move from "
            print "    current_q : %0.2f" % current_q[0]
            print "    current_p : %0.2f" % current_p[0]
            print "to"
            print "    q : %0.2f" % q[0]
            print "    p : %0.2f" % p[0]
            print ""
        # accept
        return q
    else:
        if verbose:
            print "Rejected move from "
            print "    current_q : %0.2f" % current_q[0]
            print "    current_p : %0.2f" % current_p[0]
            print "to"
            print "    q : %0.2f" % q[0]
            print "    p : %0.2f" % p[0]
            print ""
        # reject
        return current_q

