import numpy as _np
from mdptoolbox.mdp import MDP
import math as _math
import time as _time
import mdptoolbox.util as _util


def _computeDimensions(transition):
    A = len(transition)
    try:
        if transition.ndim == 3:
            S = transition.shape[1]
        else:
            S = transition[0].shape[0]
    except AttributeError:
        S = transition[0].shape[0]
    return S, A


class QLearning(MDP):

    """A discounted MDP solved using the Q learning algorithm.
    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    n_iter : int, optional
        Number of iterations to execute. This is ignored unless it is an
        integer greater than the default value. Defaut: 10,000.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.
    Data Attributes
    ---------------
    Q : array
        learned Q matrix (SxA)
    V : tuple
        learned value function (S).
    policy : tuple
        learned optimal policy (S).
    mean_discrepancy : array
        Vector of V discrepancy mean over 100 iterations. Then the length of
        this vector for the default value of N is 100 (N/100).
    """

    def __init__(self, transitions, reward, discount, n_iter=10000,
                 skip_check=False):
        # Initialise a Q-learning MDP.

        # The following check won't be done in MDP()'s initialisation, so let's
        # do it here
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."

        MDP.__init__(self, transitions, reward, discount, 1e-20, self.max_iter)

        if not skip_check:
            # We don't want to send this to MDP because _computePR should not
            #  be run on it, so check that it defines an MDP
            _util.check(transitions, reward)

        # Store P, S, and A
        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)

        self.R = reward

        self.discount = discount

        # Initialisations
        self.Q = _np.zeros((self.S, self.A))
        self.mean_discrepancy = []

    def run(self):
        # Run the Q-learning algoritm.
        discrepancy = []

        self.time = _time.time()

        exploration_rate = 1
        max_exploration_rate = 1
        min_exploration_rate = 0.1
        exploration_decay_rate = 0.0001

        # initial state choice
        s = _np.random.randint(0, self.S)

        for n in range(1, self.max_iter + 1):

            # Reinitialisation of trajectories every 100 transitions
            if (n % 100) == 0:
                s = _np.random.randint(0, self.S)

            # Action choice : greedy with increasing probability
            # probability 1-(1/log(n+2)) can be changed
            pn = _np.random.uniform(0, 1)
            if pn > exploration_rate:
                # optimal_action = self.Q[s, :].max()
                a = self.Q[s, :].argmax()
            else:
                a = _np.random.randint(0, self.A)

            # Simulating next state s_new and reward associated to <s,s_new,a>
            p_s_new = _np.random.random()
            p = 0
            s_new = -1
            while (p < p_s_new) and (s_new < (self.S - 1)):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]

            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            # Updating the value of Q
            # Decaying update coefficient (1/sqrt(n+2)) can be changed
            delta = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
            dQ = (1 / _math.sqrt(n + 2)) * delta
            self.Q[s, a] = self.Q[s, a] + dQ

            exploration_rate = min_exploration_rate + \
                               (max_exploration_rate - min_exploration_rate) * _np.exp(-exploration_decay_rate * n)

            # current state is updated
            s = s_new

            # Computing and saving maximal values of the Q variation
            discrepancy.append(_np.absolute(dQ))

            # Computing means all over maximal Q variations values
            if len(discrepancy) == 100:
                self.mean_discrepancy.append(_np.mean(discrepancy))
                discrepancy = []

            # compute the value function and the policy
            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)

        self._endRun()


