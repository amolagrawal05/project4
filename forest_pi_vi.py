import mdptoolbox.example
import numpy as np
from mdptoolbox.mdp import MDP
from forest_ql import QLearning
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def plot_graph(params, data, xlabel, ylabel, title):

    plt.figure()
    plt.title(title)
    plt.plot(params, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(title)
    plt.clf()

    return


if __name__ == '__main__':

    P, R = mdptoolbox.example.forest(2000, 4, 2)

    eps = 1e-20
    iterations = 1000

    param_range = np.arange(0.1, 1, 0.1)
    vi_iter_list = []
    pi_iter_list = []
    vi_time_list = []
    pi_time_list = []

    ql = QLearning(P, R, 1)
    ql.run()

    vi = mdptoolbox.mdp.ValueIteration(P, R, 1)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 1)

    vi.run()
    pi.run()

    vi_policy = vi.policy
    pi_policy = pi.policy

    '''
    for i in param_range:

        # mdp_obj = MDP(P, R, i, eps, iterations)
        vi = mdptoolbox.mdp.ValueIteration(P, R, i)
        pi = mdptoolbox.mdp.PolicyIteration(P, R, i)
        ql = QLearning(P, R, i)

        vi.run()
        pi.run()

        vi_iter_list.append(vi.iter)
        pi_iter_list.append(pi.iter)

        vi_time_list.append(vi.time)
        pi_time_list.append(pi.time)

    
    plot_graph(param_range, vi_iter_list, 'gamma (discount factor)', 'iterations to converge',
               'forest: discount factor vs iterations to converge (value iteration)')

    plot_graph(param_range, vi_time_list, 'gamma (discount factor)', 'time',
               'forest: discount factor vs time (value iteration)')

    plot_graph(param_range, pi_iter_list, 'gamma (discount factor)', 'iterations to converge',
               'forest: discount factor vs iterations to converge (policy iteration)')

    plot_graph(param_range, pi_time_list, 'gamma (discount factor)', 'time',
               'forest: discount factor vs time (policy iteration)')
    '''