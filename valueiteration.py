import numpy as np
import gym
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')


def run_episode(env, policy_temp, gamma, render = False):
    """
    Evaluates policy by using it to run an episode and finding its total reward.
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns: total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy_temp[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy_temp, gamma,  n = 100):
    """
    Evaluates a policy by running it n times.
    returns: average total reward
    """
    scores = [run_episode(env, policy_temp, gamma= gamma, render = False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(v, gamma):
    """
    Extract the policy given a value-function
    """
    policy_temp = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy_temp[s] = np.argmax(q_sa)
    return policy_temp


def value_iteration(env, gamma):
    """
    Value-iteration algorithm
    """
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    iter = 0

    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            v[s] = max(q_sa)
        if np.sum(np.fabs(prev_v - v)) <= eps:
            print('Value-iteration converged at iteration# %d.' %(i+1))
            iter = i+1
            break
    return v, iter


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

    env = gym.make("FrozenLake-v0").env

    param_range = np.arange(0, 1, 0.1)
    iter_list = []
    rewards_list = []
    time_list = []
    '''
    for i in param_range:
        t1 = time.time()
        optimal_v, iter_to_converge = value_iteration(env, gamma=i)
        policy = extract_policy(optimal_v, gamma=i)
        rewards = evaluate_policy(env, policy, gamma=i, n=100)
        t2 = time.time()
        iter_list.append(iter_to_converge)
        rewards_list.append(rewards)
        time_list.append(t2-t1)
    
    plot_graph(param_range, iter_list, 'gamma (discount factor)', 'iterations to converge',
               'frozen lake : discount factor vs iterations to converge (value iteration)')
    plot_graph(param_range, rewards_list, 'gamma (discount factor)', 'average total reward',
               'frozen lake : discount factor vs average total reward (value iteration)')
    plot_graph(param_range, time_list, 'gamma (discount factor)', 'time',
               'frozen lake : discount factor vs time (value iteration)')
    '''
    optimal_v, iter_to_converge = value_iteration(env, gamma=1)
    policy = extract_policy(optimal_v, gamma=1)
    rewards = evaluate_policy(env, policy, gamma=1, n=100)

    print('optimal policy', policy)
