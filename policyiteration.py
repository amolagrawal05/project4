import numpy as np
import gym
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')


def run_episode(env, policy, gamma, render= False):
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
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma, n= 100):
    """
    Evaluates a policy by running it n times.
    returns: average total reward
    """
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(v, gamma):
    """
    Extract the policy given a value-function
    v: value function
    gamma : discount factor
    returns optimal policy
    """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma):
    """
    Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in items of v[s]
    and solve them to find the value function.
    env : gym environment
    policy : random policy
    gamma : discount factor
    return : value
    """
    v = np.zeros(env.nS)
    eps = 1e-20
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if np.sum(np.fabs(prev_v - v)) <= eps:
            # value converged
            break
    return v


def policy_iteration(env, gamma):
    """
    Policy-Iteration algorithm
    env : gym environment
    gamma : discount factor
    return : optimal policy, no of iterations to converge
     """
    policy = np.random.choice(env.nA, size=env.nS)  # initialize a random policy
    max_iterations = 100000
    gamma = gamma
    iter = 0

    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if np.all(policy == new_policy):
            print('Policy-Iteration converged at step %d.' %(i+1))
            iter = i+1
            break
        policy = new_policy
    return policy, iter


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
        optimal_policy, iter_to_converge = policy_iteration(env, gamma=i)
        rewards = evaluate_policy(env, optimal_policy, gamma=i)
        t2 = time.time()
        iter_list.append(iter_to_converge)
        rewards_list.append(rewards)
        time_list.append(t2-t1)

    plot_graph(param_range, iter_list, 'gamma (discount factor)', 'iterations to converge',
               'frozen lake : discount factor vs iterations to converge (policy iteration)')
    plot_graph(param_range, rewards_list, 'gamma (discount factor)', 'average total reward',
               'frozen lake : discount factor vs average total reward (policy iteration)')
    plot_graph(param_range, rewards_list, 'gamma (discount factor)', 'time',
               'frozen lake : discount factor vs time (policy iteration)')
    '''
    optimal_policy, iter_to_converge = policy_iteration(env, gamma=1)
    rewards = evaluate_policy(env, optimal_policy, gamma=1)

    print('optimal policy', optimal_policy)
