import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

temp = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(temp, 'hiivemdptoolbox'))

import hiive.mdptoolbox.mdp as mdp
import hiive.mdptoolbox.example as example


def data_process(data_list):

    size = len(data_list)
    print(size)
    reward_list = []
    time_list = []
    error_list = []

    for i in range(size):
        reward_list.append(data_list[i]['Reward'])
        time_list.append(data_list[i]['Time'])
        error_list.append(data_list[i]['Error'])

    return reward_list, time_list, error_list


def plot_graph(data, xlabel, ylabel, title):

    plt.figure()
    plt.title(title)
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(title)
    plt.clf()

    return


def plot_reward(data_list, episode, label, title):

    data_per_thousand_episodes = np.split(np.array(data_list), episode / 1000)

    data_value_list = []
    counter = 1000

    for i in data_per_thousand_episodes:

        current_data_value = sum(i) / 1000
        data_value_list.append(current_data_value)
        counter = counter + 1000

    plt.title(title)
    plt.plot(data_value_list)

    plt.xlabel('number of iterations (in thousands)')
    plt.ylabel(label)
    plt.savefig(title)
    plt.clf()


def fun(stats, title):

    reward, time, error = data_process(stats)
    episode = 10000

    if title == 'q learning':
        plot_reward(reward, episode, 'reward', 'forest: number of iterations vs reward: ' + title)
        plot_reward(error, episode, 'error', 'forest: number of iterations vs error: ' + title)
        plot_reward(time, episode, 'time', 'forest: number of iterations vs time: ' + title)
    else:
        plot_graph(time, 'number of iterations', 'time', 'forest: number of iterations vs time: ' + title)
        plot_graph(error, 'number of iterations', 'error', 'forest: number of iterations vs error: ' + title)
        plot_graph(reward, 'number of iterations', 'reward', 'forest: number of iterations vs reward: ' + title)

    return


def plot_heatmap(policy, title):

    plt.figure()
    sns.heatmap(policy, cmap="YlGnBu", annot=True, cbar=False)
    plt.savefig(title)
    plt.clf()

    return


if __name__ == '__main__':

    P, R = example.forest(2000)
    episode = 10000

    #pi = mdp.PolicyIteration(P, R, 0.9)
    #vi = mdp.ValueIteration(P, R, 0.9, epsilon=1e-7)
    ql = mdp.QLearning(P, R, 0.9, alpha=0.2, epsilon=0.9, n_iter=episode)

    # pi_stats = pi.run()
    # vi_stats = vi.run()
    ql_stats = ql.run()

    # plot_heatmap(np.array(pi.policy).reshape(50, 40), 'forest: policy iteration heatmap')
    # plot_heatmap(np.array(vi.policy).reshape(50, 40), 'forest: value iteration heatmap')
    plot_heatmap(np.array(ql.policy).reshape(50, 40), 'forest: q learning heatmap')

    # fun(pi_stats, 'policy iteration')
    # fun(vi_stats, 'value iteration')
    fun(ql_stats, 'q learning')


    ql1 = mdp.QLearning(P, R, 0.9, alpha=0.2, epsilon=0.3, n_iter=episode)
    ql2 = mdp.QLearning(P, R, 0.9, alpha=0.2, epsilon=0.6, n_iter=episode)
    ql3 = mdp.QLearning(P, R, 0.9, alpha=0.2, epsilon=0.9, n_iter=episode)
    ql4 = mdp.QLearning(P, R, 0.9, alpha=0.8, epsilon=0.3, n_iter=episode)
    ql5 = mdp.QLearning(P, R, 0.9, alpha=0.8, epsilon=0.6, n_iter=episode)
    ql6 = mdp.QLearning(P, R, 0.9, alpha=0.8, epsilon=0.9, n_iter=episode)

    ql1_stats = ql1.run()
    ql2_stats = ql2.run()
    ql3_stats = ql3.run()
    ql4_stats = ql4.run()
    ql5_stats = ql5.run()
    ql6_stats = ql6.run()

    reward1, time1, error1 = data_process(ql1_stats)
    reward2, time2, error2 = data_process(ql2_stats)
    reward3, time3, error3 = data_process(ql3_stats)
    reward4, time4, error4 = data_process(ql4_stats)
    reward5, time5, error5 = data_process(ql5_stats)
    reward6, time6, error6 = data_process(ql6_stats)


    data_per_thousand_episode4 = np.split(np.array(error4), episode / 1000)

    data_value_list4 = []
    counter = 1000

    for i in data_per_thousand_episode4:
        current_data_value4 = sum(i) / 1000
        data_value_list4.append(current_data_value4)
        counter = counter + 1000

    data_per_thousand_episode5 = np.split(np.array(error5), episode / 1000)

    data_value_list5 = []
    counter = 1000

    for i in data_per_thousand_episode5:
        current_data_value5 = sum(i) / 1000
        data_value_list5.append(current_data_value5)
        counter = counter + 1000

    data_per_thousand_episode6 = np.split(np.array(error6), episode / 1000)

    data_value_list6 = []
    counter = 1000

    for i in data_per_thousand_episode6:
        current_data_value6 = sum(i) / 1000
        data_value_list6.append(current_data_value6)
        counter = counter + 1000

    data_per_thousand_episode3 = np.split(np.array(error3), episode / 1000)

    data_value_list3 = []
    counter = 1000

    for i in data_per_thousand_episode3:
        current_data_value3 = sum(i) / 1000
        data_value_list3.append(current_data_value3)
        counter = counter + 1000

    data_per_thousand_episodes1 = np.split(np.array(error1), episode / 1000)

    data_value_list1 = []
    counter = 1000

    for i in data_per_thousand_episodes1:
        current_data_value1 = sum(i) / 1000
        data_value_list1.append(current_data_value1)
        counter = counter + 1000

    data_per_thousand_episodes2 = np.split(np.array(error2), episode / 1000)

    data_value_list2 = []
    counter = 1000

    for i in data_per_thousand_episodes2:
        current_data_value2 = sum(i) / 1000
        data_value_list2.append(current_data_value2)
        counter = counter + 1000

    plt.title('forest : number of iterations vs error : q learning')
    plt.plot(data_value_list1, label='lr_0.2_ep_0.3', color='green')
    plt.plot(data_value_list2, label='lr_0.2_ep_0.6', color='blue')
    plt.plot(data_value_list3, label='lr_0.2_ep_0.9', color='red')
    plt.plot(data_value_list4, label='lr_0.8_ep_0.3', color='green', linestyle='--')
    plt.plot(data_value_list5, label='lr_0.8_ep_0.6', color='blue', linestyle='--')
    plt.plot(data_value_list6, label='lr_0.8_ep_0.9', color='red', linestyle='--')
    plt.xlabel('number of iterations (in thousands)')
    plt.ylabel('error')
    plt.legend(loc='best')
    plt.savefig('forest : number of iterations vs error : q learning')
    plt.clf()
