import gym
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import time


def plot_graph(data_list, label, title):

    data_per_thousand_episodes = np.split(np.array(data_list),
                                             10000 / 1000)

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


if __name__ == '__main__':

    env = gym.make("FrozenLake-v0").env
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    Q_table = np.zeros((state_space_size, action_space_size))

    num_of_episodes = 10000
    max_steps_per_episode = 200

    learning_rate = 0.5
    discount_rate = 0.99

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.1
    exploration_decay_rate = 0.0001

    rewards_of_all_episodes = []
    time_of_all_episodes = []
    error_of_all_episodes = []
    q_table_of_all_episode = []

    for episode in range(num_of_episodes):

        state = env.reset()
        count = 0
        done = False
        rewards_current_episode = 0
        t1 = time.time()
        error_current_episode = []

        for step in range(max_steps_per_episode):

            random_number = random.uniform(0, 1)

            if random_number > exploration_rate:
                action = np.argmax(Q_table[state, :])

            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            Q_table[state, action] = Q_table[state, action] * (1 - learning_rate) \
                                     + learning_rate * (reward + discount_rate * np.max(Q_table[new_state, :]))
            dQ = learning_rate * (reward + discount_rate * np.max(Q_table[new_state, :]) - Q_table[state, action])

            error_current_episode.append(np.absolute(dQ))

            state = new_state
            rewards_current_episode = rewards_current_episode + reward
            count += 1

            if done is True:
                break

            exploration_rate = min_exploration_rate + \
                           (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        t2= time.time()
        q_table_of_all_episode.append(Q_table.argmax(axis=1))
        rewards_of_all_episodes.append(rewards_current_episode)
        time_of_all_episodes.append(t2-t1)
        error_of_all_episodes.append(np.mean(error_current_episode))

    plot_graph(time_of_all_episodes, 'time', 'frozenlake: number of iterations vs time: q learning')
    plot_graph(rewards_of_all_episodes, 'rewards', 'frozenlake: number of iterations vs reward: q learning')
    plot_graph(error_of_all_episodes, 'error', 'frozenlake: number of iterations vs error: q learning')
    '''
    plt.figure()
    sns.heatmap(q_table_of_all_episode, cmap="YlGnBu", annot=True, cbar=False)
    plt.savefig('frozenlake: qlearning heatmap')
    plt.clf()
    '''
