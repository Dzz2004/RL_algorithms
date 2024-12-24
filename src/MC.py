import numpy as np
import gym
import matplotlib.pyplot as plt
from numpy.ma.extras import average


def create_bins(num_bins, lower_bound, upper_bound):
    return np.linspace(lower_bound, upper_bound, num_bins)


def discretize_state(state, cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins):
    cart_pos_idx = np.digitize(state[0], cart_pos_bins) - 1
    cart_vel_idx = np.digitize(state[1], cart_vel_bins) - 1
    pole_angle_idx = np.digitize(state[2], pole_angle_bins) - 1
    pole_vel_idx = np.digitize(state[3], pole_vel_bins) - 1
    return cart_pos_idx, cart_vel_idx, pole_angle_idx, pole_vel_idx


def monte_carlo_learning(env, num_episodes, num_bins, gamma=1.0):
    # 为CartPole特性创建bin
    cart_pos_bins = create_bins(num_bins, -2.4, 2.4)
    cart_vel_bins = create_bins(num_bins, -3.0, 3.0)
    pole_angle_bins = create_bins(num_bins, -0.5, 0.5)
    pole_vel_bins = create_bins(num_bins, -2.0, 2.0)

    # 初始化策略和Q-table
    action_space_size = env.action_space.n
    Q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, action_space_size))
    returns_count = np.zeros((num_bins, num_bins, num_bins, num_bins, action_space_size))
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_states_actions_rewards = []
        done = False
        total_reward = 0
        while not done:
            state_discrete = discretize_state(state, cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins)
            action = np.random.choice(env.action_space.n)  # 随即选择动作

            # 执行动作，观察结果
            next_state, reward, done, _ = env.step(action)
            episode_states_actions_rewards.append((state_discrete, action, reward))
            state = next_state

        G = 0
        for state_discrete, action, reward in reversed(episode_states_actions_rewards):
            total_reward += reward
            G = gamma * G + reward
            sa_pair = (*state_discrete, action)
            returns_count[sa_pair] += 1
            Q_table[sa_pair] += (G-Q_table[sa_pair]) / returns_count[sa_pair]

        rewards.append(total_reward)

        if episode % 1000 == 0:
            print(f"Episode {episode} complete.")

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Monte Carlo Learning, bins={num_bins}, gamma={gamma}')
    plt.savefig('../MC_results/monte_carlo_learning.png')
    plt.show()
    return Q_table

def get_policy(Q_table):
    return np.argmax(Q_table, axis=-1)

def test(env,policy,num_bins):
    state = env.reset()
    cart_pos_bins = create_bins(num_bins, -2.4, 2.4)
    cart_vel_bins = create_bins(num_bins, -3.0, 3.0)
    pole_angle_bins = create_bins(num_bins, -0.5, 0.5)
    pole_vel_bins = create_bins(num_bins, -2.0, 2.0)

    avg_reward = 0
    num_test_episodes = 100
    rewards = []
    for episode in range(num_test_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state_discrete = discretize_state(state, cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins)
            action = policy[state_discrete]
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        avg_reward += episode_reward
        if episode % 10 == 0:
            print(f"test: Episode {episode} reward: {episode_reward}")
    avg_reward /= num_test_episodes
    print(f"Average reward: {avg_reward}")
    return avg_reward, rewards


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    num_bin = 5
    num_episodes = 6000
    gamma = 0.5


    print(f"Training with {num_bin} bins and gamma={gamma}")
    Q_table = monte_carlo_learning(env, num_episodes, num_bin, gamma)
    policy = get_policy(Q_table)
    print("Training complete.")
    average_reward,rewards = test(env, policy, num_bin)

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'gamma={gamma}, bins={num_bin}, average reward={average_reward}')
    plt.show()
    env.close()