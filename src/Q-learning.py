import numpy as np
import gym
from collections import defaultdict
from matplotlib import pyplot as plt


# 离散化函数，将连续状态空间转为离散状态空间
def discretize_state(state, bins):
    state_bins = [
        np.linspace(-4.8, 4.8, bins[0] + 1)[1:-1],
        np.linspace(-4, 4, bins[1] + 1)[1:-1],
        np.linspace(-0.418, 0.418, bins[2] + 1)[1:-1],
        np.linspace(-4, 4, bins[3] + 1)[1:-1],
    ]
    return tuple(np.digitize(s, b) for s, b in zip(state, state_bins))

def choose_action(state, Q, epsilon, n_actions):
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

# Q-learning训练函数
def train_q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1, n_bins=(6, 6, 6, 6)):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    reward_episodes = []
    loss_episodes = []
    for episode in range(n_episodes):
        state = env.reset()
        state = discretize_state(state, n_bins)
        total_reward = 0
        done = False
        losses = []
        while not done:
            # epsilon-greedy 策略选择动作
            action = choose_action(state, Q, epsilon, env.action_space.n)

            # 执行动作
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state, n_bins)
            total_reward += reward

            # 更新 Q 值
            if not done:
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + gamma * Q[next_state][best_next_action]
            else:
                td_target = reward
            td_error = td_target - Q[state][action]
            losses.append(alpha*td_error)
            Q[state][action] += alpha * td_error

            # 移动到下一个状态
            state = next_state
        reward_episodes.append(total_reward)
        loss_episodes.append(max(losses))

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    plt.plot(reward_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Q-learning Training')
    # plt.savefig('../Q-learning_results/Q-learning Training.png')
    plt.show()

    return Q


# 测试函数
def test_policy(env, Q, n_episodes=100, n_bins=(6, 6, 6, 6)):
    avg_reward = 0
    reward_episodes = []
    for episode in range(n_episodes):
        state = env.reset()
        state = discretize_state(state, n_bins)
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
            state = discretize_state(state, n_bins)
            total_reward += reward
            # env.render()
        if episode % 10 == 0:
            print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")
        avg_reward += total_reward
        reward_episodes.append(total_reward)
    avg_reward /= n_episodes
    print(f"Average Reward: {avg_reward}")
    plt.plot(reward_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Q-learning Test, Average Reward: {avg_reward}')
    # plt.savefig('../Q-learning_results/Q-learning Test.png')
    plt.show()
    return avg_reward, reward_episodes


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # hyper parameters
    n_episodes = 3000 # 训练次数
    alpha = 0.1  # 学习率
    gamma = 0.99 # 折扣因子
    epsilon = 0.1 # epsilon-greedy 策略参数
    num_bin = 6  # 状态空间离散化的分箱数
    n_bins = (num_bin, num_bin, num_bin, num_bin)

    learned_Q = train_q_learning(env, n_episodes, alpha, gamma, epsilon, n_bins)
    test_policy(env, learned_Q)