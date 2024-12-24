import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm


class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, eps_start, eps_end, eps_decay, target_update,
                 memory_capacity, batch_size, device):
        self.q_network = QNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.epsilon = self.eps_start
        self.target_update = target_update
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size
        self.device = device
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                q_values = self.q_network(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_dim)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, step):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        curr_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach().unsqueeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(curr_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.eps_end, self.eps_start - step / self.eps_decay)

        if step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train():
    lr = 1e-3
    num_episodes = 250
    hidden_dim = 128
    gamma = 0.98
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 500
    target_update = 10
    memory_capacity = 10000
    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, eps_start, eps_end, eps_decay, target_update,
                memory_capacity, batch_size, device)

    return_list = []
    step = 0

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.store_transition(state, action, reward, next_state, done)
                    agent.update(step)
                    step += 1
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                    if np.mean(return_list[-10:]) > 490:
                        torch.save(agent.q_network.state_dict(), '../Policy/DQN_net_peak.pth')
                pbar.update(1)

    # 保存模型
    torch.save(agent.q_network.state_dict(), '../Policy/DQN_net.pth')

    # 画出训练结果
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'DQN on {env_name}')
    plt.savefig('../DQN_results/DQN_training.png')
    plt.show()


    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'DQN on {env_name}')
    plt.show()


    env.close()


def test():
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DQN(state_dim, hidden_dim, action_dim, 0, 0, 0, 0, 0, 0, 0, 0, device)
    agent.q_network.load_state_dict(torch.load('../Policy/DQN_net.pth'))
    agent.q_network.eval()

    num_episodes = 100
    return_list = []

    for i_episode in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        print(f'Episode {i_episode}/{num_episodes}, Total Reward: {episode_return}') if i_episode % 10 == 0 else None

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'DQN on {env_name}')
    plt.savefig('../DQN_results/DQN_testing.png')
    plt.show()

    env.close()


if __name__ == '__main__':
    train()
    test()