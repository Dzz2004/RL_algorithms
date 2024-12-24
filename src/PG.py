import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(), # 使用ReLU激活函数，增加网络的非线性
            nn.Linear(128, action_space),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


# REINFORCE算法
def reinforce(env, policy_net, optimizer, num_episodes=1000, gamma=0.99):
    reward_episode = []
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        # 收集一条轨迹
        done = False
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            probs = policy_net(state) # 获取动作概率
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            new_state, reward, done, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)
            state = new_state

        # 计算返回（回报）和更新策略
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # 标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)

        optimizer.zero_grad()
        loss = torch.cat(loss).sum()  # 将loss列表中的tensor拼接成一个tensor，然后求和
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        reward_episode.append(sum(rewards))

        # 打印进度
        if episode % 100 == 0:
            print(f'Episode {episode}/{num_episodes}, Total Reward: {sum(rewards)}')

    print('Training Complete!')
    plt.plot(reward_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Training')
    plt.show()
    torch.save(policy_net.state_dict(), '../Policy/PG_policy_net.pth')

def test_policy(env, policy_net, num_episodes=100):
    policy_net.load_state_dict(torch.load('../Policy/PG_policy_net.pth'))
    policy_net.eval()

    reward_episode = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            probs = policy_net(state)
            m = Categorical(probs)
            action = m.sample()

            state, reward, done, _ = env.step(action.item())
            total_reward += reward

        reward_episode.append(total_reward)
        print(f'Episode {episode}/{num_episodes}, Total Reward: {total_reward}')
    plt.plot(reward_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Testing')
    plt.show()


if __name__ == '__main__':
    # 初始化环境
    env = gym.make('CartPole-v1')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # 创建策略网络和优化器
    policy_net = PolicyNetwork(state_space, action_space)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)

    # 运行REINFORCE算法
    reinforce(env, policy_net, optimizer)

    # 测试策略网络
    test_policy(env, policy_net)
    env.close()