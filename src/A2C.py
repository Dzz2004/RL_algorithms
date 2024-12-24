import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class A2C:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.tensor(transition_dict['actions']).reshape((-1, 1)).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).reshape((-1, 1)).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).reshape((-1, 1)).to(self.device)

        # Compute TD target
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        # Actor Loss
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = -(log_probs * td_delta.detach()).mean()

        # Critic Loss
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())

        # Update Actor Network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Critic Network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    return middle


def train():
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2C(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    env.close()
    torch.save(agent.actor.state_dict(), '../Policy/A2C_actor.pth')
    torch.save(agent.critic.state_dict(), '../Policy/A2C_critic.pth')


    plt.plot(range(len(return_list)), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'A2C on {env_name}')
    plt.savefig('../A2C_results/A2C_training.png')
    plt.show()


def test():
    num_episodes = 100
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = A2C(state_dim, hidden_dim, action_dim, 0, 0, 0, device)
    agent.actor.load_state_dict(torch.load('../Policy/A2C_actor.pth'))
    agent.critic.load_state_dict(torch.load('../Policy/A2C_critic.pth'))
    agent.actor.eval()
    agent.critic.eval()

    return_list = []
    for i in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            state, reward, done, _ = env.step(action)
            episode_return += reward
        return_list.append(episode_return)
        print(f'Episode {i + 1} Return: {episode_return}') if (i + 1) % 10 == 0 else None

    average_return = np.mean(return_list)
    print(f'Average Return: {average_return}')

    plt.plot(range(len(return_list)), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'A2C on {env_name}, Average Return: {average_return}')
    plt.savefig('../A2C_results/A2C_testing.png')
    plt.show()

    env.close()


if __name__ == '__main__':
    train()
    test()