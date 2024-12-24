import gym
import numpy as np
import matplotlib.pyplot as plt


def create_bins(num_bins, lower_bound, upper_bound):
    return np.linspace(lower_bound, upper_bound, num_bins)


def discretize_state(state, cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins):
    cart_pos_idx = np.digitize(state[0], cart_pos_bins) - 1
    cart_vel_idx = np.digitize(state[1], cart_vel_bins) - 1
    pole_angle_idx = np.digitize(state[2], pole_angle_bins) - 1
    pole_vel_idx = np.digitize(state[3], pole_vel_bins) - 1
    return cart_pos_idx, cart_vel_idx, pole_angle_idx, pole_vel_idx


def choose_action(state, q_table, n_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(q_table[state])


def sarsa(env, n_episodes, lr, gamma, epsilon, epsilon_decay, epsilon_min, number_of_bins):
    n_actions = env.action_space.n

    # Create bins for each dimension of the state space
    cart_pos_bins = create_bins(number_of_bins, -4.8, 4.8)
    cart_vel_bins = create_bins(number_of_bins, -3.5, 3.5)
    pole_angle_bins = create_bins(number_of_bins, -0.418, 0.418)
    pole_vel_bins = create_bins(number_of_bins, -3.5, 3.5)

    # Initialize Q-table
    q_table = np.zeros((number_of_bins, number_of_bins, number_of_bins, number_of_bins, n_actions))

    # Main SARSA loop
    for episode in range(n_episodes):
        state = discretize_state(env.reset(), cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins)
        done = False
        action = choose_action(state, q_table, n_actions, epsilon)

        while not done:
            next_state_raw, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state_raw, cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins)
            next_action = choose_action(next_state, q_table, n_actions, epsilon)

            # Update Q-table
            if not done:
                td_target = reward + gamma * q_table[next_state][next_action]
            else:
                td_target = -100
            td_error = td_target - q_table[state][action]
            q_table[state][action] += lr * td_error

            state = next_state
            action = next_action

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % 50 == 0:
            print(f'Episode: {episode + 1}, Epsilon: {epsilon:.3f}')

    return q_table

def test(env, q_table, number_of_bins, n_episodes=100):
    # Create bins for each dimension of the state space (should match training)
    cart_pos_bins = create_bins(number_of_bins, -4.8, 4.8)
    cart_vel_bins = create_bins(number_of_bins, -3.5, 3.5)
    pole_angle_bins = create_bins(number_of_bins, -0.418, 0.418)
    pole_vel_bins = create_bins(number_of_bins, -3.5, 3.5)

    total_rewards = []

    for _ in range(n_episodes):
        state = discretize_state(env.reset(), cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins)
        done = False
        total_reward = 0

        while not done:
            action = np.argmax(q_table[state])  # Select action with the highest Q value
            next_state_raw, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state_raw, cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins)

            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

    average_reward = np.mean(total_rewards)
    print(f'Average reward over {n_episodes} episodes: {average_reward:.2f}')

    return average_reward,total_rewards

def test_lr(env, n_episodes, gamma, epsilon, epsilon_decay, epsilon_min, number_of_bins):
    lrs = [0.1, 0.2,0.3, 0.4,0.5, 0.6,0.7,0.8, 0.9]
    avg_rewards = []
    for lr in lrs:
        q_table = sarsa(env, n_episodes, lr, gamma, epsilon, epsilon_decay, epsilon_min, number_of_bins)
        avg_reward, _ = test(env, q_table, number_of_bins)
        avg_rewards.append(avg_reward)
        print(f'Average reward with lr={lr}: {avg_reward:.2f}')
    plt.plot(lrs, avg_rewards)
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Reward')
    plt.savefig('../SARSA_results/Average reward vs Learning Rate.png')
    plt.show()

def test_gamma(env, n_episodes, lr, epsilon, epsilon_decay, epsilon_min, number_of_bins):
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    avg_rewards = []
    for gamma in gammas:
        q_table = sarsa(env, n_episodes, lr, gamma, epsilon, epsilon_decay, epsilon_min, number_of_bins)
        avg_reward, _ = test(env, q_table, number_of_bins)
        avg_rewards.append(avg_reward)
        print(f'Average reward with gamma={gamma}: {avg_reward:.2f}')
    plt.plot(gammas, avg_rewards,label=f"lr={lr},epsilon={epsilon},epsilon_decay={epsilon_decay},epsilon_min={epsilon_min},bins={number_of_bins}")
    plt.xlabel('Discount Factor')
    plt.ylabel('Average Reward')
    plt.show()

def test_epsilon(env, n_episodes, lr, gamma, epsilon_decay, epsilon_min, number_of_bins):
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    avg_rewards = []
    for epsilon in epsilons:
        q_table = sarsa(env, n_episodes, lr, gamma, epsilon, epsilon_decay, epsilon_min, number_of_bins)
        avg_reward, _ = test(env, q_table, number_of_bins)
        avg_rewards.append(avg_reward)
        print(f'Average reward with epsilon={epsilon}: {avg_reward:.2f}')
    plt.plot(epsilons, avg_rewards)
    plt.xlabel('Epsilon')
    plt.ylabel('Average Reward')
    plt.savefig('../SARSA_results/Average reward vs Epsilon.png')
    plt.show()

if __name__ == "__main__":
    # Create CartPole environment
    env = gym.make('CartPole-v1')

    # SARSA parameters
    n_episodes = 4000 # Number of episodes to train
    lr = 0.1 # Learning rate
    gamma = 0.99  #Discount factor
    epsilon = 1.0 # Epsilon-greedy parameter
    epsilon_decay = 0.995 # Epsilon decay factor
    epsilon_min = 0 # Minimum epsilon value
    number_of_bins = 10 # Number of bins to discretize each dimension of the state space

    #Train the agent

    q_table = sarsa(env, n_episodes, lr, gamma, epsilon, epsilon_decay, epsilon_min, number_of_bins)

    # Test the trained agent
    avg_reward, total_rewards = test(env, q_table, number_of_bins)

    plt.plot(total_rewards, label=f"lr={lr},gamma={gamma},epsilon={epsilon},epsilon_decay={epsilon_decay},epsilon_min={epsilon_min},bins={number_of_bins}")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Average reward: {avg_reward:.2f}')
    plt.show()
    env.close()

    # test_lr(env, n_episodes, gamma, epsilon, epsilon_decay, epsilon_min, number_of_bins)
    # test_gamma(env, n_episodes, lr, epsilon, epsilon_decay, epsilon_min, number_of_bins)
    # test_epsilon(env, n_episodes, lr, gamma, epsilon_decay, epsilon_min, number_of_bins)


