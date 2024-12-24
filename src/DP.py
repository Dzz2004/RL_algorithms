import numpy as np
import gym
import matplotlib.pyplot as plt

def create_bins(num_bins, lower_bound, upper_bound):
    return np.linspace(lower_bound, upper_bound, num_bins)


def discretize_state(state, cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins):
    cart_pos_idx = np.digitize(state[0], cart_pos_bins) - 1
    cart_vel_idx = np.digitize(state[1], cart_vel_bins) - 1
    pole_angle_idx = np.digitize(state[2], pole_angle_bins) - 1
    pole_vel_idx = np.digitize(state[3], pole_vel_bins) - 1
    return cart_pos_idx, cart_vel_idx, pole_angle_idx, pole_vel_idx



def q_value_iteration(env, num_bins, gamma=1.0, theta=1e-5):
    cart_pos_bins = create_bins(num_bins, -2.4, 2.4)
    cart_vel_bins = create_bins(num_bins, -3.0, 3.0)
    pole_angle_bins = create_bins(num_bins, -0.5, 0.5)
    pole_vel_bins = create_bins(num_bins, -2.0, 2.0)

    action_space_size = env.action_space.n
    Q = np.zeros((num_bins, num_bins, num_bins, num_bins, action_space_size))
    V = np.zeros((num_bins, num_bins, num_bins, num_bins))
    i = 0
    while True:
        delta = 0
        print(f"Iteration{i}")
        for cart_pos_idx in range(num_bins):
            for cart_vel_idx in range(num_bins):
                for pole_angle_idx in range(num_bins):
                    for pole_vel_idx in range(num_bins):
                        for action in range(action_space_size):
                            state_values = []
                            env.reset()
                            env.env.state = (cart_pos_bins[cart_pos_idx],
                                             cart_vel_bins[cart_vel_idx],
                                             pole_angle_bins[pole_angle_idx],
                                             pole_vel_bins[pole_vel_idx])
                            next_state, reward, done, _ = env.step(action)
                            if done:
                                Q_value = reward
                            else:
                                next_state_discrete = discretize_state(next_state, cart_pos_bins, cart_vel_bins,
                                                                       pole_angle_bins, pole_vel_bins)

                                Q_value = reward + gamma * np.max(Q[next_state_discrete])

                            state_values.append(Q_value)

                            old_value = Q[cart_pos_idx, cart_vel_idx, pole_angle_idx, pole_vel_idx, action]
                            Q[cart_pos_idx, cart_vel_idx, pole_angle_idx, pole_vel_idx, action] = Q_value
                            delta = max(delta, np.abs(old_value - Q_value))

        if delta < theta:
            break
        else:
            print(f"iteration {i} complete,delta={delta}")
            i += 1

    policy = np.argmax(Q, axis=-1)
    return policy, Q


def test(env, policy, num_bins):
    state = env.reset()
    cart_pos_bins = create_bins(num_bins, -2.4, 2.4)
    cart_vel_bins = create_bins(num_bins, -3.0, 3.0)
    pole_angle_bins = create_bins(num_bins, -0.5, 0.5)
    pole_vel_bins = create_bins(num_bins, -2.0, 2.0)

    total_rewards = []
    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_discrete = discretize_state(state, cart_pos_bins, cart_vel_bins, pole_angle_bins, pole_vel_bins)
            action = policy[state_discrete]
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)

    return np.mean(total_rewards), total_rewards

def test_hyperParameter(env,test_bin=False,test_gamma=False,test_theta=False,):
    bin = 5
    gamma = 0.8
    theta = 1e-10
    bins = [1,2,3,4,5,6,7,8,9,10]
    avg_rewards = []
    if test_bin:
        for bin in bins:
            policy, Q = q_value_iteration(env, bin, gamma, theta)
            avg_reward, total_rewards = test(env, policy, bin)
            avg_rewards.append(avg_reward)
            print(f"Average reward after 100 test episodes with {bin} bins: {avg_reward}")
        plt.plot(bins,avg_rewards)
        plt.xlabel("num_bins")
        plt.ylabel("Average reward")
        plt.title("Average reward vs num_bins")
        plt.savefig('../DP_results/Average reward vs num_bins.png')
        plt.show()
        # 保存图像至父目录下DP_results文件夹

    if test_gamma:
        gammas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        for gamma in gammas:
            policy, Q = q_value_iteration(env, bin, gamma, theta)
            avg_reward, total_rewards = test(env, policy, bin)
            avg_rewards.append(avg_reward)
            print(f"Average reward after 100 test episodes with gamma={gamma}: {avg_reward}")
        plt.plot(gammas,avg_rewards,label="bins=5,theta=1e-10")
        plt.xlabel("gamma")
        plt.ylabel("Average reward")
        plt.title("Average reward vs gamma")
        plt.savefig('../DP_results/Average reward vs gamma.png')
        plt.show()

    if test_theta:
        thetas = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
        for theta in thetas:
            policy, Q = q_value_iteration(env, bin, gamma, theta)
            avg_reward, total_rewards = test(env, policy, bin)
            avg_rewards.append(avg_reward)
            print(f"Average reward after 100 test episodes with theta={theta}: {avg_reward}")
        plt.plot(thetas,avg_rewards,label="bins=5,gamma=0.5")
        plt.xlabel("theta")
        plt.ylabel("Average reward")
        plt.title("Average reward vs theta")
        plt.savefig('../DP_results/Average reward vs theta.png')
        plt.show()






if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    num_bin = 5
    gamma = 0.5
    theta = 1e-10
    # test_hyperParameter(env, test_bin=True, test_gamma=False, test_theta=False)
    print(f"Running Q-value iteration with {num_bin} bins and gamma={gamma}")
    policy, Q = q_value_iteration(env, num_bin, gamma, theta)
    print("Training complete.")
    avg_reward, total_rewards = test(env, policy, num_bin)
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"gamma={gamma}, bins={num_bin}, average reward={avg_reward}")
    plt.show()
    env.close()