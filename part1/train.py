
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Policy, Agent

def main():
    env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    # Get dimensions
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    # Create policy network
    policy = Policy(state_space, action_space)

    # Create RL agent
    agent = Agent(policy)

    # Number of training episodes
    num_episodes = 1000

    # Store rewards for plotting
    reward_history = []

    # Training loop
    for episode in range(num_episodes):

        # Reset environment
        state, _ = env.reset()

        done = False

        episode_reward = 0

        # Run one episode
        while not done:

            # Get action from policy
            action, action_log_prob = agent.get_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(
                action.detach().numpy()
            )

            # Check episode termination
            done = terminated or truncated

            # Store trajectory information
            agent.store_outcome(
                state,
                next_state,
                action_log_prob,
                reward,
                done
            )

            # Move to next state
            state = next_state

            # Accumulate reward
            episode_reward += reward

        # Update policy after episode ends
        agent.update_policy()

        # Store reward history
        reward_history.append(episode_reward)

        # Print progress
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    # Plot rewards
    plt.plot(reward_history)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("REINFORCE Training on Hopper")

    plt.savefig("final_raw_rewards.png")

    plt.show()

    # ----------------------------
    # Smoothed Reward Plot
    # ----------------------------

    def moving_average(data, window_size=50):

       return np.convolve(
           data,
           np.ones(window_size)/window_size,
           mode='valid'
       )

    smoothed_rewards = moving_average(reward_history)

    plt.plot(smoothed_rewards)

    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title("Smoothed Actor-Critic Training")

    plt.savefig("final_smoothed_rewards.png")

    plt.show()

    torch.save(
    policy.state_dict(),
       "final_actor_critic.pth"
    )

    np.save(
       "final_rewards.npy",
        reward_history
    )

    print("Average Reward:", np.mean(reward_history))

    print("Max Reward:", np.max(reward_history))


if __name__ == '__main__':
    main()