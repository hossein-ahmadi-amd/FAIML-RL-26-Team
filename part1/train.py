import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Policy, Agent


def moving_average(data, window_size=50):
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE / Actor-Critic on Hopper-v4")

    parser.add_argument(
        "--algo",
        type=str,
        choices=["reinforce", "ac"],
        default="reinforce",
        help="Algorithm to use: reinforce or ac (Actor-Critic)"
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Constant baseline for REINFORCE (e.g. 20.0). Omit for no baseline. Ignored for ac."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000)"
    )

    args = parser.parse_args()

    # Build a descriptive run name for saving files
    if args.algo == "reinforce":
        if args.baseline is None:
            run_name = "reinforce_no_baseline"
        else:
            run_name = f"reinforce_baseline_{int(args.baseline)}"
    else:
        run_name = "actor_critic"

    print(f"Algorithm : {args.algo}")
    print(f"Baseline  : {args.baseline}")
    print(f"Episodes  : {args.episodes}")
    print(f"Run name  : {run_name}")

    env = gym.make('Hopper-v4')

    print('State space :', env.observation_space)
    print('Action space:', env.action_space)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    policy = Policy(state_space, action_space)
    agent = Agent(policy, algo=args.algo, baseline=args.baseline)

    reward_history = []

    print("Starting training...")

    for episode in range(args.episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, action_log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(
                action.detach().numpy()
            )
            done = terminated or truncated
            agent.store_outcome(state, next_state, action_log_prob, reward, done)
            state = next_state
            episode_reward += reward

        agent.update_policy()
        reward_history.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(reward_history[-50:])
            print(f"Episode {episode + 1:4d}/{args.episodes} | "
                  f"Last-50 avg reward: {avg:.1f}")

    env.close()

    # ------------------------------------------------------------------
    # Save model and rewards
    # ------------------------------------------------------------------
    torch.save(policy.state_dict(), f"{run_name}.pth")
    np.save(f"{run_name}_rewards.npy", np.array(reward_history))
    print(f"Model saved  : {run_name}.pth")
    print(f"Rewards saved: {run_name}_rewards.npy")

    # ------------------------------------------------------------------
    # Raw reward plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(reward_history, alpha=0.4, label="Raw reward")
    smoothed = moving_average(reward_history, window_size=50)
    offset = len(reward_history) - len(smoothed)
    plt.plot(range(offset, offset + len(smoothed)), smoothed,
             label="Smoothed (50-ep)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Training curve — {run_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_name}_curve.png", dpi=150)
    plt.show()
    print(f"Plot saved   : {run_name}_curve.png")

    print(f"\nFinal 100-episode average : {np.mean(reward_history[-100:]):.1f}")
    print(f"Max episode reward        : {np.max(reward_history):.1f}")


if __name__ == '__main__':
    main()
