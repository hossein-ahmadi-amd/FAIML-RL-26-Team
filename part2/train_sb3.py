import argparse
import os

import gymnasium as gym
import panda_gym  # type: ignore[import-not-found]
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from rand_wrapper import RandomizationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/SAC on PandaPush-v3")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac"],
        default="sac",
        help="RL algorithm to use: ppo or sac",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=".",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for TensorBoard logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    if args.sampling_strategy != "none":
        env = RandomizationWrapper(env, strategy=args.sampling_strategy)

    # Build save name: algo_push_strategy_envtype_timestepsk
    save_name = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    save_path = os.path.join(args.save_dir, save_name)
    log_path = os.path.join(args.log_dir, save_name)

    print(f"Algorithm  : {args.algo.upper()}")
    print(f"Strategy   : {args.sampling_strategy}")
    print(f"Env type   : {args.env_type}")
    print(f"Timesteps  : {args.timesteps}")
    print(f"Save path  : {save_path}")

    # Create model
    if args.algo == "ppo":
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_path,
        )
    else:  # sac
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_path,
        )

    # Train
    model.learn(total_timesteps=args.timesteps)

    # Save
    model.save(save_path)
    print(f"Model saved: {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
