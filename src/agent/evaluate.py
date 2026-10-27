"""
Evaluation script to benchmark trained PPO model.
Calculates Success Rate, Collision Rate, and Mean Reward over multiple episodes.
"""

import os
from typing import Dict, Any
import numpy as np
from stable_baselines3 import PPO

from src.local_env.environment import LocalDrivingEnv


def evaluate_agent(
    model_path: str = "models/saved_models/ppo_autonomous_driver.zip",
    num_episodes: int = 10
) -> Dict[str, float]:
    """Run benchmark episodes and return key metrics."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    model = PPO.load(model_path)
    env = LocalDrivingEnv(
        waypoints=[(0, 0), (60, 0), (120, 40), (160, 100), (100, 150), (0, 100), (0, 0)],
        num_npcs=3
    )

    success_count = 0
    collision_count = 0
    total_rewards = []
    episode_lengths = []

    print(f"--- Running evaluation over {num_episodes} test episodes ---")
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

            if terminated:
                if info.get("reached_destination", False):
                    success_count += 1
                if info.get("is_crashed", False):
                    collision_count += 1

        total_rewards.append(ep_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1:02d}: Steps={steps}, Reward={ep_reward:.1f}, Reached={info.get('reached_destination', False)}, Crashed={info.get('is_crashed', False)}")

    metrics = {
        "success_rate": float(success_count / num_episodes),
        "collision_rate": float(collision_count / num_episodes),
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_length": float(np.mean(episode_lengths))
    }

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Success Rate:  {metrics['success_rate'] * 100:.1f}%")
    print(f"Collision Rate: {metrics['collision_rate'] * 100:.1f}%")
    print(f"Mean Reward:   {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"Mean Steps:    {metrics['mean_length']:.1f}")

    return metrics


if __name__ == "__main__":
    evaluate_agent()
