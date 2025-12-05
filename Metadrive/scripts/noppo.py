# scripts/test_no_ppo.py

import gc
import os
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import scenic
import torch
from gymnasium import ObservationWrapper, spaces
from stable_baselines3.common.monitor import Monitor

from custom.custom_gym import CustomMetaDriveEnv
from custom.custom_simulator import CustomMetaDriveSimulator

SCENIC_FILE = "./scenarios/driver.scenic"
SUMO_MAP = "./CARLA/Town01.net.xml"

max_steps = 100
episodes = 20

LOG_DIR = "./logs_metadrive"
os.makedirs(LOG_DIR, exist_ok=True)


class AutoBoxObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs, _info = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32)


class MetricsCallback:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_collisions = []
        self.episode_coverages = []

        self.cumulative_collisions = []
        self.cumulative_coverages = []

        self.total_collision = 0.0
        self.total_coverage = 0.0

        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_collision = 0
        self._ep_coverage_sum = 0.0

    def step(self, reward, done, info):
        r = float(reward)
        done = bool(done)
        info = info if isinstance(info, dict) else {}

        self._ep_reward += r
        self._ep_length += 1

        coll = int(info.get("collision", 0))
        if coll == 1:
            self._ep_collision = 1

        step_cov = float(info.get("coverage_step", 0.0))
        self._ep_coverage_sum += step_cov

        if done:
            if self._ep_length > 0:
                coverage_rate = self._ep_coverage_sum / self._ep_length
            else:
                coverage_rate = 0.0

            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self.episode_collisions.append(self._ep_collision)
            self.episode_coverages.append(coverage_rate)

            self.total_coverage += coverage_rate
            self.cumulative_coverages.append(self.total_coverage)

            self.total_collision += self._ep_collision
            self.cumulative_collisions.append(self.total_collision)

            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_collision = 0
            self._ep_coverage_sum = 0.0

    def save_csv(self, filename: str):
        n = min(
            len(self.episode_rewards),
            len(self.episode_lengths),
            len(self.episode_collisions),
            len(self.episode_coverages),
            len(self.cumulative_collisions),
            len(self.cumulative_coverages),
        )

        if n == 0:
            print("No finished episodes, skip saving metrics.")
            return

        data = {
            "episode": np.arange(n),
            "reward": self.episode_rewards[:n],
            "length": self.episode_lengths[:n],
            "collision": self.episode_collisions[:n],
            "coverage": self.episode_coverages[:n],
            "total_collision": self.cumulative_collisions[:n],
            "total_coverage": self.cumulative_coverages[:n],
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)


def make_env():
    scenario = scenic.scenarioFromFile(
        SCENIC_FILE,
        model="scenic.simulators.metadrive.model",
        mode2D=True,
    )

    action_space = spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        shape=(2,),
        dtype=np.float32,
    )

    env = CustomMetaDriveEnv(
        scenario=scenario,
        simulator=CustomMetaDriveSimulator(
            sumo_map=SUMO_MAP,
            max_steps=max_steps,
        ),
        max_steps=max_steps,
        action_space=action_space,
        file=SCENIC_FILE,
    )

    env = AutoBoxObsWrapper(env)

    env = Monitor(
        env,
        filename=os.path.join(LOG_DIR, "monitor.csv"),
        info_keywords=("collision", "coverage_total"),
    )

    return env


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    metrics = MetricsCallback()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0

        while not (done or truncated) and step < max_steps:
            # 随机动作 / 也可以改成你自己的策略
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            metrics.step(reward, done, info)
            step += 1

        if metrics.episode_rewards:
            idx = len(metrics.episode_rewards) - 1
            print(
                f"[Rollout] ep {ep + 1} | "
                f"R={metrics.episode_rewards[idx]:.2f}, "
                f"coll={metrics.episode_collisions[idx]}, "
                f"cov={metrics.episode_coverages[idx]:.3f}, "
                f"len={metrics.episode_lengths[idx]}"
            )

    metrics_csv = os.path.join(LOG_DIR, "rollout_metrics.csv")
    metrics.save_csv(metrics_csv)
    print(f"Rollout metrics saved to: {metrics_csv}")

    env.close()
    gc.collect()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")
