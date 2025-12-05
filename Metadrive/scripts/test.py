# scripts/test.py

import gc
import os
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import scenic
import torch
from gymnasium import ObservationWrapper, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from custom.custom_gym import CustomMetaDriveEnv
from custom.custom_simulator import CustomMetaDriveSimulator

SCENIC_FILE = "./scenarios/driver.scenic"
SUMO_MAP = "./CARLA/Town01.net.xml"

max_steps = 200         # 每个 episode 最多步数，由你的环境控制
episodes = 200          # 你想大概跑多少 episode，用来设 total_timesteps
total_timesteps = max_steps * episodes

LOG_DIR = "./baseline-random-1"
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


class MetricsCallback(BaseCallback):
    """
    - 按 step 收集数据
    - 每个 episode 结束时：
        * 更新累积统计
        * 立刻 append 一行到 CSV（追加写入）
    """

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)

        # 每 episode
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_collisions = []
        self.episode_coverages = []

        # 跨 episode 累积
        self.cumulative_collisions = []
        self.cumulative_coverages = []

        self.total_collision = 0.0   # 到目前为止有碰撞的 episode 数量
        self.total_coverage = 0.0

        # 当前 episode 累积
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_collision = 0
        self._ep_coverage_sum = 0.0

        # per-episode 立刻写入的 CSV
        self.csv_path = csv_path
        # 如果文件存在，后面就不再写 header
        self._csv_initialized = os.path.exists(csv_path)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if rewards is None or dones is None or infos is None:
            return True

        # 只考虑第 0 个 env（单环境）
        r = float(rewards[0])
        done = bool(dones[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        self._ep_reward += r
        self._ep_length += 1

        if isinstance(info, dict):
            # step 级 info
            coll = int(info.get("collision", 0))  # 本 episode 是否发生过碰撞（0/1）
            if coll == 1:
                self._ep_collision = 1

            step_cov = float(info.get("coverage_step", 0.0))
            self._ep_coverage_sum += step_cov

        if done:
            # 计算这个 episode 的平均 coverage（也可以改成 total，看你需求）
            if self._ep_length > 0:
                coverage_rate = self._ep_coverage_sum / self._ep_length
            else:
                coverage_rate = 0.0

            # 写入 episode 级缓存
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self.episode_collisions.append(self._ep_collision)
            self.episode_coverages.append(coverage_rate)

            # 更新累积量：
            #   total_coverage: 累加每集的 coverage（可以理解为“到目前为止的 coverage 总和”）
            #   total_collision: 有碰撞的 episode 数
            self.total_coverage += coverage_rate
            self.cumulative_coverages.append(self.total_coverage)

            self.total_collision += self._ep_collision
            self.cumulative_collisions.append(self.total_collision)

            # ====== 立刻把这一集写到 CSV（追加模式） ======
            ep_idx = len(self.episode_rewards) - 1

            row = {
                "episode": ep_idx,
                "reward": self.episode_rewards[ep_idx],
                "length": self.episode_lengths[ep_idx],
                "collision": self.episode_collisions[ep_idx],
                "coverage": self.episode_coverages[ep_idx],
                "total_collision": self.cumulative_collisions[ep_idx],
                "total_coverage": self.cumulative_coverages[ep_idx],
            }
            df = pd.DataFrame([row])

            # 第一次写入要写 header，后面追加就不用 header
            if not self._csv_initialized:
                df.to_csv(self.csv_path, index=False, mode="w", header=True)
                self._csv_initialized = True
            else:
                df.to_csv(self.csv_path, index=False, mode="a", header=False)

            if self.verbose > 0:
                print(
                    f"[Metrics] ep {ep_idx} | "
                    f"R={row['reward']:.2f}, "
                    f"coll={row['collision']}, "
                    f"cov={row['coverage']:.3f}, "
                    f"len={row['length']}"
                )

            # reset 当前 episode 累积
            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_collision = 0
            self._ep_coverage_sum = 0.0

        return True

    def save_csv(self, filename: str):
        """
        兼容你之前的接口：如果你还想在结束时一次性再导出一份完整表，也可以用这个。
        现在主要是 _on_step 里已经在持续写 self.csv_path 了。
        """
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
    print("DEBUG total_timesteps =", total_timesteps)

    env = make_env()

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # 每个 episode 写一行到这里
    metrics_csv = os.path.join(LOG_DIR, "training_metrics_incremental.csv")
    metrics_callback = MetricsCallback(csv_path=metrics_csv, verbose=1)

    # 调小 PPO rollout 长度：n_steps=max_steps
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=LOG_DIR,
        n_steps=max_steps,     # 每次 rollout 100 步
        batch_size=max_steps,  # 和 n_steps 对齐，方便
        n_epochs=1,            # 每个 batch 只训练一遍，加快速度
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=metrics_callback,
    )

    model_path = os.path.join(LOG_DIR, "ppo_metadrive_model")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # 如果你还想最终导出一份“汇总版”的 CSV，可以用之前的接口：
    final_metrics_csv = os.path.join(LOG_DIR, "training_metrics_final.csv")
    metrics_callback.save_csv(final_metrics_csv)
    print(f"Final training metrics saved to: {final_metrics_csv}")

    env.close()
    gc.collect()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")
