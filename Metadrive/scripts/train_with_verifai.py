# scripts/train_with_verifai.py

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
from stable_baselines3.common.monitor import Monitor

from custom.custom_gym import CustomMetaDriveEnv
from custom.custom_simulator import CustomMetaDriveSimulator

# NOTE: in your Scenic file, set:
#   param verifaiSamplerType = 'ce'
# or:
#   param verifaiSamplerType = 'bo'

SCENIC_FILE = "./scenarios/driver.scenic"
SUMO_MAP = "./CARLA/Town01.net.xml"

max_steps = 200         # 每个 episode 最多步数，由你的环境控制
episodes = 250          # 你想大概跑多少 episode，用来设 total_timesteps
total_timesteps = max_steps * episodes

LOG_DIR = "./verifai-ce-bo"
os.makedirs(LOG_DIR, exist_ok=True)

# 前多少个 episode 只 random，不启用 VerifAI sampler（CE/BO）
WARMUP_EPISODES = 50


def compute_verifai_feedback(
    ep_reward: float,
    ep_len: int,
    max_steps: int,
    collisions: int,
    coverage: float,
) -> float:
    """
    把 RL 的 episode 统计量映射成 VerifAI 需要的标量 feedback（VerifAI 会尝试最小化它）。

    设计思路：
      - 撞车多、提前结束、reward 很负、coverage 低 => 难度高
      - CE/BO sampler 最小化 feedback，所以我们取 feedback = -difficulty
    """
    # 可调超参数
    w_coll = 3.0       # 撞车权重
    w_early = 2.0      # 提前结束权重
    w_negR = 1.0       # 负 reward 权重（reward 有正有负）
    w_cov = 0.5        # 覆盖不足惩罚
    R_norm = 100.0     # reward 归一化尺度
    cov_target = 0.7   # 期望 coverage 下限

    # 1) 撞车：有碰撞就认为更“难”
    diff_coll = w_coll * (1.0 if collisions > 0 else 0.0)

    # 2) 提前结束：越早结束越“难”
    diff_early = w_early * max(0.0, 1.0 - ep_len / float(max_steps))

    # 3) 负 reward：更负代表更难
    diff_negR = w_negR * max(0.0, -ep_reward / R_norm)

    # 4) 覆盖不足：如果 coverage 低于 cov_target 就惩罚
    diff_cov = w_cov * max(0.0, cov_target - coverage)

    difficulty = diff_coll + diff_early + diff_negR + diff_cov

    # VerifAI sampler 最小化 feedback，我们希望“难度越大，场景越受关注”
    # 所以 feedback = -difficulty
    return -difficulty


class AutoBoxObsWrapper(ObservationWrapper):
    """把任意形状的 numpy obs 转成 float32 Box，方便 SB3 用 MlpPolicy."""

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
        * 计算 VerifAI feedback，并在 warm-up 后启用 CE/BO sampler
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

        # 单环境：只看第 0 个 env
        r = float(rewards[0])
        done = bool(dones[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        # 累积 episode reward / length
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
            # 计算这个 episode 的平均 coverage（也可以改成 total）
            if self._ep_length > 0:
                coverage_rate = self._ep_coverage_sum / float(self._ep_length)
            else:
                coverage_rate = 0.0

            # 写入 episode 级缓存
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self.episode_collisions.append(self._ep_collision)
            self.episode_coverages.append(coverage_rate)

            # 更新累积量
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

            # ====== 计算 VerifAI feedback + 控制启用时机 ======
            try:
                feedback = compute_verifai_feedback(
                    ep_reward=row["reward"],
                    ep_len=row["length"],
                    max_steps=max_steps,
                    collisions=row["collision"],
                    coverage=row["coverage"],
                )

                # 从 VecEnv / Monitor / Wrapper 里拿到底层 CustomMetaDriveEnv
                base_env = self.training_env.envs[0]
                raw_env = base_env.unwrapped  # 解包出真正的 CustomMetaDriveEnv

                if hasattr(raw_env, "set_verifai_feedback"):
                    raw_env.set_verifai_feedback(feedback)

                # warm-up 结束后，打开 VerifAI（CE/BO sampler）
                if hasattr(raw_env, "enable_verifai") and (ep_idx + 1 >= WARMUP_EPISODES):
                    raw_env.enable_verifai(True)

            except Exception as e:
                if self.verbose > 0:
                    print(f"[MetricsCallback] VerifAI feedback update failed: {e}")

            # reset 当前 episode 累积
            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_collision = 0
            self._ep_coverage_sum = 0.0

        return True

    def save_csv(self, filename: str):
        """如果你想最后再导出一份汇总 CSV，可以用这函数。"""
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

    # 每个 episode 写一行 metrics
    metrics_csv = os.path.join(LOG_DIR, "training_metrics_incremental.csv")
    metrics_callback = MetricsCallback(csv_path=metrics_csv, verbose=1)

    # PPO 设置：reward 有正有负没问题，PPO 会自己处理 advantage
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=LOG_DIR,
        n_steps=max_steps,       # 每个 rollout 正好对应一个 episode
        batch_size=max_steps,    # 一个 batch 就是一整集
        n_epochs=4,              # 每个 batch 训练几轮
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=metrics_callback,
    )

    model_path = os.path.join(LOG_DIR, "ppo_metadrive_verifai")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    final_metrics_csv = os.path.join(LOG_DIR, "training_metrics_final.csv")
    metrics_callback.save_csv(final_metrics_csv)
    print(f"Final training metrics saved to: {final_metrics_csv}")

    env.close()
    gc.collect()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")
