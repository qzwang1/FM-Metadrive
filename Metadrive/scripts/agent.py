import os
os.environ["METADRIVE_HEADLESS"] = "1"

from scenic.gym import ScenicGymEnv
import scenic
from custom.custom_simulator import CustomMetaDriveSimulation, CustomMetaDriveSimulator
from custom.custom_gym import CustomMetaDriveEnv

from scenic.simulators.metadrive import MetaDriveSimulator

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from metadrive.component.sensors.semantic_camera import SemanticCamera

import torch
import time

# ============================================================
# Config
# ============================================================

# 如果你想用 PPO 训练，把这个改成 True
USE_PPO = False

max_steps = 100
episodes_per_sampler = 10
total_timesteps = max_steps * episodes_per_sampler * 10  # 训练步数（如果 USE_PPO=True）

action_space = gym.spaces.Box(
    low=np.array([-1, -1]),
    high=np.array([1, 1]),
    shape=(2,),
    dtype=np.float32
)

observation_space = gym.spaces.Dict(
    {
        "velocity": gym.spaces.Discrete(16),
        "sensor": gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1]),
            shape=(7,),
            dtype=np.float64
        ),
        "position": gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float64
        ),
        "rotation": gym.spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            shape=(4,),
            dtype=np.float64
        ),
    }
)

SCENIC_FILE = "./scenarios/driver.scenic"
SUMO_MAP_FILE = "./CARLA/Town01.net.xml"

# 你可以根据 driver.scenic 里支持的 sampler 名字来改这一行
SAMPLERS = [
    "random",
    "halton",
    "bo"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"displaying device: {device}")


# ============================================================
# Env factory: 根据 sampler 创建一个新的 env
# ============================================================

def make_env_with_sampler(sampler_name: str) -> CustomMetaDriveEnv:
    """
    根据 sampler_name 创建 Scenic 场景和 CustomMetaDriveEnv
    sampler_name 会通过 Scenic 的 params 传给 verifaiSamplerType
    """
    params = {"verifaiSamplerType": sampler_name}

    scenario1 = scenic.scenarioFromFile(
        SCENIC_FILE,
        model="scenic.simulators.metadrive.model",
        mode2D=True,
        params=params,
    )
    scenario2 = scenic.scenarioFromFile(
        SCENIC_FILE,
        model="scenic.simulators.metadrive.model",
        mode2D=True,
        params=params,
    )

    env = CustomMetaDriveEnv(
        scenario=[scenario1, scenario2],
        simulator=CustomMetaDriveSimulator(
            sumo_map=SUMO_MAP_FILE,
            max_steps=max_steps,
        ),
        observation_space=observation_space,
        action_space=action_space,
        file=SCENIC_FILE,
    )
    return env


# ============================================================
# 训练代理（可选，只有当 USE_PPO=True 时才用）
# ============================================================

def train_agent_for_sampler(sampler_name: str):
    print(f"\n[TRAIN] sampler = {sampler_name}")
    env = make_env_with_sampler(sampler_name)
    # 如果 observation 是 Dict，用 MultiInputPolicy
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=None,
        device=device
    )
    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model


# ============================================================
# 评估一个 sampler（可以用随机策略或训练好的 PPO）
# ============================================================

def evaluate_sampler(sampler_name: str, episodes: int = 20, model: PPO = None):
    print("\n" + "#" * 10 + f" Evaluating sampler: {sampler_name} " + "#" * 10 + "\n")

    env = make_env_with_sampler(sampler_name)

    stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "crash_count": 0,
        "offroad_count": 0,
    }

    for ep in range(episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        ep_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            if model is not None:
                # PPO 策略
                action, _ = model.predict(obs, deterministic=True)
            else:
                # 随机策略
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1

            crash_flag = bool(info.get("crash", False))
            road_dev = float(info.get("road_deviation", 0.0))

            if crash_flag:
                stats["crash_count"] += 1
                # 这里可以选择 break，表示发生 crash 就结束 episode
                break

            # 你可以根据需要设置一个“出路偏移过大”的阈值
            if road_dev > 0.5:
                stats["offroad_count"] += 1
                break

        stats["episode_rewards"].append(ep_reward)
        stats["episode_lengths"].append(steps)

        print(
            f"[{sampler_name}] episode {ep + 1}/{episodes} "
            f"reward={ep_reward:.3f}, steps={steps}, "
            f"crash={crash_flag}, road_dev={road_dev:.3f}"
        )

    env.close()

    # 统计
    rewards = np.array(stats["episode_rewards"], dtype=np.float32)
    lengths = np.array(stats["episode_lengths"], dtype=np.float32)

    mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else float("nan")
    std_reward = float(np.std(rewards)) if len(rewards) > 0 else float("nan")
    mean_len = float(np.mean(lengths)) if len(lengths) > 0 else float("nan")

    crash_rate = stats["crash_count"] / episodes
    offroad_rate = stats["offroad_count"] / episodes

    print("\n" + "=" * 80)
    print(f"Sampler: {sampler_name}")
    print(f"  Episodes:        {episodes}")
    print(f"  Mean reward:     {mean_reward:.3f} (std: {std_reward:.3f})")
    print(f"  Mean ep length:  {mean_len:.1f}")
    print(f"  Crash episodes:  {stats['crash_count']}/{episodes} "
          f"({crash_rate * 100:.1f}%)")
    print(f"  Offroad episodes:{stats['offroad_count']}/{episodes} "
          f"({offroad_rate * 100:.1f}%)")
    print("=" * 80 + "\n")

    summary = {
        "sampler": sampler_name,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_len,
        "crash_rate": crash_rate,
        "offroad_rate": offroad_rate,
    }
    return summary


# ============================================================
# 主程序：对所有 sampler 做比较
# ============================================================

if __name__ == "__main__":
    start_time = time.time()

    all_results = []

    for sampler in SAMPLERS:
        if USE_PPO:
            mdl = train_agent_for_sampler(sampler)
        else:
            mdl = None
        result = evaluate_sampler(sampler, episodes=episodes_per_sampler, model=mdl)
        all_results.append(result)

    print("\n" + "#" * 30 + " Summary over samplers " + "#" * 30)
    for res in all_results:
        print(
            f"- {res['sampler']}: "
            f"mean_reward={res['mean_reward']:.3f}, "
            f"crash_rate={res['crash_rate'] * 100:.1f}%, "
            f"offroad_rate={res['offroad_rate'] * 100:.1f}%"
        )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")
import os
os.environ["METADRIVE_HEADLESS"] = "1"

from scenic.gym import ScenicGymEnv
import scenic
from custom.custom_simulator import CustomMetaDriveSimulation, CustomMetaDriveSimulator
from custom.custom_gym import CustomMetaDriveEnv

from scenic.simulators.metadrive import MetaDriveSimulator

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from metadrive.component.sensors.semantic_camera import SemanticCamera

import torch
import time

# ============================================================
# Config
# ============================================================

# 如果你想用 PPO 训练，把这个改成 True
USE_PPO = False

max_steps = 100
episodes_per_sampler = 10
total_timesteps = max_steps * episodes_per_sampler * 10  # 训练步数（如果 USE_PPO=True）

action_space = gym.spaces.Box(
    low=np.array([-1, -1]),
    high=np.array([1, 1]),
    shape=(2,),
    dtype=np.float32
)

observation_space = gym.spaces.Dict(
    {
        "velocity": gym.spaces.Discrete(16),
        "sensor": gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1]),
            shape=(7,),
            dtype=np.float64
        ),
        "position": gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float64
        ),
        "rotation": gym.spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            shape=(4,),
            dtype=np.float64
        ),
    }
)

SCENIC_FILE = "./scenarios/driver.scenic"
SUMO_MAP_FILE = "./CARLA/Town01.net.xml"

# 你可以根据 driver.scenic 里支持的 sampler 名字来改这一行
SAMPLERS = [
    "random",
    "halton",
    "bo"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"displaying device: {device}")


# ============================================================
# Env factory: 根据 sampler 创建一个新的 env
# ============================================================

def make_env_with_sampler(sampler_name: str) -> CustomMetaDriveEnv:
    """
    根据 sampler_name 创建 Scenic 场景和 CustomMetaDriveEnv
    sampler_name 会通过 Scenic 的 params 传给 verifaiSamplerType
    """
    params = {"verifaiSamplerType": sampler_name}

    scenario1 = scenic.scenarioFromFile(
        SCENIC_FILE,
        model="scenic.simulators.metadrive.model",
        mode2D=True,
        params=params,
    )
    scenario2 = scenic.scenarioFromFile(
        SCENIC_FILE,
        model="scenic.simulators.metadrive.model",
        mode2D=True,
        params=params,
    )

    env = CustomMetaDriveEnv(
        scenario=[scenario1, scenario2],
        simulator=CustomMetaDriveSimulator(
            sumo_map=SUMO_MAP_FILE,
            max_steps=max_steps,
        ),
        observation_space=observation_space,
        action_space=action_space,
        file=SCENIC_FILE,
    )
    return env


# ============================================================
# 训练代理（可选，只有当 USE_PPO=True 时才用）
# ============================================================

def train_agent_for_sampler(sampler_name: str):
    print(f"\n[TRAIN] sampler = {sampler_name}")
    env = make_env_with_sampler(sampler_name)
    # 如果 observation 是 Dict，用 MultiInputPolicy
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=None,
        device=device
    )
    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model


# ============================================================
# 评估一个 sampler（可以用随机策略或训练好的 PPO）
# ============================================================

def evaluate_sampler(sampler_name: str, episodes: int = 20, model: PPO = None):
    print("\n" + "#" * 10 + f" Evaluating sampler: {sampler_name} " + "#" * 10 + "\n")

    env = make_env_with_sampler(sampler_name)

    stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "crash_count": 0,
        "offroad_count": 0,
    }

    for ep in range(episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        ep_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            if model is not None:
                # PPO 策略
                action, _ = model.predict(obs, deterministic=True)
            else:
                # 随机策略
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1

            crash_flag = bool(info.get("crash", False))
            road_dev = float(info.get("road_deviation", 0.0))

            if crash_flag:
                stats["crash_count"] += 1
                # 这里可以选择 break，表示发生 crash 就结束 episode
                break

            # 你可以根据需要设置一个“出路偏移过大”的阈值
            if road_dev > 0.5:
                stats["offroad_count"] += 1
                break

        stats["episode_rewards"].append(ep_reward)
        stats["episode_lengths"].append(steps)

        print(
            f"[{sampler_name}] episode {ep + 1}/{episodes} "
            f"reward={ep_reward:.3f}, steps={steps}, "
            f"crash={crash_flag}, road_dev={road_dev:.3f}"
        )

    env.close()

    # 统计
    rewards = np.array(stats["episode_rewards"], dtype=np.float32)
    lengths = np.array(stats["episode_lengths"], dtype=np.float32)

    mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else float("nan")
    std_reward = float(np.std(rewards)) if len(rewards) > 0 else float("nan")
    mean_len = float(np.mean(lengths)) if len(lengths) > 0 else float("nan")

    crash_rate = stats["crash_count"] / episodes
    offroad_rate = stats["offroad_count"] / episodes

    print("\n" + "=" * 80)
    print(f"Sampler: {sampler_name}")
    print(f"  Episodes:        {episodes}")
    print(f"  Mean reward:     {mean_reward:.3f} (std: {std_reward:.3f})")
    print(f"  Mean ep length:  {mean_len:.1f}")
    print(f"  Crash episodes:  {stats['crash_count']}/{episodes} "
          f"({crash_rate * 100:.1f}%)")
    print(f"  Offroad episodes:{stats['offroad_count']}/{episodes} "
          f"({offroad_rate * 100:.1f}%)")
    print("=" * 80 + "\n")

    summary = {
        "sampler": sampler_name,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_len,
        "crash_rate": crash_rate,
        "offroad_rate": offroad_rate,
    }
    return summary


# ============================================================
# 主程序：对所有 sampler 做比较
# ============================================================

if __name__ == "__main__":
    start_time = time.time()

    all_results = []

    for sampler in SAMPLERS:
        if USE_PPO:
            mdl = train_agent_for_sampler(sampler)
        else:
            mdl = None
        result = evaluate_sampler(sampler, episodes=episodes_per_sampler, model=mdl)
        all_results.append(result)

    print("\n" + "#" * 30 + " Summary over samplers " + "#" * 30)
    for res in all_results:
        print(
            f"- {res['sampler']}: "
            f"mean_reward={res['mean_reward']:.3f}, "
            f"crash_rate={res['crash_rate'] * 100:.1f}%, "
            f"offroad_rate={res['offroad_rate'] * 100:.1f}%"
        )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")

