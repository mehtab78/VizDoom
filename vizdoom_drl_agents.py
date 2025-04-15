
import os
import gym
import numpy as np
import torch
import random
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from vizdoomgym.envs.vizdoom_basic import VizdoomBasicEnv

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_agent(algo, seed, total_timesteps=100000):
    set_seed(seed)
    env = DummyVecEnv([lambda: gym.make('VizdoomBasic-v0')])
    
    if algo == "PPO":
        model = PPO("CnnPolicy", env, verbose=0, seed=seed)
    elif algo == "A2C":
        model = A2C("CnnPolicy", env, verbose=0, seed=seed)
    else:
        raise ValueError("Unsupported algorithm")

    print(f"Training {algo} with seed {seed}...")
    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/{algo}_seed_{seed}")
    return model

def evaluate_agent(model, episodes=10):
    env = gym.make('VizdoomBasic-v0')
    total_reward = 0
    total_steps = 0
    rewards_per_ep = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1
        rewards_per_ep.append(ep_reward)
        total_reward += ep_reward
        total_steps += steps

    np.save(f"logs/{model.policy.__class__.__name__}_eval.npy", rewards_per_ep)
    return total_reward / episodes, total_steps / episodes

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    agents = ["PPO", "A2C"]
    seeds = [123, 456, 789]
    results = {}

    for agent in agents:
        all_rewards = []
        all_steps = []
        for seed in seeds:
            model = train_agent(agent, seed)
            avg_r, avg_s = evaluate_agent(model)
            all_rewards.append(avg_r)
            all_steps.append(avg_s)
        results[agent] = {"Avg Reward": np.mean(all_rewards), "Avg Steps": np.mean(all_steps)}

    print("\n=== Evaluation Summary ===")
    for agent, vals in results.items():
        print(f"{agent}: Reward={vals['Avg Reward']:.2f}, Steps={vals['Avg Steps']:.2f}")
