import os
import numpy as np
import torch
import random
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces

import vizdoom as vzd
from vizdoom import DoomGame, ScreenResolution


class VizdoomBasicEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = DoomGame()

        # Use vizdoom's built-in scenarios path
        basic_cfg = os.path.join(vzd.scenarios_path, "basic.cfg")

        if not os.path.exists(basic_cfg):
            raise FileNotFoundError(f"Cannot find basic.cfg at {basic_cfg}")

        print(f"Loading config from: {basic_cfg}")
        self.game.load_config(basic_cfg)

        # Set screen resolution and disable window
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_window_visible(False)

        # Initialize the game
        self.game.init()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # left, right, shoot

        # Important: VizDoom returns (channels, height, width) but we need (height, width, channels)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(120, 160, 3), dtype=np.uint8
        )

    def reset(self, **kwargs):
        self.game.new_episode()
        state = self.game.get_state()

        if state:
            # Transform from (channels, height, width) to (height, width, channels)
            screen = state.screen_buffer
            # Transpose from (3, 120, 160) to (120, 160, 3)
            screen = screen.transpose(1, 2, 0)
            obs = screen
        else:
            obs = np.zeros((120, 160, 3), dtype=np.uint8)

        return obs, {}  # Return observation and empty info dict

    def step(self, action):
        # Define actions: left, right, shoot
        actions = [
            [0, 0, 1],  # SHOOT
            [1, 0, 0],  # LEFT
            [0, 1, 0],  # RIGHT
        ]

        reward = self.game.make_action(actions[action], 4)
        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            # Transpose from (3, 120, 160) to (120, 160, 3)
            screen = state.screen_buffer
            obs = screen.transpose(1, 2, 0)
        else:
            obs = np.zeros((120, 160, 3), dtype=np.uint8)

        return obs, reward, done, False, {}  # obs, reward, terminated, truncated, info

    def close(self):
        self.game.close()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_env():
    return VizdoomBasicEnv()


def train_agent(algo, seed, total_timesteps=100000):
    set_seed(seed)

    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=1, seed=seed)

    if algo == "PPO":
        model = PPO("CnnPolicy", env, verbose=1, seed=seed)
    elif algo == "A2C":
        model = A2C("CnnPolicy", env, verbose=1, seed=seed)
    else:
        raise ValueError("Unsupported algorithm")

    print(f"Training {algo} with seed {seed}...")
    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/{algo}_seed_{seed}")
    return model


def evaluate_agent(model, episodes=10):
    env = VizdoomBasicEnv()
    total_reward = 0
    total_steps = 0
    rewards_per_ep = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            steps += 1
            if done:
                break
        rewards_per_ep.append(ep_reward)
        total_reward += ep_reward
        total_steps += steps

    env.close()
    np.save(f"logs/{model.policy.__class__.__name__}_eval.npy", rewards_per_ep)
    return total_reward / episodes, total_steps / episodes


if __name__ == "__main__":
    # Show VizDoom info
    print(f"VizDoom version: {vzd.__version__}")
    print(f"Scenarios path: {vzd.scenarios_path}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # First, run with a single agent and seed for testing
    print("Running a test training session to verify everything works...")
    try:
        test_model = train_agent(
            "PPO", 123, total_timesteps=1000
        )  # Short training for test
        test_reward, test_steps = evaluate_agent(test_model, episodes=2)
        print(
            f"Test run complete! Avg reward: {test_reward:.2f}, Avg steps: {test_steps:.2f}"
        )

        # Now run the full experiment
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
            results[agent] = {
                "Avg Reward": np.mean(all_rewards),
                "Avg Steps": np.mean(all_steps),
            }

        print("\n=== Evaluation Summary ===")
        for agent, vals in results.items():
            print(
                f"{agent}: Reward={vals['Avg Reward']:.2f}, Steps={vals['Avg Steps']:.2f}"
            )

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()
