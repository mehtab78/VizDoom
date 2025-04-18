import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os

# Import VizDoom
import vizdoom as vzd
from vizdoom import DoomGame, ScreenResolution


# Define the custom VizDoom environment like we did in doom_agents.py
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
        self.action_space = gym.spaces.Discrete(3)  # left, right, shoot

        # Important: VizDoom returns (channels, height, width)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 120, 160), dtype=np.uint8
        )

    def reset(self, **kwargs):
        self.game.new_episode()
        state = self.game.get_state()

        if state:
            # Use channels-first format directly (3, 120, 160)
            obs = state.screen_buffer
        else:
            obs = np.zeros((3, 120, 160), dtype=np.uint8)

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
            obs = state.screen_buffer  # Already in channels-first format
        else:
            obs = np.zeros((3, 120, 160), dtype=np.uint8)

        return obs, reward, done, False, {}  # obs, reward, terminated, truncated, info

    def close(self):
        self.game.close()


def make_env():
    return VizdoomBasicEnv()


class CNNTransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            cnn_out = self.cnn(sample)
            c, h, w = cnn_out.shape[1:]
        self.embedding_dim = c
        self.seq_len = h * w

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final = nn.Linear(self.embedding_dim * self.seq_len, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).flatten(1)
        return self.final(x)


if __name__ == "__main__":
    # Show VizDoom info
    print(f"VizDoom version: {vzd.__version__}")
    print(f"Scenarios path: {vzd.scenarios_path}")

    os.makedirs("models", exist_ok=True)

    # Create the custom VizDoom environment
    env = make_vec_env(make_env, n_envs=1)

    policy_kwargs = dict(
        features_extractor_class=CNNTransformerFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    print("Creating and training the Hybrid CNN-Transformer model...")
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, seed=123)

    try:
        model.learn(total_timesteps=100000)
        model.save("models/Hybrid_CNNTransformer")
        print("✅ Hybrid CNN-Transformer model trained and saved.")

        # Test the model
        print("Testing the trained model...")
        test_env = VizdoomBasicEnv()
        obs, _ = test_env.reset()

        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = test_env.step(action)
            total_reward += reward

        print(f"Test episode completed with reward: {total_reward}")
        test_env.close()

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
