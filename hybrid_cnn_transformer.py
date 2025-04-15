
import torch
import torch.nn as nn
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import vizdoomgym

class CNNTransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
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
    env = make_vec_env("VizdoomBasic-v0", n_envs=1)

    policy_kwargs = dict(
        features_extractor_class=CNNTransformerFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256)
    )

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, seed=123)
    model.learn(total_timesteps=100000)
    model.save("models/Hybrid_CNNTransformer")
    print("✅ Hybrid CNN-Transformer model trained and saved.")
