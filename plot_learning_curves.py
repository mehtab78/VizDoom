
import os
import matplotlib.pyplot as plt
import numpy as np

reward_logs = {
    "PPO": "logs/CnnPolicy_eval.npy",
    "A2C": "logs/CnnPolicy_eval.npy",
    "Hybrid": "logs/Hybrid_CNNTransformer_eval.npy"
}

plt.figure(figsize=(10, 6))
for agent_name, file_path in reward_logs.items():
    if os.path.exists(file_path):
        rewards = np.load(file_path)
        smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed, label=agent_name)

plt.title("Learning Curves - Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curves.png")
plt.show()
