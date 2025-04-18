#!/bin/bash
echo "🚀 Setting up ViZDoom in Google Colab..."
# Install system-level dependencies
apt-get update

# Install Python packages
pip install vizdoom
pip install gym
pip install stable-baselines3
pip install stable-baselines3[extra]
pip install opencv-python matplotlib
echo "✅ ViZDoom setup for Colab completed!"