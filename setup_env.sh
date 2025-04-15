#!/bin/bash
echo "Installing ViZDoom DRL dependencies..."
pip install --upgrade pip
pip install vizdoom gym
pip install stable-baselines3[extra]
pip install opencv-python matplotlib
pip install git+https://github.com/shakenes/vizdoomgym.git
echo "✅ Setup complete!"
