#!/bin/bash
echo "🚀 Setting up ViZDoom in Google Colab..."

# Install system-level dependencies
apt-get update
apt-get install -y cmake libboost-all-dev libopencv-dev zlib1g-dev \
    libsdl2-dev libglu1-mesa-dev libpng-dev

# Clone and build ViZDoom
git clone https://github.com/mwydmuch/ViZDoom.git
cd ViZDoom
mkdir build && cd build
cmake ..
make -j4
cd ..
pip install .
cd ..

# Install Python packages
pip install stable-baselines3[extra]
pip install opencv-python matplotlib
pip install git+https://github.com/shakenes/vizdoomgym.git

echo "✅ ViZDoom setup for Colab completed!"