#!/bin/bash
echo "Installing ViZDoom DRL dependencies..."
# Install game environment dependencies
apt-get update
apt-get install -y cmake libboost-all-dev libopencv-dev zlib1g-dev libsdl2-dev libglu1-mesa-dev libpng-dev

# Clone and build ViZDoom manually
git clone https://github.com/mwydmuch/ViZDoom.git
cd ViZDoom
mkdir build && cd build
cmake ..
make -j4
cd ..
pip install .

# Now install rest
pip install stable-baselines3[extra] opencv-python matplotlib
pip install git+https://github.com/shakenes/vizdoomgym.git
echo "✅ Setup complete!"
