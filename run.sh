#!/bin/bash
echo "🚀 Installing vizdoomgym (with delay to load properly)..."
pip install git+https://github.com/shakenes/vizdoomgym.git

# Wait a moment to ensure Python runtime refreshes
sleep 3

echo "🏃 Running DRL agents and hybrid training scripts..."
python3 vizdoom_drl_agents.py
python3 hybrid_cnn_transformer.py
python3 plot_learning_curves.py

echo "✅ All scripts executed successfully!"