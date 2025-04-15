#!/bin/bash
echo "Running all ViZDoom training scripts..."
python3 vizdoom_drl_agents.py
python3 hybrid_cnn_transformer.py
python3 plot_learning_curves.py
echo "✅ All done!"
