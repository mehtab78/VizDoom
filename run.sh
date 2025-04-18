#!/bin/bash
echo "🏃 Running DRL agents and hybrid training scripts..."
python3 doom_agents.py
python3 hybrid_cnn_transformer.py
python3 plot_learning_curves.py

echo "✅ All scripts executed successfully!"