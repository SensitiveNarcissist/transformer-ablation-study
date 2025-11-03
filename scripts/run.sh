#!/bin/bash

# Transformer Ablation Study - Run Script
# Usage: ./scripts/run.sh

set -e

echo "Setting up environment..."
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating directories..."
mkdir -p results
mkdir -p data

echo "Starting ablation study..."
python main.py

echo "Experiment completed! Results saved to ./results/"