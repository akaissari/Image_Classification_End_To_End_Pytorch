#!/bin/bash
# This script sets up a Python virtual environment and installs all required packages

echo "Creating virtual environment..."
python -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete!"
