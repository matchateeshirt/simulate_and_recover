#!/bin/bash

# Run the Python simulation script
echo "Running EZ diffusion simulation..."
python3 src/simulate.py > results.txt

echo "Running parameter recovery..."
python3 src/recover.py results.txt > recovered_results.txt

echo "Simulation and recovery complete. Check results.txt and recovered_results.txt."
