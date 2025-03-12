#!/bin/bash

# Run the Python simulation script
echo "Running EZ diffusion simulation..."
python3 src/simulate.py > results.txt

echo "Simulation complete. Results saved in results.txt."
