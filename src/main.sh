#!/bin/bash

for N in 10 40 4000; do
   echo "Running EZ diffusion simulation for N=$N..."
   python3 src/simulate.py $N 1000 > results_N${N}.txt
   echo "Finished running simulation for N=$N. Checking file size:"
   ls -lh results_N${N}.txt  
done

for N in 10 40 4000; do
   echo "Running parameter recovery for N=$N..."
   python3 src/recover.py results_N${N}.txt > recovered_results_N${N}.txt
   echo "Finished recovery for N=$N. Checking output:"
   ls -lh recovered_results_N${N}.txt 
done

echo "Simulation and recovery complete. Check results_N*.txt and recovered_results_N*.txt."
