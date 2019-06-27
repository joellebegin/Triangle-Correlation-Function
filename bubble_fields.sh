#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --mem=35000
#SBATCH --cpus-per-task=10

python gen_bubbles.py
rm bub.txt

echo 'bubbles.txt' | mpiexec -n 4 python triangleCorrelation.py
rm bubbles.txt

mv bubbles.txt.csv results_data

mv bubbles.txt.png results_figures

