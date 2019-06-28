#!/bin/bash

python gen_rand_gauss.py


echo 'rand_gauss.txt' | mpiexec -n 2 python triangleCorrelation.py
rm bubbles.txt

mv bubbles.txt.csv results_data

mv bubbles.txt.png results_figures

