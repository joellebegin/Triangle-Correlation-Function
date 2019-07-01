#!/bin/bash
python gen_rand_gauss.py

echo 'rand_gauss.txt' | mpiexec -n 3 python triangleCorrelation.py

