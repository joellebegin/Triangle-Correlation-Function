#!/bin/bash
python gen_rand_gauss.py
mkdir fields
mkdir results_figures
mkdir results_data

for f in *.txt; do
    mv ${f} fields
    done

for f in fields/*.txt; do
    echo "### doing $f ###"
    echo ${f} | mpiexec -n 3 python triangleCorrelation.py
    done

for f in fields/*.csv; do
    mv ${f} results_data
    done

for f in fields/*.png; do
    mv ${f} results_figures
    done
