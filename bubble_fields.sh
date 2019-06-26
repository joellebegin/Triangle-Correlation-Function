#!/bin/bash
python gen_bubbles.py
rm bub.txt
mkdir fields
mkdir results_figures
mkdir results_data

for f in *.txt; do
    mv ${f} fields
    done

for f in fields/*.txt; do
    echo "### doing $f ###"
    echo ${f} | mpiexec -n 5 python triangleCorrelation.py
    done

for f in fields/*.csv; do
    mv ${f} results_data
    done

for f in fields/*.png; do
    mv ${f} results_figures
    done