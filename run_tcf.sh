#!/bin/bash

for f in bubbles/*.txt;
do
	echo "#### DOING ${f} ####"
	echo ${f} |  mpiexec -n 2 python triangleCorrelation.py
done

for f in bubbles/*.csv;
do 
	mv ${f} results2
done
