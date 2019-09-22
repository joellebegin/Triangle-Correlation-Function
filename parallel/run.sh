#!/bin/bash

echo 'bubbles.txt' | mpiexec -n 8 python triangleCorrelation.py
