import numpy as np
from Random_bubbles import RandomBubbles
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

n = 64
for i in range(20):
    bubbles = RandomBubbles(DIM = n, radius = 1., nb = 60, nooverlap = True)
    bubbles.write_ionisation_field()
    data = np.loadtxt('bub.txt', delimiter = ',')
    field = fftshift(fft2(fftshift(data)))
    name = 'bubbles' + str(i) + '.txt'
    np.savetxt(name, field, delimiter=',')