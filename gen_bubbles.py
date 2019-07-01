import numpy as np
from Random_bubbles import RandomBubbles
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

n = 200
bubbles = RandomBubbles(DIM = n, radius = 10., nb = 50, nooverlap = True)
bubbles.write_ionisation_field()
data = np.loadtxt('bub.txt', delimiter = ',')
field = fftshift(fft2(fftshift(data)))
name = 'bubbles.txt'
np.savetxt(name, field, delimiter=',')
