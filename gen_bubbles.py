import numpy as np
from Random_bubbles import RandomBubbles
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

n = 100
for i in range(10):

	bubbles = RandomBubbles(DIM = n, radius = 3., nb = 20, nooverlap = False)
	bubbles.write_ionisation_field()
	data = np.loadtxt('bub.txt', delimiter = ',')

	field = fftshift(fft2(fftshift(data)))
	name = 'bubbles' + str(i) +'.txt'
	np.savetxt(name, field, delimiter=',')
