import numpy as np 
from numpy.fft import fft2, fftshift

n = 100

field = np.random.normal(size = (n,n))
fourier = fftshift(fft2(fftshift(field)))
fieldname = 'rand_gauss.txt'
np.savetxt(fieldname, fourier, delimiter=',')
