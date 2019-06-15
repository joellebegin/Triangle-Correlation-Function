import numpy as np 
from triangleCorrelation import tcf 
from time import time
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift


length = None
bins = None
cut = None

if input('     set realspace length of box? [y/n]: ') == "y":
    length = int(input('         length: '))

if input('     set number of r bins? [y/n]: ') == "y":
    bins = int(input('         nbins: '))

if input('     introduce foreground wedge? [y/n]: ') == 'y':
    cut = True

field_type = input('     field from file or random gaussian? [file/gaussian]: ')
if  field_type == 'gaussian':
    n = int(input('         number of pixels: '))
    field = np.random.normal(size = (n,n)) + 1j*np.random.normal(size = (n,n))
elif field_type == 'file':
    loc = input('         name of field file: ')
    f = np.loadtxt(loc, delimiter=',')
    field = fftshift(fft2(fftshift(f)))
print('Beginning computation')

if length is not None and bins is not None:
    start = time()
    stuff = tcf(field, length, bins, cutoff = cut)
    end = time()
elif length is None and bins is not None:
    start = time()
    stuff = tcf(field, rbins = bins, cutoff = cut)
    end = time()
else:
    start = time()
    stuff = tcf(field, cutoff= cut)
    end = time()

print('         Total runtime was ', end -start, ' seconds.')

if input('         Display results? [y/n]:  ') == 'y':
    fig, ax = plt.subplots(figsize = (20,10))
    ax.plot(stuff[0], np.real(stuff[1]))
    
    plt.rcParams.update({'font.size' :15, 'axes.labelsize': 30})

    ax.grid()
    ax.grid(color = '0.7')
    ax.set_facecolor('0.8')

    ax.set_ylim(-0.01, 0.05)
    ax.set_xlabel('r (Mpc)')
    ax.set_ylabel('s(r)')

    plt.show()