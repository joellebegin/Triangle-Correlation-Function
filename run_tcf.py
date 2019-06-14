import numpy as np 
from triangleCorrelation import tcf 
from time import time
import matplotlib.pyplot as plt

n = int(input('     number of pixels: '))
length = None
bins = None

if input('     set realspace length of box? [y/n] ') == "y":
    length = int(input('                length: '))

if input('     set number of r bins? [y/n] ') == "y":
    bins = int(input('              nbins: '))

field = np.random.normal(size = (n,n)) + 1j*np.random.normal(size = (n,n))
print('Beginning computation')

if length is not None and bins is not None:
    start = time()
    stuff = tcf(field, length, bins)
    end = time()
elif length is None and bins is not None:
    start = time()
    stuff = tcf(field, rbins = bins)
    end = time()
else:
    start = time()
    stuff = tcf(field)
    end = time()

print('         Total runtime was ', end -start, ' seconds.')

if input('         Display results? [y/n]  ') == 'y':
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