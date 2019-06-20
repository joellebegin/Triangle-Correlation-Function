import numpy as np
from scipy.special import j0
from tqdm import tqdm
import matplotlib.pyplot as plt

def sr(r_i, spec, n_k, n_q, p):
    '''given a value of r, determines which bispectra satisfy the pi/r cutoff.
    Computes window function for each value, sums over w*B, and multiplies by 
    prefactor.'''
    
    #indices in norms where k&&q <= pi/r
    ind_kq = np.argwhere( (n_k > np.pi/r_i) & (n_q > np.pi/r_i) )

    spec[ind_kq] =0
    p[ind_kq] =0
    
    window = j0(r_i*p)
    window[ind_kq] =0

    sum_r = np.sum(spec*window)
    return ((r_i/L)**3)*sum_r

def compute_tcf(r, bispectra, n_k, n_q, p):
    '''iterates through the correlation scales'''
    t = []
    for scale in tqdm(r, desc='Computing s(r)'): 
        t.append(sr(scale, bispectra, n_k, n_q, p))
    return np.array(t)

def main():
    data = np.loadtxt('bispectra_et_al.txt', delimiter=',')
    bispectra = data[0]
    k_norms = np.real(data[1])
    q_norms = np.real(data[2])
    p = np.real(data[3])

    bins = 200
    if input('\tSet number of r bins? default is 200. [y/n]: ') == "y":
        bins = int(input('\t\tnbins: '))

    r = np.linspace(0.5, 30, bins)
    triangle_corr = compute_tcf(r, bispectra, k_norms, q_norms, p)

    if input('\n\tDisplay results? [y/n]:  ') == 'y':
        fig, ax = plt.subplots(figsize = (20,10))
        ax.plot(triangle_corr[0], np.real(triangle_corr[1]))
        
        plt.rcParams.update({'font.size' :15, 'axes.labelsize': 30})

        ax.grid()
        ax.grid(color = '0.7')
        ax.set_facecolor('0.8')

        #ax.set_ylim(-0.01, 0.5)
        ax.set_xlabel('r (Mpc)')
        ax.set_ylabel('s(r)')

        plt.show()
        