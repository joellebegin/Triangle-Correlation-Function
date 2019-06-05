import numpy as np
from scipy.special import j0

def k_vects(n):
    '''Constructs array of all possible k_vectors for an n by n grid, excluding 0 vector. 
    Returns array of the form: k = [[k1x, k1y], [k2x, k2y], ...]'''
    delta_k = 2*np.pi/L #scaling factor for k vectors
    
    #kx, ky contain horizontal and vertical coordinates of all vectors in a box with origin at n//2
    #and of dimensions n by n
    kx,ky = np.indices((n,n)) -n//2 
    k = np.array(list(zip(kx.flatten(), ky.flatten())))
    norms = np.array([np.linalg.norm(inds) for inds in k])
    ind = np.argsort(norms)
    norms_sorted = norms[ind]*delta_k
    k_sorted = k[ind]
    return k_sorted[1:], norms_sorted[1:] 

def k_condition(r):
    '''returns the k vectors that satisfy norm(k) <= pi/r'''
    norm_indices = []
    for i in range(len(k_norms)): 
        if k_norms[i] <= np.pi/r:
            norm_indices.append(i)
    return k_vals[norm_indices]

def in_range(vector):
    norm_vect = np.linalg.norm(vector)
    if norm_vect >= 0.5*n - 0.5:
        inrange = False
    else:
        inrange = True
    return inrange

def bispec(k, q, sum_kq,r):
    sq3 = np.sqrt(3)
    sum_kq_ind = np.round(sum_kq) #what pixel k+q falls into
    #GO BACK TO COORDINATE SYSTEM OF FIELD IE VECTS + N//2
    px = k[0] + 0.5*q[0] + 0.5*sq3*q[1]
    py = k[0] - 0.5*sq3*q[0] + 0.5*q[1]
    norm_p = np.linalg.norm((px,py))
    
    b = epsilon_k[k[0], k[1]]*epsilon_k[q[0], q[1]]*np.conj(epsilon_k[sum_kq_ind[0], sum_kq_ind[1]])
    window = j0(norm_p*r)
    
    return b*window
    
def sum_q(k_array, i, r):
    '''for a given k vector, this sums over all the possible q. Essentially "inner" sum in the 2D sum'''
    summond = 0
    k_i = k_array[i]
    for q in k_array[i+1:]: #we slice k_array in this way to avoid duplicates
        sum_kq = k_i + q
        if in_range(sum_kq): # Makes sure k+q is not outside box
            summond += bispec(k_i, q, sum_kq,r)
        else:
            summond += 0
    return summond
            
def s(r):
    k_available = k_condition(r) #k that satisfy k<= pi/r
    sum_r = 0
    for i in range(len(k_available)): #looping over all possible k
        sum_r += sum_q(k_available, i, r)
    sum_r *= (r/L)**(3)
    return sum_r

def tcf(field, length):
    '''given a field in fourier space, and the real space length of its side'''
    global epsilon_k, k_vals, k_norms, n, L
    epsilon_k = field/np.abs(field) #phase factor of field
    L = length
    n = field.shape[0]
    k_vals, k_norms = k_vects(n) #all possible k vectors and their norm, sorted
    
    #list of correlation scales we want to probe
    r = np.linspace(0.5, 50,20)
    triangle_corr = []
    
    for scale in r: 
        triangle_corr.append(s(scale))
        
    return r,triangle_corr