import numpy as np 
from scipy.special import j0
from tqdm import tqdm
import sys


def remove_k(vectors):
    '''If we are considering the foreground wedge, this function returns the 
    indices of the k vectors which we can still consider'''

    theta = int(input('\t\tcutoff line inclination (in degrees): '))
    theta_rad = theta*(np.pi/180)

    c_ind = []

    vx = vectors[:,0]
    vy =vectors[:,1]

    line = vx*np.tan(theta_rad) #modes below this line are discarded

    for i, point in enumerate(line):
        if vy[i] > point:
            c_ind.append(i)

    return c_ind
    
def k_vects():
    '''Constructs array of all vectors for an n by n grid, excluding 
    0 vector. 
    
    Returns array of the form: k = [[k1x, k1y], [k2x, k2y], ...]'''
    
    #going to system where orign is pixel at (n//2,n//2)
    x,y = np.indices((n,n)) - n//2 
    delta_k = 2*np.pi/L #scaling factor to go from real space to fourier space
    
    
    k = np.vstack((x.flatten(),y.flatten())).transpose()
    norms_k= np.linalg.norm(k, axis = 1)*delta_k

    ind = np.argsort(norms_k)

    if k_cutoff:
        cutoff_indices = remove_k(k)
        vectors = k[cutoff_indices]
        norms = norms_k[cutoff_indices]
        return vectors, norms
    else:
        #slice at one in order to not include zero vector
        return k[ind][1:], norms_k[ind][1:]


def bispectrum(k,q,s):
    '''evaluates bispectrum and constructs p vector
    
    Returns b = {B(k,q)}, for the given k and q, and p, the corresponding norm 
    of p for the bispectra'''

    sq3 = np.sqrt(3)
    delta_k = 2*np.pi/L
    #separating components, going back to coordinate system where field is stored
    kx,ky = k[0],k[1] 
    qx,qy = q[:,0], q[:,1]
    sx,sy = s[:,0], s[:,1]

    #evaluating bispectrum
    #have to give y index first since that's how the field data structure works
    b = epsilon_k[ky+ n//2,kx+ n//2]*epsilon_k[qy+ n//2,qx+ n//2]*np.conj(epsilon_k[sy+ n//2,sx+ n//2])
    
    #going from indices to actual units
    kx, ky = kx*delta_k, ky*delta_k
    qx, qy = qx*delta_k, qy*delta_k
    #constructing p vector and taking norm
    px = kx + 0.5*qy + 0.5*sq3*qy
    py = ky - 0.5*sq3*qx + 0.5*qy
    p_vects = np.vstack((px,py))
    norm_p = np.linalg.norm(p_vects, axis = 0)
    
    return b,norm_p

def bispec_k(i):
    '''Does the actual computation, whearas compute_bispectrum simply helps 
    organize and performs the iteration.'''

    k_i = k_vals[i] #k vector we're "fixing"
    q_vects = k_vals[i+1:]
    sum_kq = k_i + q_vects
    
    #periodic boundary conditions for out of range vectors
    sum_kq = np.where(sum_kq <=(n//2-1), sum_kq, sum_kq -n)
    sum_kq = np.where(sum_kq >= -1*n//2, sum_kq, sum_kq +n)
        
    spec,p = bispectrum(k_i, q_vects, sum_kq)
        
    norms_q = k_norms[i+1:]
    norms_k = k_norms[i]*np.ones(len(norms_q))
        
    return spec, norms_k, norms_q, p

def bispec_length():
    '''returns the number of bispectra that will be computed'''
    N = n**2 -1
    length = 0
    for i in range(1,N):
        length += N -i
    return length

def compute_bispectrum(): 
    '''Computes the bispectrum for every combination of two vectors in kspace,
    as well as the norms norms of the two vectors for which the bispectrum was 
    evaluated, and the vector p. 
    
    I.e: bispec[i] = B(k,q),where norm(k) = norms_k[i] and norm(q) = norms_q[i]'''
    
    length = bispec_length()
    
    #initializing lists
    bispec = np.zeros(length) + 1j*np.zeros(length)
    norms_k = np.zeros(length)
    norms_q = np.zeros(length)
    p_bispec = np.zeros(length)
                        
    start_ind = 0
    end_ind = 0
    #iterating through every vector and filling up lists
    for i in tqdm(range(len(k_vals) -1), desc='Computing bispectra'):
        data = bispec_k(i)
        end_ind = start_ind + len(data[1])
        
        bispec[start_ind:end_ind] = data[0]
        norms_k[start_ind:end_ind] = data[1]
        norms_q[start_ind:end_ind] =data[2]
        p_bispec[start_ind:end_ind] =data[3]
        
        start_ind = end_ind
    return bispec, norms_k, norms_q, p_bispec
    

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

    
def tcf(field, length = 400, rbins = 200, cutoff = False):
    '''computes the triangle correlation function for given field
    -field: field, already in fourier space
    -length: realspace length of box(survey size)
    -rbins: number of r for which we want to compute s(r)
    -cutoff: whether we want to include the foreground wedge'''

    #declaring some global constans
    global epsilon_k, k_vals, k_norms, L, n, k_cutoff
    epsilon_k = field/np.abs(field) #phase factor of field
    n = field.shape[0]
    L = length
    k_cutoff = cutoff
    k_vals, k_norms = k_vects() #all k vectors to consider, and their norms
    
    bispectra, norms_k, norms_q, p = compute_bispectrum()

    array_memory = norms_k.nbytes/(10**9)
    print('\n in total, the bispectra, norms_k, norms_q and p arrays take up ', array_memory*3 + 2*array_memory, ' gigs of memory\n')
    
    r = np.linspace(0.5, 30, rbins)
    triangle_corr = compute_tcf(r, bispectra, norms_k, norms_q, p)
    
    return r, triangle_corr