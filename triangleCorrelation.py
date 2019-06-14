import numpy as np 
from scipy.special import j0
from tqdm import tqdm

def k_vects():
    '''Constructs array of all possible k_vectors for an n by n grid, excluding 
    0 vector. Returns array of the form: k = [[k1x, k1y], [k2x, k2y], ...]'''
    x,y = np.indices((n,n))
    delta_k = 2*np.pi/L #scaling factor for k vector
    
    #start at one in order to not include zero vector
    k = np.vstack((x.flatten(),y.flatten())).transpose()[1:]
    norms_k= np.linalg.norm(k, axis = 1)
    ind = np.argsort(norms_k) #sorting
    
    return k[ind], norms_k[ind]*delta_k


def bispectrum(k,q,s):
    '''evaluates bispectrum and constructs p vector'''
    sq3 = np.sqrt(3)
    #separating components
    kx,ky = k[0],k[1]
    qx,qy = q[:,0], q[:,1]
    sx,sy = s[:,0], s[:,1]

    #evaluating bispectrum
    b = epsilon_k[kx,ky]*epsilon_k[qx,qy]*np.conj(epsilon_k[sx,sy])
    
    #constructing p vector and taking norm
    px = kx + 0.5*qy + 0.5*sq3*qy
    py = ky - 0.5*sq3*qx + 0.5*qy
    p_vects = np.vstack((px,py))
    norm_p = np.linalg.norm(p_vects, axis = 0)
    
    return b,norm_p

def bispec_k(i):
    '''Does the actual computation, whearas compute_bispectrum simply helps 
    organize and performs the iteration.'''

    k_i = k_vals[i]
    q_vects = k_vals[i+1:]
    sum_kq = k_i + q_vects
    
    #the elements in (k + q) that are out of range. we simply don't include these
    indices = np.unique(np.argwhere(sum_kq >=n)[:,0])
    
    if len(indices) == len(sum_kq): #if True, all (k+q) are out of range
        return 
    else: 
        #deleting elements that result in an out of range vector
        s_inrange = np.delete(sum_kq, indices,0) 
        q_inrange = np.delete(q_vects, indices,0)
        
        spec,p = bispectrum(k_i, q_inrange, s_inrange)
        
        
        norms_q = np.delete(k_norms[i+1:], indices)
        norms_k = k_norms[i]*np.ones(len(norms_q))
        
        #norms_kq = np.vstack((norm_k*np.ones(len(norms_q)),norms_q)).transpose()
        
        return spec, norms_q, norms_k, p

def compute_bispectrum(): 
    '''Computes the bispectrum for every combination of two vectors in kspace,
    as well as the norms norms of the two vectors for which the bispectrum was 
    evaluated, and the vector p. 
    
    I.e: bispec[i] = B(k,q),where norm(k) = norms[i][0] and norm(q) = norms[i][1]'''
    
    #initializing arrays
    bispec = []
    norms_k = []
    norms_q = []
    p_bispec = []
    
    #iterating through every vector
    for i in tqdm(range(len(k_vals) -1), desc= 'Computing bispectra'):
        data = bispec_k(i)

        if data is not None: 
            bispec.append(data[0])
            norms_q.append(data[1])
            norms_k.append(data[2])
            p_bispec.append(data[3])        
   
    return np.hstack(bispec), np.hstack(norms_k), np.hstack(norms_q), 
        np.hstack(p_bispec)
    
    
def sr(r_i, spec, n_k, n_q, p):
    '''given the value of r, determines which bispectra satisfy the pi/r cutoff.
    Computes window function for each value, sums over w*B, and multiplies by 
    prefactor.'''
    #indices in norms where k&&q <= pi/r
    ind_kq = np.argwhere((n_k <= np.pi/r_i) & (n_q <= np.pi/r_i) )

    spec_good = spec[ind_kq]
    p_good = p[ind_kq]
    
    window = j0(r_i*p_good)
    sum_r = np.sum(spec_good*window)
    return ((r_i/L)**3)*sum_r

def compute_tcf(r, bispectra, n_k, n_q, p):
    '''iterates through the correlation scales'''
    t = []
    for scale in tqdm(r, desc='Computing s(r)'): 
        t.append(sr(scale, bispectra, n_k, n_q, p))
    return np.array(t)

    
def tcf(field, length = 400, rbins = 200):
    '''computes the triangle correlation function for given field
    -field: field, already in fourier space
    -length: realspace length of box(survey size)'''

    global epsilon_k, k_vals, k_norms, L, n
    epsilon_k = field/np.abs(field)
    n = field.shape[0]
    L = length
    k_vals, k_norms = k_vects()
    
    bispectra, norms_k, norms_q, p = compute_bispectrum()
   
    r = np.linspace(0.5, 50, rbins)
    triangle_corr = compute_tcf(r, bispectra, norms_k, norms_q, p)
    
    return r, triangle_corr
    

