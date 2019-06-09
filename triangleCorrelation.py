import numpy as np
from scipy.special import j0

def k_vects(n):
    '''Constructs array of all possible k_vectors for an n by n grid, excluding 0 vector. 
    Returns array of the form: k = [[k1x, k1y], [k2x, k2y], ...]. The array is sorted
    in terms of increasing norm, and also returns the corresponding norms.'''
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
    for i in range(len(k_norms)): #maybe use np.argwhere()? prob more efficient, idk if neglibily so. More elegant for sure though
        if k_norms[i] <= np.pi/r:
            norm_indices.append(i)
    return k_vals[norm_indices]

def in_range(vector):
    '''makes sure that the vector k + q is not out of range of the box'''
    norm_vect = np.linalg.norm(vector)
    if norm_vect >= 0.5*n - 0.5:
        inrange = False
    else:
        inrange = True
    return inrange

def translation(vector):
    '''goes back to the np coordinate system where [0,0] is first element of array.
    We need to do this because the field data is stored in this way.'''

    '''I don't think this is actually necessary... why is there a problem with having
    coordinate system centered at the topmost pixel of the grid? Can still sum vectors'''
    return vector + n//2

def bispec(k, q, sum_kq,r):
    '''computes the bispectrum for the three vectors k,q, and (k + q), and returns
    the product of the bispectrum and window function'''
    sq3 = np.sqrt(3)
    #p vector as defined in SOURCE
    px = k[0] + 0.5*q[0] + 0.5*sq3*q[1]
    py = k[0] - 0.5*sq3*q[0] + 0.5*q[1]
    norm_p = np.linalg.norm((px,py))

    #going back to numpy "coordinate system"
    k_np, q_np, sum_np = translation(k), translation(q), translation(sum_kq)

    sum_kq_ind = np.round(sum_np) #what pixel k+q falls into
    
    b = epsilon_k[k_np[0], k_np[1]]*epsilon_k[q_np[0], q_np[1]]*np.conj(epsilon_k[sum_kq_ind[0], sum_kq_ind[1]])
    window = j0(norm_p*r)
    
    return b*window
    
def sum_q(k_array, i, r):
    '''for a given k vector, this sums over all the possible q. Essentially "inner" sum in the 2D sum.
    -k_array: array of vectors satisfying norm(k) <= pi/r
    -i: index in k_array of the given k we are fixing
    -r: correlation scale for which we are computing s(r)'''

    summond = 0
    k_i = k_array[i]
    for q in k_array[i+1:]: #we slice k_array in this way to avoid duplicates
        sum_kq = k_i + q
        if in_range(sum_kq): # Makes sure k+q is not outside box
            summond += bispec(k_i, q, sum_kq,r) #computing bispectrum + window
        else:
            summond += 0
    return summond
            
def s(r):
    '''computes s(r) for a given r vector. s(r) as defined in INCLUDE SOURCE'''
    
    k_available = k_condition(r) # k_available = { k | norm(k) <= pi/r }
    sum_r = 0
    for i in range(len(k_available)): 
        sum_r += sum_q(k_available, i, r)
    sum_r *= (r/L)**(3)
    return sum_r

def tcf(field, length, NBINS = 20):
    '''computes the TCF for a given field.  
    -field: the field in question, in fourier space.
    -length: length of the box in real space units (eg Mpc)
    -NBINS: determines the number of correlation scales (r) that the TCF will be computed for'''

    #declaring some global constants 
    global epsilon_k, k_vals, k_norms, n, L
    epsilon_k = field/np.abs(field) #phase factor of field
    L = length
    n = field.shape[0]
    k_vals, k_norms = k_vects(n) #all possible k vectors and their norm, sorted
    
    #list of correlation scales we want to probe
    r = np.linspace(0.5, 50, NBINS)
    triangle_corr = []
    
    for scale in r: 
        triangle_corr.append(s(scale))
        
    return r,triangle_corr