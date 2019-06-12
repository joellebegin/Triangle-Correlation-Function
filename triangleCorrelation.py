def k_vects():
    x,y = np.indices((n,n))
    delta_k = 2*np.pi/L
    
    #start at one in order to not include zero vector
    k = np.vstack((x.flatten(),y.flatten())).transpose()[1:]
    norms_k= np.linalg.norm(ks, axis = 1)
    ind = np.argsort(norms_k) #sorting
    
    return k[ind], norms_k[ind]*delta_k


def bispectrum(k,q,s):
    sq3 = np.sqrt(3)
    kx,ky = k[0],k[1]
    qx,qy = q[:,0], q[:,1]
    sx,sy = s[:,0], s[:,1]

    b = field[kx,ky]*field[qx,qy]*np.conj(field[sx,sy])
    
    px = kx + 0.5*qy + 0.5*sq3*qy
    py = ky - 0.5*sq3*qx + 0.5*qy
    p_vects = np.vstack((px,py))
    norm_p = np.linalg.norm(p_vects, axis = 0)
    
    return b,norm_p

def bispec_k(i):
    k_i = k_vals[i]
    q_vects = k_vals[i+1:]
    sum_kq = k_i + q_vects
    
    #the elements in (k + q) that are out of range
    indices = np.unique(np.argwhere(sum_kq >=n)[:,0])
    
    if len(indices) == len(sum_kq):
        return 
    else: 
        s_inrange = np.delete(sum_kq, indices,0)
        q_inrange = np.delete(q_vects, indices,0)
        
        spec,p = bispectrum(k_i, q_inrange, s_inrange)
        
        norm_k = k_norms[i]
        norms_q = np.delete(k_norms[i+1:], indices)
        norms_kq = np.vstack((norm_k*np.ones(len(norms_q)),norms_q)).transpose()
        
        return spec, norms_kq, p

def compute_bispectrum(): 
    #super inelegant, better way?
    bispec = np.array([])
    norms = np.array([[0,0]])
    p_bispec = np.array([])
    
    for i in range(len(k_vals) -1):
        data = bispec_k(i)
        if data is not None:
            bispec = np.append(bispec, data[0])
            norms = np.append(norms, data[1], axis = 0)
            p_bispec = np.append(p_bispec, data[2])        
    return bispec, norms[1:], p_bispec
    
    
def bin_r(r_i, spec, norms, p):
    indk = np.argwhere(norms[:,0] <= np.pi/r_i).flatten()
    indq = np.argwhere(norms[:,1] <= np.pi/r_i).flatten()
    ind_kq = np.intersect1d(indk, indq) #indices in norms where k&&q <= pi/r
    
    spec_good = spec[ind_kq]
    p_good = p[ind_kq]
    
    window = j0(r_i*p_good)
    sum_r = np.sum(spec_good*window)
    return ((r_i/L)**3)*sum_r

def compute_tcf(r, bispectra, norms_kq, p):
    t = []
    for scale in r: 
        t.append(bin_r(scale, bispectra, norms_kq, p))
    return t

    
def tcf(field, length):
    global k_vals, k_norms, n, L
    n = field.shape[0]
    L = length
    k_vals, k_norms = k_vects()
    
    bispectra, norms_kq, p = compute_bispectrum()
    
    r = np.linspace(0.5, 50, 500)
    triangle_corr = compute_tcf(r, bispectra, norms_kq, p)
    
    return r, triangle_corr
    