import numpy as np 
from scipy.special import j0
from tqdm import tqdm
from mpi4py import MPI
import matplotlib.pyplot as plt


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
    N_pix = n**2 -1
    length = 0
    for i in range(1,N_pix):
        length += N_pix -i
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
    for i in tqdm(range(len(k_vals)), desc='Computing bispectra'):
        data = bispec_k(i)
        end_ind = start_ind + len(data[1])
        
        bispec[start_ind:end_ind] = data[0]
        norms_k[start_ind:end_ind] = data[1]
        norms_q[start_ind:end_ind] =data[2]
        p_bispec[start_ind:end_ind] =data[3]
        
        start_ind = end_ind
    
    return bispec, norms_k, norms_q, p_bispec


def display(sr_data):
    x = np.linspace(0.5, 30, 500)
    y = sr_data

    fig, ax = plt.subplots(figsize = (20,10))
    ax.plot(x, y)
    
    plt.rcParams.update({'font.size' :15, 'axes.labelsize': 30})

    ax.grid()
    ax.grid(color = '0.7')
    ax.set_facecolor('0.8')

    #ax.set_ylim(-0.01, 0.5)
    ax.set_xlabel('r (Mpc)')
    ax.set_ylabel('s(r)')

    plt.show()

def main():
    comm = MPI.COMM_WORLD
    myID = comm.rank

    master = 0
    numHelpers = comm.size -1

    N = 100
    field = np.random.normal(size = (N,N)) + 1j*np.random.normal(size = (N,N))
    #declaring some global constans
    global epsilon_k, k_vals, k_norms, L, n
    epsilon_k = field/np.abs(field) #phase factor of field
    n = field.shape[0]
    L = 400
    k_vals, k_norms = k_vects() #all k vectors to consider, and their norms

    spec, n_k, n_q, p = compute_bispectrum()

    r = np.linspace(0.5, 30, 500)
    num_tasks = len(r)
    num_active_helpers = min(numHelpers, num_tasks)

    if myID == master:
        
        triangle_corr = np.zeros(len(r))
        
        num_sent = 0
        print('sending out assignments')
        for helperID in range(1, num_active_helpers+1):
            print('I asked', helperID, 'to do index', helperID)
            comm.send(helperID-1, dest = helperID, tag = helperID)
            num_sent += 1

        for i in range(1, num_tasks+1):
            status = MPI.Status()
            temp = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            sender = status.Get_source()
            tag = status.Get_tag()
            triangle_corr[tag] = temp

            if num_sent < num_tasks:
                comm.send(num_sent, dest = sender, tag = 1)
                print('I asked', sender, 'to do index', num_sent +1)
                num_sent += 1
            else:
                print('everything is done so I asked', sender, 'to pack up')
                comm.send(0, dest = sender, tag = 0)

        display(triangle_corr)
    
    elif myID <= num_active_helpers:
        complete = False
        while (complete == False):
            status = MPI.Status()
            assignment = comm.recv(source = master, tag = MPI.ANY_TAG, status = status)
            tag = status.Get_tag()

            if tag ==0:
                complete = True
            else:
                r_i = r[assignment]

                ind_kq = np.argwhere( (n_k > np.pi/r_i) & (n_q > np.pi/r_i) )

                spec[ind_kq] =0
                p[ind_kq] =0
                
                window = j0(r_i*p)
                window[ind_kq] =0

                sum_r = np.sum(spec*window)
                
                comm.send(((r_i/L)**3)*sum_r, dest = master, tag = assignment)

    comm.Barrier()

if __name__ == "__main__":
    main()