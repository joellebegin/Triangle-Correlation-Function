import numpy as np 
from scipy.special import j0
from tqdm import tqdm
from mpi4py import MPI
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from time import time 

def e_k(field):
    delta_k = 2*np.pi/L
    abs_field = np.abs(field)
    epsilonk = np.ones((n,n), dtype=complex)

    zero_ind = np.argwhere(abs_field < delta_k)
    epsilonk[zero_ind.transpose()[0], zero_ind.transpose()[1]] = 0

    ind_1 = np.where(epsilonk)
    epsilonk[ind_1]*= field[ind_1]/abs_field[ind_1]

    return epsilonk

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
    for i in tqdm(range(len(k_vals) -1), desc='computing bispectra'):
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
    
    # if max(n_k < np.pi/r_i):
    #     window = j0(r_i*p)
    #     sum_r = np.sum(spec*window)
    # else:
    #indices in norms where k&&q <= pi/r
    ind_kq = np.argwhere( (n_k > np.pi/r_i) & (n_q > np.pi/r_i) )
    if len(ind_kq) == 0:
        print('true')
        window = j0(r_i*p)
        sum_r = np.sum(spec*window)
    else:
        spec[ind_kq] =0
        p[ind_kq] =0
        
        window = j0(r_i*p)
        window[ind_kq] =0

        sum_r = np.sum(spec*window)
    return ((r_i/L)**3)*sum_r

def main():
    #initiating communication
    comm = MPI.COMM_WORLD
    myID = comm.rank

    master = 0
    numHelpers = comm.size -1

    r = np.linspace(0.5, 30, 200) #list of correlation scales
    num_tasks = len(r)
    num_active_helpers = min(numHelpers, num_tasks)

    if myID == master:
        file_loc = input() #master reads file location
        for helperID in range(1, num_active_helpers+1):
            comm.send(file_loc, dest = helperID, tag = helperID)
    
    elif myID <= num_active_helpers:
        #every helper loads file
        loc = comm.recv(source = master, tag = MPI.ANY_TAG)
        field = np.loadtxt(loc, dtype = complex, delimiter=',')

        #declaring some global constans
        #MAKE THIS STUFF INTO A METHOD/CLASS
        global epsilon_k, k_vals, k_norms, L, n
        n = field.shape[0]
        L = 400
        k_vals, k_norms = k_vects() #all k vectors to consider, and their norms
        epsilon_k = e_k(field) #phase factor of field

        spec, n_k, n_q, p = compute_bispectrum()


    if myID == master:
        
        triangle_corr = np.zeros(len(r))
        
        num_sent = 0
        
        #sending out initial assignments
        for helperID in range(1, num_active_helpers+1):
            print('I asked', helperID, 'to do index', helperID)
            comm.send(helperID-1, dest = helperID, tag = helperID)
            num_sent += 1

        #sending out tasks
        for i in range(1, num_tasks+1):
            status = MPI.Status()
            temp = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            sender = status.Get_source()
            tag = status.Get_tag()
            triangle_corr[tag] = np.real(temp)

            if num_sent < num_tasks:
                comm.send(num_sent, dest = sender, tag = 1)
                print('I asked', sender, 'to do index', num_sent +1)
                num_sent += 1
            else:
                #print('everything is done so I asked', sender, 'to pack up')
                comm.send(0, dest = sender, tag = 0)


        #MAKE THIS INTO A FUNCTION AS WELL
        fig, ax = plt.subplots(figsize = (20,10))
        ax.plot(r, triangle_corr)
        
        plt.rcParams.update({'font.size' :15, 'axes.labelsize': 30})

        ax.grid()
        ax.grid(color = '0.7')
        ax.set_facecolor('0.8')

        ax.set_ylim(-0.1, 0.5)
        ax.set_xlabel('r (Mpc)')
        ax.set_ylabel('s(r)')

        name = file_loc +'.png'
        plt.savefig(name)


        name = file_loc + '.csv'
        np.savetxt(name, triangle_corr, delimiter=',')
        
    elif myID <= num_active_helpers:
        complete = False
        while (complete == False):
            status = MPI.Status()
            assignment = comm.recv(source = master, tag = MPI.ANY_TAG, status = status)
            tag = status.Get_tag()

            if tag ==0:
                complete = True
            else:
                ri = r[assignment]
                sumr = sr(ri, spec, n_k, n_q, p)
                comm.send(sumr, dest = master, tag = assignment)

    comm.Barrier()

if __name__ == "__main__":
    main()
