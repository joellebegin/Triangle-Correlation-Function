import numpy as np 
from scipy.special import j0

class Bispectra:

    def __init__(self, field, length = 400):
        self.field = field
        self.L = length
        
        self.epsilon_k = self.field/np.abs(self.field) #phase factor of field
        self.n = self.field.shape[0]
        self.k_vals, self.k_norms = k_vects()
        self.delta_k = 2*np.pi/self.L


        arr_length = bispec_length()
    
        self.bispec = np.zeros(arr_length) + 1j*np.zeros(arr_length)
        self.norms_k = np.zeros(arr_length)
        self.norms_q = np.zeros(arr_length)
        self.p_bispec = np.zeros(arr_length)

    def k_vects(self):
        '''Constructs array of all vectors for an n by n grid, excluding 
        0 vector. 

        Returns array of the form: k = [[k1x, k1y], [k2x, k2y], ...]'''

        #going to system where orign is pixel at (n//2,n//2)
        x,y = np.indices((self.n,self.n)) - self.n//2

        k = np.vstack((x.flatten(),y.flatten())).transpose()
        norms_k= np.linalg.norm(k, axis = 1)*delta_k

        ind = np.argsort(norms_k)

        #slice at one in order to not include zero vector
        return k[ind][1:], norms_k[ind][1:]
    
    def bispec_length(self):
        '''returns the number of bispectra that will be computed'''
        N = self.n**2 -1
        length = 0
        for i in range(1,N):
            length += N -i
        return length
    
    def bispectrum(k,q,s):
        '''evaluates bispectrum and constructs p vector

        Returns b = {B(k,q)}, for the given k and q, and p, the corresponding norm 
        of p for the bispectra'''

        sq3 = np.sqrt(3)
        
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
    
    
    def compute_bispectra(self):
        start_ind = 0
        end_ind = 0
        #iterating through every vector and filling up lists
        for i in tqdm(range(len(k_vals) -1), desc='computing bispectra'):
            data = bispec_k(i)
            end_ind = start_ind + len(data[1])

            self.bispec[start_ind:end_ind] = data[0]
            self.norms_k[start_ind:end_ind] = data[1]
            self.norms_q[start_ind:end_ind] =data[2]
            self.p_bispec[start_ind:end_ind] =data[3]

            start_ind = end_ind
            

def main():
    N = 6
    f = np.random.normal(size = (N,N)) + 1j*np.random.normal(size = (N,N))
    
    stuff = Bispectra(f, 200)
    stuff.compute_bispectra()