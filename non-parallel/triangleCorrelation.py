import numpy as np
from mpi4py import MPI
from scipy.special import j0
from tqdm import tqdm
import matplotlib.pyplot as plt 
from time import time

class GridVectors():
    '''class that creates object of all vectors defined by the grid spacing, 
    as well as defining all the operations on them that compute the triangle
    correlation function'''
    def __init__(self, field, L=80):
    
        self.L = L #real space length of box
        self.delta_k = 2*np.pi/self.L #scaling factor for k space
        self.field = field 
        self.n = field.shape[0]

        x,y = np.indices((self.n,self.n)) -self.n//2
        flattened = np.vstack((x.flatten(),y.flatten())).transpose()
        norms = np.linalg.norm(flattened, axis = 1)*self.delta_k
        
        ind = np.argsort(norms)
        
        #vectors defined by grid spacing and their norms
        self.norms = norms[ind]
        self.vectors = flattened[ind]

        #defining phase factor of field
        abs_field = np.abs(self.field)
        
        self.epsilon_k = np.ones((self.n, self.n), dtype=complex)

        #we set pixels with absolute value < delta_k to have phase factor 0
        zero_ind = np.argwhere(abs_field < self.delta_k)
        self.epsilon_k[zero_ind.transpose()[0], zero_ind.transpose()[1]] = 0

        one_ind = np.where(self.epsilon_k) 
        self.epsilon_k[one_ind] *= self.field[one_ind]/abs_field[one_ind]
        
    def avail(self, num):
        '''determines set of { vect | norm(vect) <= num} for a given num'''
        if max(self.norms) <= num:#ie all vectors are available
            self.vect_avail = self.vectors
        else:
            ind = np.argmax(self.norms >= num)

            self.vect_avail = self.vectors[:ind] 
        
    def p_vector(self):
        sq3 = np.sqrt(3)

        #going from indices to actual units
        kx,ky = self.kx*self.delta_k, self.ky*self.delta_k
        qx,qy = self.qx*self.delta_k, self.qy*self.delta_k

        #constructing p vector and taking norm
        px = kx + 0.5*qy + 0.5*sq3*qy
        py = ky - 0.5*sq3*qx + 0.5*qy
        p_vects = np.vstack((px,py))
        norm_p = np.linalg.norm(p_vects, axis = 0)

        return norm_p

    def bispectrum(self):
        '''evaluates bispectrum for given k and arrray of q'''

        #have to go back to indices corresponding to the np array where data is stored
        kx, ky = self.kx + self.n//2, self.ky + self.n//2 
        qx, qy = self.qx + self.n//2, self.qy + self.n//2 
        sx, sy = self.sx + self.n//2, self.sy + self.n//2 

        return self.epsilon_k[ky,kx]*self.epsilon_k[qy,qx]*np.conj(self.epsilon_k[sy,sx])
    
    def summond(self):
        '''computes the sum for a fixed value of k'''
        self.kx,self.ky = self.k[0], self.k[1]
        self.qx,self.qy = self.q[:,0], self.q[:,1]
        self.sx,self.sy = self.s[:,0], self.s[:,1]  

        
        p = self.p_vector()
        b = self.bispectrum()

        temp = j0(p*self.r)*b

        return sum(temp)

    def tcf(self, i, r):
        self.r = r

        self.k = self.vect_avail[i] #k vector being 'fixed'
        self.q = self.vect_avail

        self.s = self.q+self.k

        #periodic boundary conditions for out of range vectors
        self.s = np.where(self.s <=(self.n//2-1), self.s, self.s-self.n)
        self.s = np.where(self.s >= -1*self.n//2, self.s, self.s +self.n)
        
        
        self.spec = self.summond()

def main():    
    
    field_loc = 'bubbles.txt'
    field = np.loadtxt(field_loc, delimiter=',', dtype=complex)
    k_vectors = GridVectors(field)
    
    #correlation scales for which to compute the TCF
    r = np.linspace(0.5, 15, 10)

    triangle_corr = np.zeros(len(r))  

    for ind in tqdm(range(len(r)), desc= 'iterating through r'):
        corr_ind = 0
        #determining vectors that satisfy norm(v) <= pi/r
        k_vectors.avail(np.pi/r[ind])

        for i in tqdm(range(len(k_vectors.vect_avail)), desc='computing sum'):
            k_vectors.tcf(i, r[ind])
            corr_ind += k_vectors.spec

        triangle_corr[ind] = np.real(corr_ind*((r[ind]/k_vectors.L)**3))
    
    np.savetxt('triangle_corr.txt', triangle_corr, delimiter=',')
if __name__ == "__main__":
    main()
    