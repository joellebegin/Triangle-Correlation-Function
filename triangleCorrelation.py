import numpy as np
from mpi4py import MPI
from scipy.special import j0
from tqdm import tqdm
import matplotlib.pyplot as plt 
from time import time

'''THIS IS LATEST VERSION'''
class GridVectors():
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

        zero_ind = np.argwhere(abs_field < self.delta_k)
        self.epsilon_k[zero_ind.transpose()[0], zero_ind.transpose()[1]] = 0

        one_ind = np.where(self.epsilon_k) 
        self.epsilon_k[one_ind] *= self.field[one_ind]/abs_field[one_ind]

    def avail(self, num):
        '''determines { vects | norm(vects) <= num} for a given num'''
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
        '''evaluates bispectrum for given k and q'''
        kx, ky = self.kx + self.n//2, self.ky + self.n//2 
        qx, qy = self.qx + self.n//2, self.qy + self.n//2 
        sx, sy = self.sx + self.n//2, self.sy + self.n//2 

        return self.epsilon_k[ky,kx]*self.epsilon_k[qy,qx]*np.conj(self.epsilon_k[sy,sx])
    
    def summond(self):
        '''computes sum'''
        self.kx,self.ky = self.k[0], self.k[1]
        self.qx,self.qy = self.q[:,0], self.q[:,1]
        self.sx,self.sy = self.s[:,0], self.s[:,1]  

        
        p = self.p_vector()
        b = self.bispectrum()

        temp = j0(p*self.r)*b

        return sum(temp)

    def tcf(self, i, r):
        self.r = r

        self.k = self.vect_avail[i]
        self.q = self.vect_avail[i:]

        self.s = self.q+self.k

        #periodic boundary conditions for out of range vectors
        self.s = np.where(self.s <=(self.n//2-1), self.s, self.s-self.n)
        self.s = np.where(self.s >= -1*self.n//2, self.s, self.s +self.n)
        
        
        self.spec = self.summond()

def display(r,tcf):
    fig, ax = plt.subplots(figsize = (20,10))
    ax.plot(r, tcf)
    plt.show()

def main():
    comm = MPI.COMM_WORLD
    myID = comm.rank

    master = 0 
    num_helpers = comm.size -1
    
    if myID == master:
        field_loc = input()
        for helperID in range(1, num_helpers + 1):
            comm.send(field_loc, dest = helperID, tag = helperID)

    elif myID <= num_helpers:
        field_loc = comm.recv(source = master, tag = MPI.ANY_TAG)
        
    field = np.loadtxt(field_loc, delimiter=',', dtype=complex)
    k_vectors = GridVectors(field)

    r = np.linspace(0.5, 15, 400)

    triangle_corr = np.zeros(len(r))  

    for ind in tqdm(range(len(r)), desc= 'iterating through r'):
        
        k_vectors.avail(np.pi/r[ind])
        num_tasks = len(k_vectors.vect_avail)
        num_active_helpers = min(num_helpers, num_tasks)
        
        if myID == master:   

            corr_ind = 0
            num_sent = 0 
            
            for helperID in range(1, num_active_helpers + 1):
                comm.send(helperID -1, dest = helperID, tag = helperID)
                num_sent += 1


            for i in tqdm(range(1, num_tasks+1), desc='computing sum'):
                status = MPI.Status()
                temp = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
                sender = status.Get_source()
                tag = status.Get_tag()

                corr_ind += temp
                if num_sent < num_tasks:
                    comm.send(num_sent, dest = sender, tag = 1)
                    num_sent+=1

                else:
                    comm.send(0, dest = sender, tag = 0)

            triangle_corr[ind] = np.real(corr_ind*(r[ind]/k_vectors.L)**3)
        
        elif myID <= num_active_helpers:
            complete = False
            while complete == False:
                status = MPI.Status()
                assignment = comm.recv(source = master, tag = MPI.ANY_TAG, status = status)
                tag = status.Get_tag()
                if tag == 0:
                    complete = True
                else:
                    k_vectors.tcf(assignment, r[ind])
                    # print(k_vectors.s)
                    comm.send(k_vectors.spec, dest = master, tag = assignment)

        comm.Barrier()

        
    if myID == master:
        filename = field_loc + '.csv'        
        np.savetxt(filename, triangle_corr, delimiter = ',')
if __name__ == "__main__":
    main()
    
