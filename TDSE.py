import numba
import numpy as np
import xc
from tqdm import tqdm
from scipy import linalg
from ase.units import Hartree, Bohr
from itertools import product
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.utilities import unpack
from gpaw.mixer import DummyMixer

class TDSE(object):
    def __init__(self,calc):
        self.calc=calc
        self.wfs=self.calc.wfs
        self.NK=len(calc.wfs.kpt_u)
        self.nbands=calc.get_number_of_bands()
        self.E=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.norm=calc.wfs.gd.dv
        self.fqn=np.zeros((self.NK,self.nbands))
        for q in range(self.NK):
            kpt=calc.wfs.kpt_u[q]
            self.fqn[q]=kpt.f_n
            self.E[q]=np.diag(kpt.eps_n)
        self.get_momentum_matrix()
    def get_momentum_matrix(self):
        self.momentum=np.zeros((3,self.NK,self.nbands,self.nbands),dtype=np.complex)
        for q in range(self.NK):
            kpt = self.calc.wfs.kpt_u[q]
            G=self.wfs.pd.get_reciprocal_vectors(q=kpt.q,add_q=False)
            for n in range(self.nbands):
                for m in range(self.nbands):
                    for i in range(3):
                        self.momentum[i,q,n,m]=self.wfs.pd.integrate(kpt.psit_nG[m],G[:,i]*kpt.psit_nG[n])
        return self.momentum
                        
    def current(self):
        return np.einsum('qn,qin,qjn,dqij->d',self.fqn,self.wfn.conj(),self.wfn,self.momentum)
        
    def propagate(self,dt,steps,A):
        self.wfn=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        for q in range(self.NK):
            self.wfn[q]=np.eye(self.nbands)        
        self.J=np.zeros((steps,3),dtype=np.complex)
        I=np.eye(self.nbands)
        for t in tqdm(range(steps)):
            self.J[t]=self.current()
            H=self.E+np.einsum('iqnm,i->qnm',self.momentum,A(t*dt))
            for q in range(self.NK):
                H_left = I+0.5j*dt*H[q]            
                H_right= I-0.5j*dt*H[q]
                self.wfn[q]=linalg.solve(H_left, H_right@self.wfn[q]) 
                
                
                
                
                
                
                
                
                
                