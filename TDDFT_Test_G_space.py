import numba
import numpy as np
import xc
from tqdm import tqdm
from scipy import linalg
from ase.units import Hartree, Bohr
from itertools import product
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor


class TDDFT(object):
    def __init__(self,calc,nbands=None):
        self.calc=calc
        self.K=calc.get_ibz_k_points() # reduced Brillioun zone
        self.NK=self.K.shape[0] 
        
        self.wk=calc.get_k_point_weights() # weight of reduced Brillioun zone
        self.nbands=calc.get_number_of_bands()
        self.nvalence=int(calc.get_number_of_electrons()/2)
        
        self.EK=[calc.get_eigenvalues(k)[:self.nbands] for k in range(self.NK)] # bands energy
        self.EK=np.array(self.EK)/Hartree
      
        self.volume = np.abs(np.linalg.det(calc.wfs.gd.cell_cv)) # volume of cell
        self.norm=calc.wfs.gd.dv # 
        self.Fermi=calc.get_fermi_level()/Hartree #Fermi level
        
        #desriptors at q=gamma for Hartree
        self.kd=KPointDescriptor([[0,0,0]]) 
        self.pd=PWDescriptor(ecut=calc.wfs.pd.ecut,gd=calc.wfs.gd,kd=self.kd,dtype=complex)
        
        #distribution
        self.fkn=np.zeros((self.NK,self.nbands))
        self.Hartree_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.LDAx_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.LDAc_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        
        G=self.pd.get_reciprocal_vectors()
        G2=np.linalg.norm(G,axis=1)**2;G2[G2==0]=np.inf
        for k1 in tqdm(range(self.NK)):
            kpt1=calc.wfs.kpt_u[k1]
            self.fkn[k1]=kpt1.f_n
            for b in range(self.nbands):
                psi=calc.wfs.pd.ifft(kpt1.psit_nG[b],kpt1.q)
                den=self.norm*psi.conj()*psi
                Hart=4*np.pi*self.pd.fft(den)/G2
                LDAx=self.pd.fft(xc.VLDAx(den))
                LDAc=self.pd.fft(xc.VLDAc(den))
                
                for k2 in range(self.NK):
                    kpt2=calc.wfs.kpt_u[k2]
                    for n in range(self.nbands):
                        psi_n=calc.wfs.pd.ifft(kpt2.psit_nG[n],kpt2.q)
                        for m in range(self.nbands):
                            psi_m=calc.wfs.pd.ifft(kpt2.psit_nG[m],kpt2.q)
                            rho_nm=self.pd.fft(psi_n.conj()*psi_m)*self.norm
                            self.Hartree_elements[k1,b,k2,n,m]=np.sum(Hart*rho_nm)/self.volume 
                            self.LDAx_elements[k1,b,k2,n,m]=np.sum(LDAx*rho_nm)/self.volume 
                            self.LDAx_elements[k1,b,k2,n,m]=np.sum(LDAc*rho_nm)/self.volume 
                 
        self.wavefunction=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex) 
        self.Kinetic=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex) 
        for k in range(self.NK):
            self.wavefunction[k]=np.eye(self.nbands)
            self.Kinetic[k]=np.diag(self.EK[k])
            
        occ=2*np.sum(self.fkn[:,:,None]*np.abs(self.wavefunction)**2,axis=1)   
        self.VH0=np.einsum('kn,knqij->qij',occ,self.Hartree_elements)
        self.VLDAc0=np.einsum('kn,knqij->qij',occ,self.LDAc_elements)
        self.VLDAx0=np.einsum('kn,knqij->qij',occ,self.LDAx_elements)
    
    
    def get_transition_matrix(self,direction):
        direction/=np.linalg.norm(direction)
        self.dipole=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        for k in range(self.NK):
            kpt = self.calc.wfs.kpt_u[k]
            G=self.calc.wfs.pd.get_reciprocal_vectors(q=k,add_q=True)
            G=np.sum(G*direction[None,:],axis=1)
            for n in range(self.nvalence):
                for m in range(self.nvalence,self.nbands):
                    wfn=kpt.psit_nG[n];wfm=kpt.psit_nG[m]
                    self.dipole[k,n,m]=self.calc.wfs.pd.integrate(wfm,G*wfn)/(self.EK[k,n]-self.EK[k,m])
                    self.dipole[k,m,n]=-self.dipole[k,n,m].conj()
        return self.dipole
    
   
    def fast_hamiltonian(self,wavefunction):
        occ=2*np.sum(self.fkn[:,:,None]*np.abs(wavefunction)**2,axis=1)
        VH=np.einsum('kn,knqij->qij',occ,self.Hartree_elements)-self.VH0
        VLDAx=np.einsum('kn,knqij->qij',occ,self.LDAx_elements)-self.VLDAx0
        VLDAc=np.einsum('kn,knqij->qij',occ,self.LDAc_elements)-self.VLDAc0
        H=self.Kinetic+VH+VLDAx+VLDAc
        return H
    
    def propagate(self,dt,steps,E,direction):
        
        dipole=self.get_transition_matrix(direction)
        self.time_occupation=np.zeros((steps,self.nbands),dtype=np.complex) 
        self.polarization=np.zeros(steps,dtype=np.complex)
        
        for k in range(self.NK):
            operator=np.linalg.multi_dot([self.wavefunction[k].T.conj(),dipole[k],self.wavefunction[k]])
            self.polarization[0]+=self.wk[k]*np.sum(operator.diagonal())
        
        for t in tqdm(range(1,steps)):
            H = self.fast_hamiltonian(self.wavefunction)+E[t]*self.dipole
            for k in range(self.NK):
                H_left = np.eye(self.nbands)+0.5j*dt*H[k]            
                H_right= np.eye(self.nbands)-0.5j*dt*H[k]
                self.wavefunction[k]=linalg.solve(H_left, H_right@self.wavefunction[k]) 
                operator=np.linalg.multi_dot([self.wavefunction[k].T.conj(),dipole[k],self.wavefunction[k]])
                self.polarization[t]+=self.wk[k]*np.sum(operator.diagonal())
                
                
                
                
                
                
                
                
                
                
                