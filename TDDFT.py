import numba
import numpy as np
import xc
from tqdm import tqdm
from scipy import linalg
from ase.units import Hartree, Bohr
from itertools import product
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor

@numba.jit(nopython=True,parallel=True,fastmath=True)
def operator_matrix_periodic(matrix,operator,wf_conj,wf):
    """perform integration of periodic part of Kohn Sham wavefunction"""
    NK=matrix.shape[0]
    nbands=matrix.shape[1]
    for k in numba.prange(NK):
        for n1 in range(nbands):
            for n2 in range(nbands):
                matrix[k,n1,n2]=np.sum(operator[2:-1,2:-1,2:-1]*wf_conj[k,n1][2:-1,2:-1,2:-1]*wf[k,n2][2:-1,2:-1,2:-1])
    return matrix

class TDDFT(object):
    """
    Time-dependent DFT+Hartree-Fock in Kohn-Sham orbitals basis:
    
        calc: GPAW calculator (setups='sg15')
        nbands (int): number of bands in calculation
        
    """
    
    def __init__(self,calc,nbands=None):
        self.calc=calc
        self.K=calc.get_ibz_k_points() # reduced Brillioun zone
        self.NK=self.K.shape[0] 
        
        self.wk=calc.get_k_point_weights() # weight of reduced Brillioun zone
        if nbands is None:
            self.nbands=calc.get_number_of_bands()
        else:
            self.nbands=nbands
        self.nvalence=int(calc.get_number_of_electrons()/2)
        
        self.EK=[calc.get_eigenvalues(k)[:self.nbands] for k in range(self.NK)] # bands energy
        self.EK=np.array(self.EK)/Hartree
        self.shape=tuple(calc.get_number_of_grid_points()) # shape of real space grid
        self.density=calc.get_pseudo_density()*Bohr**3 # density at zero time
        
        
        # array of u_nk (periodic part of Kohn-Sham orbitals,only reduced Brillion zone)
        self.ukn=np.zeros((self.NK,self.nbands,)+self.shape,dtype=np.complex) 
        for k in range(self.NK):
            kpt = calc.wfs.kpt_u[k]
            for n in range(self.nbands):
                psit_G = kpt.psit_nG[n]
                psit_R = calc.wfs.pd.ifft(psit_G, kpt.q)
                self.ukn[k,n]=psit_R 
                
        self.icell=2.0 * np.pi * calc.wfs.gd.icell_cv # inverse cell 
        self.cell = calc.wfs.gd.cell_cv # cell
        self.r=calc.wfs.gd.get_grid_point_coordinates()
        for i in range(3):
            self.r[i]-=self.cell[i,i]/2.
        self.volume = np.abs(np.linalg.det(calc.wfs.gd.cell_cv)) # volume of cell
        self.norm=calc.wfs.gd.dv # 
        self.Fermi=calc.get_fermi_level()/Hartree #Fermi level
        
        #desriptors at q=gamma for Hartree
        self.kd=KPointDescriptor([[0,0,0]]) 
        self.pd=PWDescriptor(ecut=calc.wfs.pd.ecut,gd=calc.wfs.gd,kd=self.kd,dtype=complex)
        
        
        #Fermi-Dirac temperature
        self.temperature=calc.occupations.width
        
        #Fermi-Dirac distribution
        self.f=1/(1+np.exp((self.EK-self.Fermi)/self.temperature))
        
        self.Hartree_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.LDAx_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.LDAc_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        
        G=self.pd.get_reciprocal_vectors()
        G2=np.linalg.norm(G,axis=1)**2;G2[G2==0]=np.inf
        matrix=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        
        for k in tqdm(range(self.NK)):
            for n in range(self.nbands):
                
                density=self.norm*np.abs(self.ukn[k,n])**2
                
                operator=xc.VLDAx(density)
                self.LDAx_elements[k,n]=operator_matrix_periodic(matrix,operator,self.ukn.conj(),self.ukn)*self.norm
                operator=xc.VLDAc(density)
                self.LDAc_elements[k,n]=operator_matrix_periodic(matrix,operator,self.ukn.conj(),self.ukn)*self.norm
                
                density=self.pd.fft(density)
                operator=4*np.pi*self.pd.ifft(density/G2)  
                self.Hartree_elements[k,n]=operator_matrix_periodic(matrix,operator,self.ukn.conj(),self.ukn)*self.norm
        
        self.wavefunction=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex) 
        self.Kinetic=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex) 
        for k in range(self.NK):
            self.wavefunction[k]=np.eye(self.nbands)
            self.Kinetic[k]=np.diag(self.EK[k])
            
        self.VH0=np.einsum('kn,knqij->qij',self.occupation(self.wavefunction),self.Hartree_elements)
        self.VLDAc0=np.einsum('kn,knqij->qij',self.occupation(self.wavefunction),self.LDAc_elements)
        self.VLDAx0=np.einsum('kn,knqij->qij',self.occupation(self.wavefunction),self.LDAx_elements)
    
    
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
                    self.dipole[k,m,n]=self.dipole[k,n,m].conj()
        return self.dipole
    
    def occupation(self,wavefunction):
        return 2*np.sum(self.wk[:,None,None]*self.f[:,None,:]*np.abs(wavefunction)**2,axis=2)
    
    def fast_Hartree_matrix(self,wavefunction):
        return np.einsum('kn,knqij->qij',self.occupation(wavefunction),self.Hartree_elements)-self.VH0
    
    def fast_LDA_correlation_matrix(self,wavefunction):
        return np.einsum('kn,knqij->qij',self.occupation(wavefunction),self.LDAc_elements)-self.VLDAc0
    
    def fast_LDA_exchange_matrix(self,wavefunction):
        return np.einsum('kn,knqij->qij',self.occupation(wavefunction),self.LDAx_elements)-self.VLDAx0
    
    def propagate(self,dt,steps,E,direction,corrections=10):
        
        dipole=self.get_transition_matrix(direction)
        
        
        self.time_occupation=np.zeros((steps,self.nbands),dtype=np.complex) 
        self.polarization=np.zeros(steps,dtype=np.complex)
        
        self.time_occupation[0]=np.sum(self.occupation(self.wavefunction),axis=0)
        for k in range(self.NK):
            operator=np.linalg.multi_dot([self.wavefunction[k].T.conj(),dipole[k],self.wavefunction[k]])
            self.polarization[0]+=self.wk[k]*np.sum(operator.diagonal())
        
        for t in tqdm(range(1,steps)):
            H = self.Kinetic+E[t]*self.dipole
            H+= self.fast_Hartree_matrix(self.wavefunction)
            H+= self.fast_LDA_correlation_matrix(self.wavefunction)
            H+= self.fast_LDA_exchange_matrix(self.wavefunction)
            for k in range(self.NK):
                H_left = np.eye(self.nbands)+0.5j*dt*H[k]            
                H_right= np.eye(self.nbands)-0.5j*dt*H[k]
                self.wavefunction[k]=linalg.solve(H_left, H_right@self.wavefunction[k]) 
                operator=np.linalg.multi_dot([self.wavefunction[k].T.conj(),dipole[k],self.wavefunction[k]])
                self.polarization[t]+=self.wk[k]*np.sum(operator.diagonal())
            self.time_occupation[t]=np.sum(self.occupation(self.wavefunction),axis=0)