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
class TDDFT(object):
    def __init__(self,calc):
        self.calc=calc
        self.wfs=self.calc.wfs
        
        self.K=calc.get_ibz_k_points() # reduced Brillioun zone
        self.NK=self.K.shape[0] 
        self.wk=calc.get_k_point_weights() # weight of reduced Brillioun zone
        self.nbands=calc.get_number_of_bands()
        self.norm=calc.wfs.gd.dv # 

        self.Ekin=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.norm=calc.wfs.gd.dv
        self.fqn=np.zeros((self.NK,self.nbands))
        N = calc.wfs.pd.tmp_R.size
        
        self.psi=np.zeros((self.NK,self.nbands,)+tuple(self.wfs.gd.N_c[:]),dtype=np.complex)
        self.Hartree_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.LDAx_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.LDAc_elements=np.zeros((self.NK,self.nbands,self.NK,self.nbands,self.nbands),dtype=np.complex)
        self.kd=KPointDescriptor([[0,0,0]]) 
        self.pd=PWDescriptor(ecut=calc.wfs.pd.ecut,gd=calc.wfs.gd,kd=self.kd,dtype=complex)
        
        G=self.pd.get_reciprocal_vectors();
        G2=np.linalg.norm(G,axis=1);
        G2=G2**2;
        G2[0]=np.inf   
        
        for q in range(self.NK):
            kpt=self.wfs.kpt_u[q]
            self.Ekin[q]=np.diag(kpt.eps_n)
            self.fqn[q]=kpt.f_n
            for n in range(self.nbands):
                self.psi[q,n]=self.wfs.pd.ifft(kpt.psit_nG[n],kpt.q)
                
        for q in tqdm(range(self.NK)):
            for n in range(self.nbands):
                density=np.abs(self.psi[q,n])**2
                
                VLDAx=xc.VLDAx(density)
                VLDAc=xc.VLDAc(density)
                
                nG=self.pd.fft(density)                
                VHG=4*np.pi*nG/G2
                VH=self.pd.ifft(VHG)
                
                self.Hartree_elements[q,n]=np.einsum('kixyz,kjxyz,xyz->kij',self.psi.conj(),self.psi,VH)
                self.LDAx_elements[q,n]=np.einsum('kixyz,kjxyz,xyz->kij',self.psi.conj(),self.psi,VLDAx)
                self.LDAc_elements[q,n]=np.einsum('kixyz,kjxyz,xyz->kij',self.psi.conj(),self.psi,VLDAc)
                
        self.Hartree_elements*=self.norm
        self.LDAx_elements*=self.norm
        self.LDAc_elements*=self.norm


    def get_transition_matrix(self,direction):
        direction/=np.linalg.norm(direction)
        self.dipole=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        for k in range(self.NK):
            kpt = self.calc.wfs.kpt_u[k]
            G=self.wfs.pd.get_reciprocal_vectors(q=k,add_q=True)
            G=np.sum(G*direction[None,:],axis=1)
            for n in range(self.nbands):
                for m in range(self.nbands):
                    if n!=m:
                        self.dipole[k,n,m]=self.wfs.pd.integrate(kpt.psit_nG[m],G*kpt.psit_nG[n])
                        self.dipole[k,n,m]/=(kpt.eps_n[n]-kpt.eps_n[m])
                        
    def polarization(self):
        return np.einsum('qn,qin,qjn,qij',self.fqn,self.wfn.conj(),self.wfn,self.dipole)
       
    def hamiltonian(self,wfn):
        occ=2*np.einsum('qn,qin->qn',self.fqn,np.abs(wfn)**2)
        VH=np.einsum('kn,knqij->qij',occ,self.Hartree_elements)-self.VH0
        VLDAx=np.einsum('kn,knqij->qij',occ,self.LDAx_elements)-self.VLDAx0
        VLDAc=np.einsum('kn,knqij->qij',occ,self.LDAc_elements)-self.VLDAc0
        return self.Ekin+VH+VLDAx+VLDAc
        
    def propagate(self,dt,steps,E,direction,n_corrections=3):
        self.wfn=np.zeros((self.NK,self.nbands,self.nbands),dtype=np.complex)
        I=np.eye(self.nbands)
        for q in range(self.NK):self.wfn[q]=I
        occ=2*np.einsum('qn,qin->qn',self.fqn,np.abs(self.wfn)**2)
        self.VH0=np.einsum('kn,knqij->qij',occ,self.Hartree_elements)
        self.VLDAx0=np.einsum('kn,knqij->qij',occ,self.LDAx_elements)
        self.VLDAc0=np.einsum('kn,knqij->qij',occ,self.LDAc_elements)
        self.get_transition_matrix(direction)
        self.P=np.zeros(steps,dtype=np.complex)
        H=self.hamiltonian(self.wfn)
        for t in tqdm(range(steps)):
            self.P[t]=self.polarization()
            wfn_next=np.copy(self.wfn)
            for i in range(n_corrections):
                H_next=self.hamiltonian(wfn_next)+E(t*dt)*self.dipole
                H_mid=0.5*(H+H_next)
                for q in range(self.NK):
                    H_left = I+0.5j*dt*H_mid[q]            
                    H_right= I-0.5j*dt*H_mid[q]
                    wfn_next[q]=linalg.solve(H_left, H_right@self.wfn[q]) 
            self.wfn=np.copy(wfn_next)
            H=H_next
                
                
                
                
                
                
                
                
                
                