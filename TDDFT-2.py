import numpy as np
from gpaw.utilities import unpack
from gpaw.mixer import DummyMixer
from ase.units import Hartree
from gpaw.utilities import unpack
import scipy
class TDDFT(object):
    def __init__(self,calc,direction=[0,0,1]):
        calc.density.mixer.__init__(DummyMixer(),calc.density.ncomponents,calc.density.gd)
        self.calc=calc
        self.direction=np.array(direction)
        self.nbands=calc.get_number_of_bands()
        self.wk=calc.get_k_point_weights()
        self.NK=self.wk.shape[0] 
        
        self.G_q=[self.calc.wfs.pd.get_reciprocal_vectors(q) for q in range(self.NK)]
        self.G_q=[np.sum(self.G_q[q]*self.direction[None,:],axis=1) for q in range(self.NK)]  
        
        self.norm=calc.wfs.gd.dv/calc.wfs.gd.N_c.prod()
        
        self.psi=[calc.wfs.kpt_u[q].psit_nG.copy() for q in range(self.NK)]
        self.wfs=np.zeros((self.NK,self.nbands,self.nbands),complex)
        self.fkn=np.zeros((self.NK,self.nbands),complex)
        #transition dipole element
        self.TDE=np.zeros((self.NK,self.nbands,self.nbands),complex)
        for q in range(self.NK):
            kpt=calc.wfs.kpt_u[q]
            self.wfs[q]=np.eye(self.nbands)
            self.fkn[q]=kpt.f_n
            for n in range(self.nbands):
                for m in range(n+1,self.nbands):
                    self.TDE[q,n,m]=np.sum(kpt.psit_nG[m].conj()*self.G_q[q]*kpt.psit_nG[n])
                    self.TDE[q,n,m]/=(kpt.eps_n[n]-kpt.eps_n[m])
                    self.TDE[q,n,m]*=self.norm
                    self.TDE[q,m,n]=self.TDE[q,n,m].conj()
                    
    def get_hamiltonian(self,q,E):
        H_GG, S_GG =self.calc.wfs.hs(self.calc.hamiltonian,q)    
        H_nn = np.dot(self.psi[q].conj(), np.dot(H_GG, self.psi[q].T))
#         S_nn = np.dot(self.psi[q].conj(), np.dot(S_GG, self.psi[q].T))
        return H_nn+self.TDE[q]*E
    
    def update(self):
        self.calc.occupations.calculate(self.calc.wfs)
        self.calc.density.update(self.calc.wfs)
        self.calc.hamiltonian.update(self.calc.density)        
        
    def propagate(self,E,dt,steps):
        self.P=np.zeros(steps,dtype=np.complex)
        for t in range(steps):
            for q in range(self.NK):
                H_nn=self.get_hamiltonian(q,E[t])
#                 eps,psi_new=scipy.linalg.eigh(H_nn)
                H_left = np.eye(self.nbands)+0.5j*dt*H_nn          
                H_right= np.eye(self.nbands)-0.5j*dt*H_nn
                self.wfs[q]=scipy.linalg.solve(H_left, np.dot(H_right,self.wfs[q]))
#                 self.wfs[q]=np.einsum('jk,lk,k,il->ij',psi_new.conj(),psi_new,np.exp(-1j*eps*dt),self.wfs[q])
                operator=np.linalg.multi_dot([self.wfs[q].T.conj(),self.TDE[q],self.wfs[q]])
                self.P[t]+=self.wk[q]*np.trace(self.fkn[q,:,None]*operator)
                
                kpt=self.calc.wfs.kpt_u[q]
#                 kpt.eps_n=eps
                kpt.psit.array[:]=np.dot(self.wfs[q],kpt.psit.array[:])
            self.update()