import numpy as np
from ase.units import Bohr,Hartree
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from scipy.integrate import complex_ode
from tqdm import tqdm
class DensityMatrix(object):
    def __init__(self,calc):
        self.calc=calc
        self.wfs=self.calc.wfs
        self.wk=calc.get_k_point_weights() # weight of reduced Brillioun zone
        self.NK=self.wk.shape[0] 
        self.FBZ=calc.get_bz_k_points()
        self.FNK=self.FBZ.shape[0]
        self.nbands=calc.get_number_of_bands()
        self.nvalence=calc.get_number_of_electrons()
        self.vb=np.arange(0,self.nvalence/2,dtype=int)
        self.cb=np.arange(self.nvalence/2,self.nbands,dtype=int)  
        
        self.kd=KPointDescriptor([[0,0,0]])
        self.pd=PWDescriptor(ecut=self.wfs.ecut,gd=self.wfs.gd,kd=self.kd,dtype=complex)
        self.G=self.pd.get_reciprocal_vectors()
        self.NG=self.G.shape[0]
        self.volume = np.abs(np.linalg.det(calc.wfs.gd.cell_cv))
        self.alpha=0.3
        
    def get_coloumb_potential(self):
        G2=np.linalg.norm(self.G,axis=1)**2
        G2[G2==0]=np.inf
        return 4*np.pi/G2
    def get_long_range_xc(self):
        G2=np.linalg.norm(self.G,axis=1)**2
        G2[G2==0]=np.inf
        return -self.alpha/G2
        
    def get_pair_density(self,kpt,n_n,m_m):
        pair=np.zeros((len(n_n),len(m_m),self.NG),dtype=complex)
        for i,n in enumerate(n_n):
            psi_n=self.wfs.pd.ifft(kpt.psit_nG[n],kpt.q)
            for j,m in enumerate(m_m):
                psi_m=self.wfs.pd.ifft(kpt.psit_nG[m],kpt.q)
                pair[i,j]=self.pd.fft(psi_n.conj()*psi_m)
        pair*=self.wfs.gd.dv
        return pair

    def get_transition_dipole(self,kpt,n_n,m_m,direction):
        direction=np.array(direction)
        TD=np.zeros((len(n_n),len(m_m)),dtype=complex)
        G=self.wfs.pd.get_reciprocal_vectors(kpt.q)
        G=np.sum(G*direction[None,:],axis=1)
        for i,n in enumerate(n_n):
            for j,m in enumerate(m_m):
                if n!=m:
                    TD[i,j]=self.wfs.pd.integrate(kpt.psit_nG[m],G*kpt.psit_nG[n])
                    TD[i,j]/=(kpt.eps_n[n]-kpt.eps_n[m])
        return TD
    
    def get_transition_energy(self,kpt,n_n,m_m):
        TE=np.zeros((len(n_n),len(m_m)),dtype=float)
        for i,n in enumerate(n_n):
            for j,m in enumerate(m_m):
                TE[i,j]=kpt.eps_n[n]-kpt.eps_n[m]
        return TE
    
    def get_hamitonian(self,n_n,m_m,direction):
        self.NH=self.NK*len(n_n)*len(m_m)
        F=np.zeros((self.NK,len(n_n),len(m_m),
                    self.NK,len(n_n),len(m_m)),dtype=complex)
        TD=np.zeros((self.NK,len(n_n),len(m_m)),dtype=complex)
        TE=np.zeros((self.NK,len(n_n),len(m_m)),dtype=complex)
        V=self.get_coloumb_potential()+self.get_long_range_xc()
        
        for k1 in tqdm(range(self.NK)):
            kpt1=self.wfs.kpt_u[k1]
            TE[k1]+=self.get_transition_energy(kpt1,n_n,m_m)
            TD[k1]+=self.get_transition_dipole(kpt1,n_n,m_m,direction)
            rho1=self.get_pair_density(kpt1,n_n,m_m)
            for k2 in range(self.NK):
                kpt2=self.wfs.kpt_u[k2]
                rho2=self.get_pair_density(kpt2,n_n,m_m)
                F[k1,:,:,k2,:,:]+=np.einsum('G,ijG,klG->ijkl',V,rho1,rho2)/self.volume
        return F,TE,TD
    
    def propagate(self,n_n,m_m,E,steps,dt,T2,direction=[0,0,1]):
        F,TE,TD=self.get_hamitonian(n_n,m_m,direction)
        def d_rho(t,rho):
            rho=rho.reshape((self.NK,len(n_n),len(m_m)))
            drho=1j*(TD*E(t)-TE*rho-np.einsum('m,mlknij,mlk->nij',self.wk,F,rho)+1j*rho/T2)
            return drho.ravel()
        
        self.solver=complex_ode(d_rho)
        self.solver.set_initial_value(np.zeros(self.NH,complex), 0)
        self.P=np.zeros(steps,complex)
        self.time=np.zeros(steps)
        for i in tqdm(range(steps)):
            y=self.solver.y.reshape((self.NK,len(n_n),len(m_m)))
            self.P[i]=np.einsum('m,mlk,mlk',self.wk,TD,y)
            self.time[i]=self.solver.t
            self.solver.integrate(self.solver.t+dt)
            
            

            
            