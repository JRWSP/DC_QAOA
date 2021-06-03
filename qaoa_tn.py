import numpy as np
from numpy import tensordot, kron, ones
import time
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint, Bounds

class Relaxation_RoundUp():
    
    def __init__(self, N, M, Q, A, e):
        
        # Number of assets
        self.N = N
        
        # Number of actions
        self.M = M
        
        # Onsite field
        self.h = np.copy(Q)
        self.h[:,0] += -e*np.sum(A,axis=1)
        
        # Coupling
        self.J = e*np.copy(A)
        
        # Epsilon
        self.e = e
        
        # Optimized cost
        self.cost_min = None
        
        # Optimized bit
        self.x = None
        
    def cost(self, params):
        
        assert len(params) == self.N * self.M
        
        c = 0       

        for i in range(self.N):
            
            for j in range(self.M):
                
                c -= self.h[i,j] * params[self.M*i+j]
                
            for ii in range(i): 
            
                c -= self.J[i,ii] * params[self.M*i] * params[self.M*ii]
        
        return c
    
    
    def optimized(self, method='trust-constr',disp=False,maxiter=50):
        
        self.costs = []
        
        params = np.random.rand(self.M * self.N)
        
        A = np.zeros((self.N*self.M,self.N*self.M))
        for i in range(self.N):
            A[i*self.M:(i+1)*self.M][:,i*self.M:(i+1)*self.M] += 1
        
        L = LinearConstraint(A, np.ones(self.N*self.M), np.ones(self.N*self.M))
        b = Bounds(np.zeros(self.N*self.M), np.ones(self.N*self.M))
        
        self.res = minimize(self.cost, 
                            params, 
                            method=method, 
                            options={'disp':True,'maxiter':maxiter},
                            constraints=L,
                            bounds = b)
        
        x  = np.reshape(self.res.x,(self.N,self.M))
        self.x = np.zeros(self.N*self.M)
        for i in range(self.N):
            self.x[i*self.M + np.argmax(x[i,:])] = 1
            
        self.cost_min = self.cost(self.x)
            

class Relaxation():
    
    def __init__(self, N, M, Q, A, e):
        
        # Number of assets
        self.N = N
        
        # Number of actions
        self.M = M
        
        # Onsite field
        self.h = np.copy(Q)
        self.h[:,0] += -e*np.sum(A,axis=1)
        
        # Coupling
        self.J = e*np.copy(A)
        
        # Epsilon
        self.e = e
        
        # Optimized cost
        self.cost_min = None
        
        # Optimized bit
        self.x = None
        
    def cost(self, params):
        
        assert len(params) == self.N * self.M
        
        c = 0       

        for i in range(self.N):
            
            for j in range(self.M):
                
                c -= self.h[i,j] * params[self.M*i+j]
                
            for ii in range(i): 
            
                c -= self.J[i,ii] * params[self.M*i] * params[self.M*ii]
        
        return c
    
    
    def optimized(self, method='trust-constr',disp=False,maxiter=50):
        
        self.costs = []
        
        params = np.random.rand(self.M * self.N)
        
        A = np.zeros((self.N*self.M,self.N*self.M))
        for i in range(self.N):
            A[i*self.M:(i+1)*self.M][:,i*self.M:(i+1)*self.M] += 1
        
        L = LinearConstraint(A, np.ones(self.N*self.M), np.ones(self.N*self.M))
        b = Bounds(np.zeros(self.N*self.M), np.ones(self.N*self.M))
        
        self.res = minimize(self.cost, 
                            params, 
                            method=method, 
                            options={'disp':True,'maxiter':maxiter},
                            constraints=L,
                            bounds = b)
        
        self.x  = np.reshape(self.res.x,(self.N * self.M))
        #self.x = np.zeros(self.N*self.M)
        #for i in range(self.N):
        #    self.x[i*self.M + np.argmax(x[i,:])] = 1
            
        self.cost_min = self.cost(self.x)
           

class Exact():
    
    def __init__(self, N, M, Q, A, e):
        
        # Number of assets
        self.N = N
        
        # Number of actions
        self.M = M
        
        # Onsite field
        self.h = np.copy(Q)
        self.h[:,0] += -e*np.sum(A,axis=1)
        
        # Coupling
        self.J = e*np.copy(A)
        
        # Epsilon
        self.e = e
        
        # Exact cost
        self.costs = None
        
        # Optimized cost
        self.cost_min = None
        
        # Optimized bit
        self.x = None
        
        
    def optimized(self):
        
        self._set_exact_energies()
        
        self.cost_min = min(self.costs)
        
        self.x = self._index2bitstring(np.argmin(self.costs))
        
        
    def _index2bitstring(self,idx):
        
        x = np.zeros(self.N)
        
        num = np.array([int(i) for i in np.base_repr(idx, base=self.M)])
        
        x[(self.N-len(num)):] = num
       
        return x
    
    
    def _set_exact_energies(self):
        
        
        if self.costs is None:
        
            self.costs = np.zeros(self.M**self.N)
        
            coupling_node = np.zeros(self.M)
            coupling_node[0] = 1
            
            for i in range(self.N):
            
                self.costs += self._onsite_op(-self.h[i,:], i)
            
                for ii in range(i):
                    
                    if self.J[i,ii] != 0:
                    
                        self.costs += -self.J[i,ii] * self._twosite_op(coupling_node, coupling_node,ii,i)
                
        return 

        
    def _onsite_op(self,P,i):
        
        return kron(ones(self.M**i),kron(P,ones(self.M**(self.N-i-1))))

    
    def _twosite_op(self,P,Q,i,j):
        
        assert i < j
        
        return kron(ones(self.M**i),kron(P,kron(ones(self.M**(j-i-1)),kron(Q,ones(self.M**(self.N-j-1))))))
                
      
        
class QAOA_TN():
  
    def __init__(self, N, M, Q, A, e, T):
        
        # Number of assets
        self.N = N
        
        # Number of actions
        self.M = M
        
        # Onsite field
        self.h = np.copy(Q)
        self.h[:,0] += -e*np.sum(A,axis=1)
        
        # Coupling
        self.J = e*np.copy(A)
        
        # Epsilon
        self.e = e
        
        # Number of cycles
        self.T = T
        
        # Quantum state
        self.psi = None
        
        # Optimized costs
        self.costs = None
        
        # Optimized results
        self.res = None
        
        # The exact energies
        self.exact_energies = None
        
        
    def cost(self, params):
        
        assert len(params) == 2 * self.T 
        
        self.evolve(params)
        
        psi_bra = np.copy(np.conj(self.psi))
        
        c = 0
        
        for i in range(self.N): 
            
            psi_ket = np.copy(self.psi)
        
            c += self.inner_product(psi_bra, self._apply_h_B_onsite(psi_ket,i))                      
                
            for ii in range(i):
                
                if self.J[i,ii] != 0:
                    
                    psi_ket = np.copy(self.psi)
            
                    c += self.inner_product(psi_bra, self._apply_h_B_coupling(psi_ket,i,ii))
        
        self.costs += [np.real(c)]
        
        return np.real(c)
    
    
    def optimized(self, method='COBYLA',disp=False,maxiter=50):
        
        self.costs = []
        
        params = np.random.rand(2*self.T)
        
        self.res = minimize(self.cost, params, method=method, options={'disp':True,'maxiter':maxiter})
                
      
    def evolve(self, params):
        
        assert len(params) == 2 * self.T 
        
        self.restart(state="ones")
        
        for t in range(self.T):
            self._apply_U_B(params[2*t])
            self._apply_U_A(params[2*t+1])
 

    def restart(self,state="default"):
        
        if state=="default":
            self.psi = np.zeros(self.M**self.N,dtype='complex')
            self.psi[0] = 1
           
        if state=="ones":
            self.psi = np.ones(self.M**self.N,dtype='complex') / np.sqrt(self.M**self.N)
            
        self.psi = np.reshape(self.psi,[self.M]*self.N)  
        
        
    def inner_product(self,psi_1,psi_2):
        
        return np.tensordot(psi_1, psi_2, axes=(np.arange(self.N),np.arange(self.N)))
        
    
    # Apply the total U_A 
    def _apply_U_A(self, beta_t):
    
        u = self._get_u_A(beta_t)
    
        for i in range(self.N):
        
            assert np.shape(self.psi)[i] == self.M
        
            self.psi = np.tensordot(u,self.psi,axes=(1,i))  
          

    # Apply the total U_B
    def _apply_U_B(self, gamma_t):
    
        for i in range(self.N):
        
            assert np.shape(self.psi)[i] == self.M       
        
            self._apply_u_B_onsite(gamma_t,i)
                       
            for ii in range(i):
                
                if self.J[i,ii] != 0:
            
                    self._apply_u_B_coupling(gamma_t,i,ii)   
                
                
    # Get a tight-binding operator acting on each asset 
    def _get_u_A(self, beta_t):
    
        global vs, es
    
        # Eigenstates  |E_k> = a^\dagger_k |0>
        if 'vs' not in globals():
            vs = np.exp(-1j*np.array([[ 2*np.pi*k*j/self.M for k in range(self.M)] for j in range(self.M)]))
            vs = vs/ np.sqrt(self.M)
        
        # Eigenvalues e^{-iE_k} where E_k = 2*cos(k)
        if 'es' not in globals(): 
            es = np.exp(-1j*2*np.cos(np.arange(self.M)*2*np.pi/self.M))
        
        return np.conj(vs.T).dot(np.power(es,beta_t)[:,None]*vs)   
    
    
    # Apply an onsite term in U_B 
    def _apply_u_B_onsite(self, gamma_t,i):
    
        assert i < self.N
    
        u = np.exp(-1j*(-self.h[i,:]) * gamma_t)
    
        idx = '[' +'None,'*i + ':' + ',None'*(self.N-i-1) + ']'
    
        exec('self.psi *= u'+idx)


    # Apply a coupling term in U_B 
    def _apply_u_B_coupling(self, gamma_t,i,ii):
    
        assert i>ii
    
        idx = '['+':,'*ii + '0,' + ':,'*(i-ii-1) + '0' +',:'*(self.N-i-1) + ']'
    
        exec('self.psi' + idx +'*= np.exp(-1j*(-self.J[i,ii])*gamma_t)')
        
        
    # Apply an onsite term in H_B 
    def _apply_h_B_onsite(self, psi, i):
    
        assert i < self.N
    
        u = -self.h[i,:]
    
        idx = '[' +'None,'*i + ':' + ',None'*(self.N-i-1) + ']'
    
        exec('psi *= u'+idx)
        
        return psi


    def _apply_h_B_coupling(self, psi,i,ii):
    
        assert i>ii
    
        # -J * n_i * n_j
        h_B_coupling = np.zeros((self.M**2,self.M**2))
        h_B_coupling[0,0] = -self.J[i,ii]
        h_B_coupling = np.reshape(h_B_coupling,[self.M]*4)
            
        return np.tensordot(psi, h_B_coupling, axes=([ii,i],[0,1]))
    
    

