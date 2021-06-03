import numpy as np
from numpy import tensordot, kron, ones
import time
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint, Bounds
import networkx as nx


class QAOA_DC():
  
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
        
        #Top t string for candidate states
        self.candidates = None
        
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
    

    #Convert arg into element in Psi.
    
    def _toStr(self, n):
        convertString = "0123456789ABCDEF"
        if n < self.M:
            return convertString[n]
        else:
            return self._toStr(n//self.M) + convertString[n%self.M]

    def _getStr(self, n):
        Str = self._toStr(n)
        return "0"*(self.N - len(Str)) + Str

    #Get index of t top maximal states.
    def getCandidate(self, t):
        assert type(t) == int

        self.candidates = {}
        re_psi = np.reshape(self.psi,-1)
        argsort = np.argsort(re_psi*np.conjugate(re_psi))
        for ii in argsort[-t:]:
            ind = self._getStr(ii)
            exec("self.candidates.update({'" + str(ind) + "' : self.psi["+','.join(ind)+"]})")
        return self.candidates

#Get set of bitstring.
def get_bit(N, vec_size):
    size = int(np.log2(int(vec_size)))
    return "{0:{fill}{width}b}".format(int(N), fill='0', width=size)

#Get Prob. distribution in computatinal basis.
def get_dict(eigenstate):
    eigenstate = np.abs(eigenstate)**2
    dic = {}
    for i in range(len(eigenstate)):
        bit = get_bit(i, len(eigenstate))
        dic.update({bit[::-1]:eigenstate[i]})
    return dic


#Input with graph g and k size of shared nodes
def LGP(g, k):
    counter = 1 #Number of sharing nodes
    while counter < k:
        connectivity = [list(x) for x in g.edges()]
        paths = []
        if counter == 1:
            paths = list(g.nodes())
            paths = [[x] for x in paths]
        elif counter == 2:
            paths = connectivity
        else:
            nested = counter - 2
            pos_paths = []
            for nest in range(nested):
                for u in paths:
                    for v in connectivity:
                        if u[-1] == v[0]:
                            pos_paths.append(u.append(v[-1]))
            for pos_path in pos_paths:
                if len(np.unique(pos_path)) == counter:
                    paths.append(pos_path)
                    
        for p in paths:
            GG = g.copy()
            GG.remove_nodes_from(p)
            S = [list(c) for c in nx.connected_components(GG)]
            if len(S) == 2:
                print("shared : ", counter)
                Sub_graph = []
                for shared_nodes in S:
                    G_temp = g.copy()
                    G_temp.remove_nodes_from(shared_nodes)
                    Sub_graph.append(G_temp)
                return Sub_graph
        counter += 1
    print("G has connectivity above k")
    return None

#Input to be graph1, graph2, string_count1, string_count2
def QSR(g_1, g_2, str_cnt1, str_cnt2):
    com_cnt = {} #complete counts
    common_node = np.intersect1d(g_1.nodes(), g_2.nodes())
    nodes_g1, nodes_g2 = list(g_1.nodes()), list(g_2.nodes())
    for (str1, cnt1) in str_cnt1.items():
        for (str2, cnt2) in str_cnt2.items():
            #Check equality for shared bits
            for v in common_node:
                s_bit1, s_bit2 = nodes_g1.index(v), nodes_g2.index(v)
                validity = [str1[s_bit1] == str2[s_bit2] for _ in common_node]
            if  False not in validity:
                com_str = "" #Initialized with empty string
                for i in np.unique(list(g_1.nodes()) + list(g_2.nodes())):
                    if i in g_1.nodes():
                        com_str = com_str + str1[nodes_g1.index(i)]
                    else:
                        com_str = com_str + str2[nodes_g2.index(i)]
                        
                #min reconstruction scheme here
                #com_cnt[com_str] = np.min([cnt1, cnt2])
                #Or multiply
                com_cnt[com_str] = np.multiply(cnt1, cnt2)
    
    #Sort string-count map by counts in reverse order
    #Not used
    #com_cnt = sorted(com_cnt.items(), key=lambda x: x[1], reverse=True)
    return com_cnt

#Input to be asset_idx_1, asset_idx_2, string_count1, string_count2
def QSR2(idx1, idx2, str_cnt1, str_cnt2, numCandidates):
    com_cnt = {} #complete counts
    common_node = np.intersect1d(idx1, idx2)

    assert len(common_node) != 0, "No shared nodes."

    nodes_g1, nodes_g2 = list(idx1), list(idx2)
    for (str1, cnt1) in str_cnt1.items():
        for (str2, cnt2) in str_cnt2.items():
            #Check equality for shared bits
            for v in common_node:
                s_bit1, s_bit2 = nodes_g1.index(v), nodes_g2.index(v)
                validity = [str1[s_bit1] == str2[s_bit2] for _ in common_node]
            if  False not in validity:
                com_str = "" #Initialized with empty string
                for i in np.unique(list(idx1) + list(idx2)):
                    if i in idx1:
                        com_str = com_str + str1[nodes_g1.index(i)]
                    else:
                        com_str = com_str + str2[nodes_g2.index(i)]
                        
                #min reconstruction scheme here
                #com_cnt[com_str] = np.min([cnt1, cnt2])
                #Or multiply
                com_cnt[com_str] = np.multiply(cnt1, cnt2)
    
    #Sort string-count map by counts in reverse order
    #sort and get first numCandidates states.
    com_cnt = {kkeys: vvalues for kkeys, vvalues in sorted(com_cnt.items(), key=lambda x: x[1]*x[1].conj(), reverse=True)}
    com_cnt = dict(list(com_cnt.items())[0:numCandidates])
    return com_cnt

#Get cost from string-state.
def _getCost(state, Q, A, e):
    assert type(state) == str

    h = np.copy(Q)
    h[:,0] += -e*np.sum(A,axis=1)
    J = e*np.copy(A)
    c = 0
    #Cost onsite
    for i, action_i in enumerate(state):
        c -= h[i, int(action_i)]
    #Cost coupling
        for ii, action_ii in enumerate(state[:i]):
            if int(action_i) * int(action_ii) == 0:
                cash_flow = 0
            else:
                cash_flow = 1
            c -= J[i,ii] * cash_flow
    return c

def NearNode_DC_QAOA(subgroups_list, Qs, As, cycles=1, numCandidates=20):
    #Create input's parameters
    # Number of actions
    M = 7
    # Selected actions
    action_idx = np.arange(M)
    # epsilon
    e = 1

    cnt_list = []
    for count, idx_list in enumerate(subgroups_list):
        # Number of assets
        N = len(idx_list)
        # Q values
        Q = np.array(Qs[idx_list][:,action_idx].todense()) / 8000
        # A values
        A = np.array(As[idx_list][:,idx_list].todense())
        qaoa = QAOA_DC(N, M, Q, A, e, cycles)
        qaoa.optimized(maxiter=20,method='BFGS')
        cnt_list.append(qaoa.getCandidate(numCandidates))
    return cnt_list
    #Recursively reconstruct the quantum state
    #QSR2(subgroups_list[0], subgroups_list[1], cnt_list[0], cnt_list[1], numCandidates=numCandidates)