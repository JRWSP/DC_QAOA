# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 23:23:34 2021

@author: OKCOM
"""

import networkx as nx
import numpy as np

from qiskit import BasicAer
#from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.applications.ising import max_cut
#from qiskit.optimization.applications.ising.common import random_graph, sample_most_likely

#from qiskit import Aer
#from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import COBYLA
#from qiskit.visualization import plot_histogram

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
                Sub_graph = []
                for shared_nodes in S:
                    G_temp = g.copy()
                    G_temp.remove_nodes_from(shared_nodes)
                    Sub_graph.append(G_temp)
                return Sub_graph
        print(counter)
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
                com_cnt[com_str] = np.min([cnt1, cnt2])
                #Or multiply
                #com_cnt[com_str] = np.multiply(cnt1, cnt2)
    
    #Sort string-count map by counts in reverse order
    #Not used
    #com_cnt = sorted(com_cnt.items(), key=lambda x: x[1], reverse=True)
    return com_cnt

#Conventional QAOA by Qiskit.
#return Dict of counts (Prob. Distribution)
def qiskit_QAOA(g,p):
    w = nx.to_numpy_array(g) #Convert graph to transition matrix
    qubit_op, offset = max_cut.get_operator(w) #Get QUBO of MaxCut
    optimizer = COBYLA() #Choose a classical optimizer
    qaoa = QAOA(qubit_op, optimizer, quantum_instance=BasicAer.get_backend('statevector_simulator'), p=p)
    result = qaoa.compute_minimum_eigenvalue()
    vector = get_dict(result.eigenstate)
    return vector

#DC_QAOA
#g input graph; p layer of QAOA; t top_samples; k max qubit size.
def DC_QAOA(g, p, t, k):
    if len(g.nodes()) > k:
        #get exactly two subgraphs with JPG policy
        g1, g2 = LGP(g, k)
        common_node = np.intersect1d(g1.nodes(), g2.nodes())
        str_cnt1 = DC_QAOA(g1, p, t, k)
        str_cnt2 = DC_QAOA(g2, p, t, k)
        #weighted string-count maps by node size

        str_cnt1 = {k: v*len(k) for k, v in str_cnt1.items()}
        str_cnt2 = {k: v*len(k) for k, v in str_cnt2.items()}
        #reconstruct string-count map with LGP policy
        out_cnt = QSR(g1, g2, str_cnt1, str_cnt2)
    else:
        out_cnt = qiskit_QAOA(g, p)
        #sort
        #out_cnt = sorted(out_cnt.items(), key=lambda x: x[1], reverse=True)
    """
    #Not used for now. 
    #retain only top t(str, cnt) pairs by sorted order
    top_cnt = out_cnt[:t]
    #rescale total nubmer of counts to s or around
    cnt_sum = sum count for count in out_cnt
    out_cnt = {k:int(s*v/cnt_sum) for (k, v) in out_cnt}
    """
    return out_cnt