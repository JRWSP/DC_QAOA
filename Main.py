# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 19:56:14 2021

@author: OKCOM
"""
#Some libraries are unused for now.
#import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
import Func

from qiskit import BasicAer
#from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.applications.ising import max_cut
#from qiskit.optimization.applications.ising.common import random_graph, sample_most_likely

#from qiskit import Aer
#from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.visualization import plot_histogram

if __name__ == '__main__':
    #Define graph from the original paper
    #Please add each nodes before adding edges to avoid incorrect bitstring ordering.
    G_paper = nx.Graph()
    for ii in range(5):
        G_paper.add_node(ii)
    edges = [[0, 1],
             [0, 2],
             [1, 2],
             [2, 3],
             [3, 4]]
    for edge in edges:
        G_paper.add_edge(edge[0], edge[1], weight=1.0)
    #Show the graph
    #nx.draw(G_paper)
        
    #Testing graph.
    G_test = nx.Graph()
    for ii in range(6):
        G_test.add_node(ii)
    edges = [(0, 1),  
             (0, 3),
             (1, 2),
             (1, 3), 
             (2, 0), 
             (2, 3), 
             (3, 4), 
             (3, 5), 
             (4, 5)]
    for edge in edges:
        G_test.add_edge(edge[0], edge[1], weight=1.0)

    

    p = 1 #QAOA circuit depth
    
    #QAOA Optimize the whole graph.
    vc = Func.qiskit_QAOA(G_test, p)
    plot_histogram(vc, figsize=(10, 8), title="QAOA, p="+str(p), bar_labels=False)
    
    #Divide-and-Conquer QAOA. 
    # variable k is size of sub-graph, t is not used at the moment. 
    dc = Func.DC_QAOA(G_test, p=p, t=5, k=5)
    plot_histogram(dc, figsize=(10, 8), title="DC-QAOA, p="+str(p), bar_labels=False)
    