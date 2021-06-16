import numpy as np
from qaoa_tn import Relaxation, Relaxation_RoundUp, Exact
from qaoa_dc import QAOA_DC, NearNode_DC_QAOA
from qaoa_dc import LGP, QSR2, get_bit, get_dict, _getCost
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import sparse
import networkx as nx
import community
from tqdm import tqdm

def NearNode_Main(community, As_nx, G, *args,**kwargs):
    assert type(community) is dict, "community should be input as dictionary."
    
    #Prepare graph from input community
    keys = np.array(list(community.keys()))
    vals = np.array(list(community.values()))
    #asset_idx = keys[np.where(vals==np.random.randint(max(vals)))[0]]
    #g = As_nx.subgraph(asset_idx)

    subgroups_list = []
    #Loop though each subgroups to get new subgroup including nearest neigher.
    #Original subgroup.
    for subgroup in np.unique(vals):
        asset_idx = keys[np.where(vals==subgroup)[0]]
        g = As_nx.subgraph(asset_idx)
    #Check each assets for any external edge.
    #Add an indice of external asset into asset_idx and recreate new subgroup.
        edges_ext = [] #Need this line to prevent memory bug.
        for asset in g.nodes:
            if len(G.edges(asset)) != len(g.edges(asset)):
                edges_ext = [int(v) for (u,v) in list(G.edges(asset)) if (u,v) not in list(g.edges(asset)) ]
            asset_idx = np.append(asset_idx, edges_ext)
        asset_idx = np.unique(asset_idx).astype('int') #Ensure that no duplicated asset index.
        subgroups_list.append(asset_idx)

    for idx_list in subgroups_list:
        try:
            assert len(idx_list) < 8, "subgroup size + shared nodes > 8"
        except AssertionError:
            return None
    
    numCandidates = 50
    cycles = 1
    cnt_result = NearNode_DC_QAOA(subgroups_list, Qs, As, cycles=cycles, numCandidates=numCandidates)

    group_final = subgroups_list[0]
    cnt_final = cnt_result[0]

    #Construct state using results of each subgroups.
    while len(subgroups_list) > 1:
        loop_idx = 1
        while loop_idx < len(subgroups_list):
            if len(np.intersect1d(group_final, subgroups_list[loop_idx]) ) != 0:            
                cnt_final = QSR2(group_final, subgroups_list[loop_idx], cnt_final, cnt_result[loop_idx], numCandidates=numCandidates)
                group_final = np.unique(list(group_final) + list(subgroups_list[loop_idx]))
                #Remove subgroup that already be reconstructed.
                del subgroups_list[loop_idx], cnt_result[loop_idx]
                break
            #Check the next subgroup.
            loop_idx += 1

    M = 7 #Num of actions
    action_idx = np.arange(M)
    Q = np.array(Qs[G.nodes][:,action_idx].todense()) / 8000
    A = np.array(As[G.nodes][:,G.nodes].todense())
    e = 1
    cost_dc = []
    #print("20 states with hightest probability. \n")
    for state, prob_den in list(cnt_final.items())[0:20]:
        #prob = prob_den*prob_den.conj()/len(G)
        cost = _getCost(state, Q, A, e)
        #print("%s with cost %.3f"%(state, cost))
        cost_dc.append(cost)
    
    return cost_dc

#10k assets
#As = sparse.load_npz('data/a_matrix.npz')
#Qs = sparse.load_npz('data/q_matrix.npz')

#As_nx = nx.from_scipy_sparse_matrix(As)
#groups = nx.algorithms.community.modularity_max.greedy_modularity_communities(As_nx)

#600 assets
As = np.load('data/A-600.npy')
Qs = sparse.load_npz('data/Q-600-modified.npz')

As_nx = nx.from_numpy_matrix(As)
groups = community.best_partition(As_nx) #600 assets don't need more division.

G = As_nx

costs = NearNode_Main(groups, As_nx, G)
#print(costs)
if costs == None:
    print("Subgroup size > 8.")
else:
    res_dict = {"Asset_idx":idx, "costs":costs}
    res.append(res_dict)

np.save("res600", res, allow_pickle=True)

#For 10k assets
"""
res = []
#Calculate ascending order of subgroup size.
num_subgroups = len(groups)
for idx in tqdm(groups[-num_subgroups: -400]):
    subgroups = community.best_partition(As_nx.subgraph(idx))
    G = As_nx.subgraph(subgroups)
    
    costs = NearNode_Main(subgroups, As_nx, G)
    #print(costs)
    if costs == None:
        print("Subgroup size > 8.")
    else:
        res_dict = {"Asset_idx":idx, "costs":costs}
        res.append(res_dict)

np.save("res600", res, allow_pickle=True)
"""

"""
plt.figure()
plt.scatter(x=range(len(costs)), y=costs, label="Near-Node DC-QAOA.")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Size of the selected group: %d"%len(idx))
plt.xlabel('20 First states from Near-Node DC-QAOA',fontsize=14)
plt.ylabel('Cost',fontsize=14)
plt.legend()
plt.show()
"""