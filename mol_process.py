
from networkx.readwrite.json_graph.adjacency import adjacency_data
import rdkit
import rdkit.Chem as Chem
import torch_geometric.data as Data
import torch
import networkx as nx
import pysmiles
import matplotlib.pyplot as plt

# def mol2heteroGraph(mol, mps, use_mps):
#     ## graph = pysmiles.read_smiles(smi)
#     ## pos_nodes = nx.spring_layout(graph)
#     ## nx.draw(graph, pos_nodes, with_labels=True)
#     ## plt.savefig("nx-mol.png")    
    
#     atomType_enc = {6:'C', 7:'N', 8:'O', 9:'F', 16:'S', 17:'Cl', 35:'Br', 53:'I'}
#     bonds = mol.GetBonds()
#     meta_paths = []
#     meta_paths_dic = {}
#     bondType_enc = {Chem.rdchem.BondType.SINGLE:'SINGLE',
#                     Chem.rdchem.BondType.DOUBLE:'DOUBLE',
#                     Chem.rdchem.BondType.TRIPLE:'TRIPLE',
#                     Chem.rdchem.BondType.AROMATIC:'AROMATIC'}

#     atomIdx_dic = {}
#     atom_feats = {'C':{},'N':{},'O':{}}


smi = 'CN1CC(O)(C=O)C1=O' #'OC12CC(C1)C1CC2O1'
mol = Chem.MolFromSmiles(smi)
nxgraph = pysmiles.read_smiles(smiles=smi)
#, approach-1
myadj = nxgraph.adjacency()
myadj_list = {nodeIdx:neighbs for nodeIdx,neighbs in myadj}

#, approach-2
adj_graph = adjacency_data(nxgraph)
nodes = adj_graph['nodes']
adj_edges = adj_graph['adjacency']
# print("edges in method-1:\n", myadj_list)
# print("edges in method-2:\n", adj_edges)
# print("nodes:\n", nodes)

# pos_nodes = nx.spring_layout(nxgraph)
# nx.draw(nxgraph, pos_nodes, with_labels=True)
# plt.savefig("nx-mol.png")    

    
atomType_enc = {6:'C', 7:'N', 8:'O', 9:'F', 16:'S', 17:'Cl', 35:'Br', 53:'I'}
bondType_enc = {Chem.rdchem.BondType.SINGLE:'SINGLE',
                    Chem.rdchem.BondType.DOUBLE:'DOUBLE',
                    Chem.rdchem.BondType.TRIPLE:'TRIPLE',
                    Chem.rdchem.BondType.AROMATIC:'AROMATIC'}
colorDict = {
    "C": "#18A8B4",
    "N": "#F898B4",
    "O": "#F8F8B4"
}


# visualization, given a graph G
positions = nx.spring_layout(nxgraph) # 建立布局 
node_labels = nx.get_node_attributes(nxgraph, name="element") #  {0: 'C', 1: 'N', 2: 'C', 3: 'C', 4: 'O', 5: 'C', 6: 'O', 7: 'C', 8: 'O'}
node_colors = [colorDict[node_labels[key]] for key in node_labels]

edge_labels = nx.get_edge_attributes(nxgraph, name="order") # {(0, 1): 1, (1, 2): 1, (1, 7): 1, (2, 3): 1, (3, 4): 1, (3, 5): 1, (3, 7): 1, (5, 6): 2, (7, 8): 2}

nx.draw_networkx_edge_labels(nxgraph, positions, edge_labels=edge_labels, 
                                    font_size=10, font_color="k")

nx.draw(nxgraph, positions, labels=node_labels, with_labels = True,width=3.0, font_size=20)#font_color="ed"
nx.draw_networkx_nodes(nxgraph, positions, nodelist=node_labels, node_color=node_colors, 
            node_size=600,label=node_labels)


adj_dense = nx.adjacency_matrix(nxgraph, weight="order").todense()


plt.savefig("mole_nx.png")
plt.show()

from torch_geometric.utils import dense_to_sparse
#, convert: dense adj-matrix -> sparse adj-matrix(edge_Indices, edge_attributes)
adj_sparse = dense_to_sparse(torch.tensor(adj_dense))
print("adj_matrix:\n",adj_sparse)
'''
adj_dense:
 [[0 1 0 0 0 0 0 0 0]
 [1 0 1 0 0 0 0 1 0]
 [0 1 0 1 0 0 0 0 0]
 [0 0 1 0 1 1 0 1 0]
 [0 0 0 1 0 0 0 0 0]
 [0 0 0 1 0 0 2 0 0]
 [0 0 0 0 0 2 0 0 0]
 [0 1 0 1 0 0 0 0 2]
 [0 0 0 0 0 0 0 2 0]]
adj_matrix:
(tensor([[0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7, 8], #, edge-index(src-nodes)
          [1, 0, 2, 7, 1, 3, 2, 4, 5, 7, 3, 3, 6, 5, 1, 3, 8, 7] #, edge-index(target-nodes)
          ]) 
 ,tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2]) #, edge-attr
)
'''


plt.savefig("mole_nx.png")
plt.show()
