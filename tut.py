import numpy
import pickle
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([(0, 3), (1, 2), (2, 1), (1, 3)])
G.nodes[0]['region'] = 'b'
G.nodes[1]['region'] = 'a'
G.nodes[2]['region'] = 'b'
G.nodes[3]['region'] = 'a'

for i in range(G.number_of_nodes()) :
    for j in G.adj[i] :
        if G.nodes[i]['region'] == G.nodes[j]['region'] :
            G.edges[i, j]['infl_prob'] = 0.2
        else :
            G.edges[i, j]['infl_prob'] = 0.1

#print(G.edges)
#for (u, v, prob) in G.edges.data('infl_prob') :
#    print(u, v, prob)

#for node in G.nodes :
#    print(len(G.successors(node)))
#    for n in G.successors(node) :
#        print(node, n)
G = pickle.load(open('graph.pickle', 'rb'))

print(len(G.edges))