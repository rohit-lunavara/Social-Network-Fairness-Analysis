from random import uniform, seed
import numpy as np
import time
import pickle
import networkx as nx

G = pickle.load(open('graph.pickle', 'rb'))

#print(G.edges)

# Setting the influence probability for connected individuals from the same regions to be 0.2 and 0.1 otherwise.
for i in G.nodes :
    for j in G.adj[i] :
        if G.nodes[i]['region'] == G.nodes[j]['region'] :
            G.edges[i, j]['infl_prob'] = 0.2
        else :
            G.edges[i, j]['infl_prob'] = 0.1

# Same regions
#print(G.nodes[0])
#print(G.nodes[1])
#print(G.edges[0, 1])

# Different regions
#print(G.nodes[481])
#print(G.nodes[31])
#print(G.edges[481, 31])

# Printing edges with their probability
#for (u, v, prob) in G.edges.data('infl_prob') :
#   print(u, v, prob)

# Independent Cascade function
def independent_cascade(G, seeds, ic_iter = 500) :
    """
    Input : Graph, seeds and number of Monte-Carlo simulations
    Output : Average number of nodes influenced by the seed nodes
    """

    # Loop over the number of iterations 
    spread = []
    for i in range(ic_iter) :

        # Simulate propagation process
        new_active, activated = seeds[:], seeds[:]

        while new_active :
            
            # For each newly activated node, find its neighbors that become activated
            new_ones = []
            for node in new_active :

                # Determine those neighbors that become infected
                np.random.seed(i)
                for node_neighbor in G.successors(node) :
                    #print(node, node_neighbor, G.edges[node, node_neighbor]['infl_prob'])
                    success = np.random.uniform(0, 1) < G.edges[node, node_neighbor]['infl_prob']
                    #print(success)
                    if success : new_ones.append(node_neighbor)

                new_active = list(set(new_ones) - set(activated))

                # Add newly activated nodes to the set of activated nodes
                activated += new_active

        spread.append(len(activated))

    return np.mean(spread)

#S = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#mean_spread = independent_cascade(G, S)
#print(mean_spread)

def greedy(G, k, g_iter = 1) :
    """
    Input : Graph, number of seeds and number of Monte-Carlo simulations
    Output : Optimal seed set, resulting spread, time for each iteration
    """

    seeds, spread, timelapse, start_time = [], [], [], time.time()

    # Find k nodes with largest marginal gain
    for _ in range(k) :

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in (set(G.nodes) - set(seeds)) :

            # Processing message
            #print('Processing...')

            # Get the spread
            s = independent_cascade(G, seeds + [j])

            # Update the winning node and spread so far
            if s > best_spread : best_spread, node = s, j

        # Add the selected node to the seed set
        seeds.append(node)

        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(round(time.time() - start_time, 2))

    return (seeds, spread, timelapse)

greedy_output = greedy(G, 25)

print('\nSeeds :', greedy_output[0], '\n')
print('\nSpread :', greedy_output[1], '\n')
print('\nTimelapse :', greedy_output[2], '\n')
#print(greedy_output)

def max_fair_alloc(G, k, mfa_iter = 1) :
    """
    Input : Graph, number of seeds and number of Monte-Carlo simulations
    Output : Maximum fairness seed set, resulting spread, time for each iteration
    """

    pass