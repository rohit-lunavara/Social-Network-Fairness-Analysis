from random import uniform, seed
import matplotlib.pyplot as plt
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

# Obtaining a list of all regions

region_list = []
for i in G.nodes :
    if G.nodes[i]['region'] not in region_list :
        region_list.append(G.nodes[i]['region'])

#print(region_list)

# Obtaining total number of people in each region

region_total = {}
for i in G.nodes :
    region_total[G.nodes[i]['region']] = region_total.get(G.nodes[i]['region'], 0) + 1

#print(region_total.items())

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

# Task 1.1 - Independent Cascade function
def independent_cascade(G, seeds, get_region_spread = True, ic_iter = 500) :
    """
    Input : Graph, seeds and number of Monte-Carlo simulations
    Output : Average number of nodes influenced by the seed nodes
    """

    # Loop over the number of iterations 
    spread, region_spread = [], []
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


        # Proportion of people from each region
        if get_region_spread :
            region_iter = {}
            for i in region_list :
                region_iter[i] = 0
            for i in activated :
                region_iter[G.nodes[i]['region']] = region_iter.get(G.nodes[i]['region'], 0) + 1
            for k, v in region_iter.copy().items() :
                region_iter[k] = round(v / region_total[k], 4)
            #print(region_iter, end = '\n\n')
            region_spread.append(region_iter)

        spread.append(len(activated))

    # Averaging over all iterations
    if get_region_spread :
        region_mean = {}
        for i in region_list :
            region_mean[i] = 0
        for i in range(len(region_spread)) :
            for region in region_list :
                region_mean[region] += (region_spread[i][region])
        for i in region_mean :
            region_mean[i] = round(region_mean[i] / ic_iter, 8)
        #print(region_mean)
        return np.mean(spread), region_mean
    else :
        return np.mean(spread)

#S = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#mean_spread = independent_cascade(G, S)
#print(mean_spread)

def greedy(G, k, ic_iter = 10) :
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

            # Get the spread
            s = independent_cascade(G, seeds + [j], get_region_spread = True, ic_iter = ic_iter)

            # Update the winning node and spread so far
            if s > best_spread : best_spread, node = s, j

        # Add the selected node to the seed set
        seeds.append(node)

        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(round(time.time() - start_time, 2))

    return (seeds, spread, timelapse)

# Task 1.2 - Compute Greedy algorithm output

#greedy_output = greedy(G, 25)

#print('\nSeeds :', greedy_output[0], '\n')
#print('\nSpread :', greedy_output[1], '\n')
#print('\nTimelapse :', greedy_output[2], '\n')
#print(greedy_output)

# Task 1.3 - Compute proportion of people receiving information for each region averaged across 500 simulations.

#seeds = [19, 264, 17, 13, 265, 282, 263, 460, 18, 296, 26, 240, 155, 136, 266, 82, 30, 409, 12, 268, 424, 74, 313, 29, 399]
#mean_spread, region_spread = independent_cascade(G, seeds)
#print(mean_spread)
#plt.bar(region_spread.keys(), region_spread.values())
#plt.xticks(region_list, region_list, rotation = 90)
#plt.margins(0.2)
#plt.gcf().subplots_adjust(bottom=0.15)
#plt.tight_layout()
#plt.show()

#for k,v in region_spread.items() :
#    print(v, k)

# Task 2.1 - Fairness

def max_fair_alloc(G, k, precision = 2, ic_iter = 10) :
    """
    Input : Graph, number of seeds, precision of variance and number of Monte-Carlo simulations
    Output : Maximum fairness seed set, resulting spread, time for each iteration
    """

    seeds, spread, timelapse, cumulative_variance, start_time = [], [], [], [], time.time()

    # Find k nodes with largest marginal gain
    for _ in range(k) :

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        records = []
        for j in (set(G.nodes) - set(seeds)) :

            # Get the spread
            s, region_spread = independent_cascade(G, seeds + [j], get_region_spread = True, ic_iter = ic_iter)

            # Store records for later
            variance = round(np.var(list(region_spread.values())), precision)
            records.append(tuple((seeds + [j], s, variance)))

        # Sort the records to minimize the variance and maximize spread

        # First, sort to minimize variance
        records.sort(key = lambda x : x[2])

        # Second, reverse sort only those records which have equal variance
        minval = records[0][2]
        selected_records, minindex = [], 0
        for i in range(len(records)) :
            if records[i][2] == minval : minindex = i
        selected_records = records[0 : minindex + 1]
        selected_records.sort(key = lambda x : x[1], reverse = True)

        # Extract the best record
        best_record = selected_records.pop(0)
        print(best_record)
        print('Seeds :', best_record[0])
        node = best_record[0].pop(-1)
        print('Added node :', node)
        fair_spread = best_record[1]
        print('Spread with added node :', fair_spread)
        node_var = best_record[2]
        print('Variance with added node :', node_var, end = '\n\n')

        # Add the selected node to the seed set
        seeds.append(node)

        # Add estimated spread, elapsed time and variance upon adding each node
        spread.append(fair_spread)
        timelapse.append(round(time.time() - start_time, 2))
        cumulative_variance.append(node_var)

    return (seeds, spread, timelapse, cumulative_variance)

# Task 2.1 - Compute Fairness algorithm output

fairness_output = max_fair_alloc(G, 25, precision = 4, ic_iter = 500)

print('\nSeeds :', fairness_output[0], '\n')
print('\nSpread :', fairness_output[1], '\n')
print('\nTimelapse :', fairness_output[2], '\n')
print('\nVariance :', fairness_output[3], '\n')
print(fairness_output)

seeds = fairness_output[0]
mean_spread, region_spread = independent_cascade(G, seeds, ic_iter = 500)
print(mean_spread)
plt.bar(region_spread.keys(), region_spread.values())
plt.xticks(region_list, region_list, rotation = 90)
plt.margins(0.2)
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.show()