from random import uniform, seed
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import networkx as nx

G = pickle.load(open('graph.pickle', 'rb'))

# Setting the influence probability for connected individuals from the same regions to be 0.2 and 0.1 otherwise.
for i in G.nodes :
    for j in G.successors(i) :
        if G.nodes[i]['region'] == G.nodes[j]['region'] :
            G.edges[i, j]['infl_prob'] = 0.2
        else :
            G.edges[i, j]['infl_prob'] = 0.1

# Obtaining a list of all regions

region_list = []
for i in G.nodes :
    if G.nodes[i]['region'] not in region_list :
        region_list.append(G.nodes[i]['region'])

# Obtaining total number of people in each region

region_total = {}
for i in G.nodes :
    region_total[G.nodes[i]['region']] = region_total.get(G.nodes[i]['region'], 0) + 1

# Task 1.1 - Implement Independent Cascade 

def independent_cascade(G, seeds, get_region_spread = True, ic_iter = 500) :
    """
    Input : 
        G - Graph upon which we perform the Independent Cascade
            NetworkX Directed Graph object
        seeds - The seeds which we use for Independent Cascade
            List
        get_region_spread - True by default, set to False if proportion of nodes influenced per region is not required
            Boolean
        ic_iter - Number of simulations, 500 by default, provide a higher value for greater accuracy
            Integer
    Output : 
        Average number of nodes influenced by the seed nodes and 
        Proportion of nodes influenced per region
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
                    success = np.random.uniform(0, 1) < G.edges[node, node_neighbor]['infl_prob']
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
        return np.mean(spread), region_mean
    else :
        return np.mean(spread)

# Executing Independent Cascade function
#seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#mean_spread = independent_cascade(G, seeds, get_region_spread = False)
#print(mean_spread)

# Task 1.2 - Implement Greedy algorithm

def greedy(G, k, ic_iter = 500) :
    """
    Input : 
        G - Graph upon which we perform the Greedy algorithm
            NetworkX Directed Graph object
        k - The number of seeds required
            List
        ic_iter - Number of simulations, 500 by default, provide a higher value for greater accuracy
            Integer
    Output : 
        Optimal seed set
        Average number of nodes influenced by the seed set for each iteration
        Time for each iteration
    Reference :
        David Kempe, Jon Kleinberg, and Éva Tardos. 2003. Maximizing the spread of influence through a social network. In Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '03). ACM, New York, NY, USA, 137-146. DOI=http://dx.doi.org/10.1145/956750.956769
    Link :
        https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf
    """

    seeds, spread, timelapse, start_time = [], [], [], time.time()

    # Find k nodes with largest marginal gain
    for _ in range(k) :

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in (set(G.nodes) - set(seeds)) :

            # Get the spread
            s = independent_cascade(G, seeds + [j], get_region_spread = False, ic_iter = ic_iter)

            # Update the winning node and spread so far
            if s > best_spread : best_spread, node = s, j

        # Add the selected node to the seed set
        seeds.append(node)

        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(round(time.time() - start_time, 2))

    return (seeds, spread, timelapse)

# Executing the Greedy algorithm

#greedy_output = greedy(G, 25)
#print('\nSeeds :', greedy_output[0], '\n')
#print('\nSpread :', greedy_output[1], '\n')
#print('\nTimelapse :', greedy_output[2], '\n')
#print(greedy_output)

# Task 1.3 - Compute proportion of people receiving information for each region averaged across 500 simulations for your best seed set.

#seeds = [19, 264, 17, 13, 265, 282, 263, 460, 18, 296, 26, 240, 155, 136, 266, 82, 30, 409, 12, 268, 424, 74, 313, 29, 399]
#mean_spread, region_spread = independent_cascade(G, seeds)

# View in command line
#print(mean_spread)
#print('Regions\t| Percentage of people in that region receiving the information', end = '\n\n')
#for k,v in region_spread.items() : 
#   print(k, '|', (v*100))

# View as a bar graph
#plt.bar(region_spread.keys(), region_spread.values())
#plt.title('Greedy Algorithm Results')
#plt.xlabel('Regions')
#plt.ylabel('Proportion of Spread in each Region')
#plt.xticks(region_list, region_list, rotation = 90)
#plt.margins(0.2)
#plt.gcf().subplots_adjust(bottom = 0.15)
#plt.tight_layout()
#plt.show()

# Task 2.1 - Implement Maximum Fairness algorithm

def max_fair(G, k, precision = 4, ic_iter = 500) :
    """
    Input : 
        G - Graph upon which we perform the Greedy algorithm
            NetworkX Directed Graph object
        k - The number of seeds required
            List
        precision - The precision with which we calculate variance between proportion of people receiving information for each region, 4 by default (i.e. Lowest value for variance is 0.0001), provide a higher value for greater control
            Integer
        ic_iter - Number of simulations, 500 by default, provide a higher value for greater accuracy
            Integer
    Output : 
        Maximum fairness seed set
        Average number of nodes influenced by the seed set for each iteration
        Time for each iteration
    Reference :
        Tsang, A., Wilder, B., Rice, E., Tambe, M., and Zick, Y. 2019. Group-fairness in influence maximization. arXiv preprint arXiv:1903.00967.
    Link :
        https://arxiv.org/pdf/1903.00967.pdf
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
        node = best_record[0].pop(-1)
        fair_spread = best_record[1]
        node_var = best_record[2]

        # Add the selected node to the seed set
        seeds.append(node)

        # Add estimated spread, elapsed time and variance upon adding each node
        spread.append(fair_spread)
        timelapse.append(round(time.time() - start_time, 2))
        cumulative_variance.append(node_var)

    return (seeds, spread, timelapse, cumulative_variance)

# Task 2.2 - Executing the Maximum Fairness algorithm

#fairness_output = max_fair(G, 25)
#print('\nSeeds :', fairness_output[0], '\n')
#print('\nSpread :', fairness_output[1], '\n')
#print('\nTimelapse :', fairness_output[2], '\n')
#print('\nVariance :', fairness_output[3], '\n')
#print(fairness_output)

# Compute proportion of people receiving information for each region averaged across 500 simulations for your best seed set.

#seeds = [111, 320, 56, 447, 458, 74, 314, 99, 31, 344, 210, 47, 334, 179, 430, 366, 184, 304, 30, 350, 114, 446, 122, 373, 35]
#mean_spread, region_spread = independent_cascade(G, seeds, ic_iter = 500)

# View in command line
#print(mean_spread)
#print('Regions\t| Percentage of people in that region receiving the information', end = '\n\n')
#for k,v in region_spread.items() : 
#    print(k, '|', (v*100))

# View as a bar graph
#plt.bar(region_spread.keys(), region_spread.values())
#plt.title('Fairness Algorithm Results')
#plt.xlabel('Regions')
#plt.ylabel('Proportion of Spread in each Region')
#plt.xticks(region_list, region_list, rotation = 90)
#plt.margins(0.2)
#plt.gcf().subplots_adjust(bottom=0.15)
#plt.tight_layout()
#plt.show()