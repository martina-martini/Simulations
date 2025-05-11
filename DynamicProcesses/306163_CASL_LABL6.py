############################## IMPORT ##############################
from queue import PriorityQueue
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, probplot, chisquare, binom, t, chi2
import pandas as pd
############################## CLASS NODE ##############################
# a node is characterized by the index, the vote (i.e., the state variable) and neighbors (the nodes connected with it)
class Node:
    def __init__(self, idx):
        self.idx = idx
        self.vote = -1
        self.neighbors = []

    def __str__(self):
        return f'{(self.idx, self.vote)}'

######################### GRAPHS GENERATION #######################
# the following function creates n instances of the class Node with initial vote -1
def create_nodes(n):
    nodes = []
    for node_index in range(n):
        # vote = random.choice([1, -1])
        vote = -1
        node = Node(node_index)
        node.vote = vote
        nodes.append(node)
    return nodes


# the following function creates a graph with n nodes, probability of being linked p and probability of voting +1 equal to p1
# the list of the initial nodes (non-linked yet) is provided as parameter
def create_graph(n, p, nodes): # G(n, p) graph generation
    graph = {}
    degrees = {}
    for node in nodes:
        graph[node] = []
        k = np.random.poisson(n * p)  # number of edges for each node -> it is chosen from a poisson distribution
        # (the explanation is given in the report)
        degrees[node.idx] = k  # number of degrees = number of links for each node
        other_nodes = [x for i, x in enumerate(nodes) if x.idx != node.idx]
        # chose the nodes to be linked with from the list of the other nodes excluded the one under consideration
        for _ in range(k):
            new_node = random.choice(other_nodes)
            graph[node].append(new_node)
            node.neighbors.append(new_node)
            # k nodes are chosen as neighbors and added to the list of neighbors of the node under consideration

    # match the links: if the node i is present also in the list of neighbors of another node j, then add j as
    # neighbor of i
    for node in nodes:
        for tupla, lista in graph.items():
            if node in lista:
                graph[node].append(tupla)

    # remove duplicate nodes
    for node in nodes:
        graph[node] = list(set(graph[node]))

    return graph, degrees


def generate_grid_graph_2(n, nodes): # 2-dimensional grid generation
    grid_size = int(n ** (1 / 2))
    graph = {}
    degrees = {}

    for node in nodes:
        graph[node] = []

        if node.idx % grid_size != 0:
            # if the node is not the first in a row, connect it to the neighbor on its left
            neighbor = Node(node.idx - 1)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)
        if node.idx // grid_size != 0:
            # if the node is not the first in a column, connect it to the neighbor on the top
            neighbor = Node(node.idx - grid_size)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)
        if node.idx % grid_size != grid_size - 1:
            # if the node is not the last of the row, connect it with the neighbor on the right
            neighbor = Node(node.idx + 1)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)
        if node.idx // grid_size != grid_size - 1:
            # if the node is not the last of the column connect it with the neighbor on the bottom
            neighbor = Node(node.idx + grid_size)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)

        degrees[node] = len(node.neighbors)
        # the degree of a node is 4 if it is central, 3 if it is in one of the sides of the grid or 2 if in the corner

    return graph, degrees


def generate_3d_grid_graph(n, nodes): # 3-dimensional grid generation
    grid_size = int(n ** (1 / 3))
    graph = {}
    degrees = {}

    for node in nodes:
        graph[node] = []
        x, y, z = node.idx % grid_size, (node.idx // grid_size) % grid_size, node.idx // (grid_size ** 2)
        if x > 0:
            # if the node is not in the first column, connect it with the neighbor on the left
            neighbor = Node(node.idx - 1)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)
        if x < grid_size - 1:
            # if the node is not in the last column, connect it with the neighbor on the right
            neighbor = Node(node.idx + 1)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)
        if y > 0:
            # if the node is not in the first row, connect it with the neighbor on the top
            neighbor = Node(node.idx - grid_size)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)
        if y < grid_size - 1:
            # if the node is not in the last row, connect it with the neighbor on the bottom
            neighbor = Node(node.idx + grid_size)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)
        if z > 0:  # Connect to front neighbor if not in the first layer
            # if the node is not in the first layer, connect it with the neighbor in front of it
            neighbor = Node(node.idx - grid_size ** 2)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)
        if z < grid_size - 1:
            # if the node is not in the last layer, connect it with the neighbor behind it
            neighbor = Node(node.idx + grid_size ** 2)
            graph[node].append(neighbor)
            node.neighbors.append(neighbor)

        degrees[node] = len(node.neighbors)
        #  the degree of a node is 6 if it is central, 5 if it is in the side, 3 if it is in the corner

    return graph, degrees


def graph_update(graph, node, vote):
    for tupl in graph.keys():
        if tupl.idx == node.idx:
            tupl.vote = vote

    for tupl, lista in graph.items():
        for neighbor in lista:
            if neighbor.idx == node.idx:
                neighbor.vote = vote
# if any node of the graph is changing opinion, change the dictionary representing the graph

########################### FES METHODS ################################
# the fes is introduced as a priority queue (type FIFO):
# an event of the type of 'activation' or 'update' is associate with each node
# they are such as arrival events: the inter-arrival time is following the exponential distribution with rate lambda

def activation(time, fes, graph, lambda_rate, p1, queue):
    # schedule next activation event
    activation_time = random.expovariate(1 / lambda_rate)
    fes.put((time + activation_time, 'activation'))
    # initially all the nodes are in state -1
    initial_nodes = [node for node in graph.keys() if node.vote == -1]
    if initial_nodes:  # if some votes are still == -1
        node = random.choice(initial_nodes)
        value = np.random.choice([1, -1], p=[p1, 1 - p1]) # each node has the p1 probability to vote +1
        node.vote = value
        graph_update(graph, node, value)
        # schedule update event for the neighbors of the activated node
        neighbors = node.neighbors
        if neighbors:
            for neighbor in neighbors:
                update_time = random.expovariate(1 / lambda_rate)
                queue.append(neighbor)
                fes.put((time + update_time, 'update'))


def update(time, fes, graph, queue):
    # select a node from the queue and update the state based on the voter model dynamics
    node = queue.pop(0)
    neighbors = node.neighbors
    # voter model dynamics: retrieve the state of a random neighbor and copy the state of the neighbor
    # for each triggered event, look at the neighbors of the node: each vote of each neighbor has the same uniform
    # probability to be selected -> in fact, the node selects one of its neighbors and copies its opinion
    # the same works for both the votes +1 or -1
    if neighbors:
        selected_neighbor = random.choice(neighbors)
        if selected_neighbor.vote == 1:
            node.vote = 1
            graph_update(graph, node, 1)
        else:
            node.vote = -1
            graph_update(graph, node, -1)
    # progressively, the system takes the neighbors and change the state variables, since the system goal is to reach
    # the consensus (i.e., all the linked nodes of a graph have the same state variable)


# function to plot Quantile-Quantile plot for comparing the analytical and the empirical degrees distribution
def qq_plot(data, distribution, params):
    plt.figure(figsize=(12, 3))
    probplot(data, dist=distribution, sparams=params, plot=plt)
    plt.title('Q-Q Plot')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered Values')
    plt.show()


################################ SIMULATION DYNAMICS #####################
def simulate_(p1, n, dimension, num_of_runs, end_time):
    lista_of_1_nodes = []
    perc_of_1 = 0.0
    prob = 0.0
    count_consensus = 0

    for run in range(num_of_runs):
        # print(f'run {run+1}')
        # print(f'WORKING WITH LAMBDA = {lambda_rate}, end time = {end_time}')

        p = 10 / n  # probability of being linked

        nodes = create_nodes(n)  # the nodes are created -> all nodes has vote -1
        if dimension == 1:
            graph, degrees = create_graph(n, p, nodes)  # the G(n, p) graph is created
        elif dimension == 2:
            graph, degrees = generate_grid_graph_2(n, nodes) # the 2-dimensional grid (Z^2 lattice) is created
        elif dimension == 3:
            graph, degrees = generate_3d_grid_graph(n, nodes) # the 3-dimensional grid (Z^3 lattice) is created
        else:
            assert dimension > 3 or dimension < 1, "Invalid dimension: insert 1 for G(n, p) graph, 2 for Z^2 lattice and 3 for Z^3 lattice"

        # FES is initialized
        time = 0
        queue = []
        fes = PriorityQueue()
        fes.put((0, 'activation'))

        # print(f'STARTING SIMULATION WITH n = {n}, p = {p}, p1 = {p1}')
        all_nodes_vote_1 = False # flag variable to check if all the nodes are voting +1
        while time < end_time:
            time, event_type = fes.get()
            if event_type == 'activation':
                activation(time, fes, graph, lambda_rate, p1, queue)
                # if all nodes reached state 1 -> consensus reached
                if all(node.vote == 1 for node in graph.keys()):
                    all_nodes_vote_1 = True
                    break  # stop simulation
            elif event_type == 'update':
                update(time, fes, graph, queue)

        if all_nodes_vote_1:
            # print(f"All nodes reached state 1 at time {time} -> {(sum(node_data.vote == 1 for node_data in graph.keys()))/len(graph)*100}% of nodes +1")
            count_consensus += 1 # count how many times in the number of runs the graph reaches the consensus
            times.append(time) # store the time that the graph takes to reach consensus
            lista_of_1_nodes.append((sum(node_data.vote == 1 for node_data in graph.keys())) / len(graph) * 100)

        else: # if the system does not reach the consensus before the chosen end_time, count how many nodes are voting +1 until now
            print(
                f"At least one simulation completed (end_time = {end_time} reached) without all nodes reaching state 1 -> {(sum(node_data.vote == 1 for node_data in graph.keys())) / len(graph) * 100}% of nodes +1")
            lista_of_1_nodes.append((sum(node_data.vote == 1 for node_data in graph.keys())) / len(graph) * 100)

    prob = count_consensus / num_of_runs * 100 # count the probability over the chosen number of runs to reach consensus
    if lista_of_1_nodes:
        perc_of_1 = np.mean(lista_of_1_nodes)
        # print(f'the average percentage of +1 nodes considering {num_of_runs} runs for 1 simulation is: {perc_of_1}%')
        # print(f'probability of reaching +1-consensus: {prob}%')
    return degrees, lista_of_1_nodes, times, prob

###################### COINFIDENCE INTERVALS COMPUTATION ############
def confidence_int(data, conf):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / (len(data) ** (1 / 2))  # this is the standard error

    interval = t.interval(confidence=conf,  # confidence level
                          df=max(len(data) - 1, 1),  # degree of freedom
                          loc=mean,  # center of the distribution
                          scale=se  # spread of the distribution
                          # (we use the standard error as we use the extimate of the mean)
                          )

    MOE = interval[1] - interval[0]  # this is the margin of error
    re = (MOE / (2 * abs(mean)))  # this is the relative error
    acc = 1 - re  # this is the accuracy
    return interval, acc


##################################### MAIN LOOP ###############################################
# disclaimer: the simulation may take at most 30 minutes to completely finish
# (3 minutes for each p1 if dim = 1 -> 15 minutes for G(n, p) graph and 1000 nodes
# the remaining 15 minutes to run the simulations of Z^2 lattice and Z^3 lattice for 100 nodes and 1000 nodes

# to save time, the simulation can be run with only 100 nodes: it would take 1 minute and it provides equally good results to evaluate

# introducing a dataframe for a better visualization of the results
columns = ["p1", "dimension", "n", "p", "mean avg time (3 batches)", "probability", "accuracy", "chi^2", "p-value"]
main_df = pd.DataFrame(columns=columns)

##### INPUTS #####
p1s = {0.51, 0.55, 0.6, 0.7}  # probability of having vote = +1

seed = 42
random.seed(seed)  # introduced for reproducibility purposes
probs = []

cl = 0.90 # confidence level
acceptance = 0.94 # level of acceptance
batches = 3 # number of batches for computing the confidence intervals
count = 0
lambda_rate = random.choice([1, 1.2, 2]) # the arrival rate for scheduling the inter-arrival times is chosen randomly between this list
available_dimensions = [1, 2, 3] # dimension = 1 is referring t the G(n, p) graph, dimension = 2 to the Z^2 lattice and dimension = 3 to the Z^3 lattice

for dim in available_dimensions:
    print(f'\n\n::::::::::::::::: dimension = {dim} :::::::::::::::::')

    if dim == 1:
        n = 1_000 # number of nodes
        num_of_runs = 5
        end_time = 3_000
        for p1 in p1s:
            print(f'---------------------p1 = {p1}-----------------')
            times = []
            acc_times = 0.0 # accuracy
            avg_times = []

            while acc_times < acceptance:
                for batch in range(batches):
                    # print(f'batch = {batch}')

                    degrees, nodes_1, times, prob = simulate_(p1, n, dim, num_of_runs, end_time)
                    p = 10 / n # probability of link two nodes
                    int1, acc_times = confidence_int(times, cl)

                    if acc_times > acceptance:
                        print(f'\nthe accuracy is: {acc_times}')

                        print(f'the average time to reach consensus for p1 = {p1} is {np.mean(times)}')
                        avg_times.append(np.mean(times))
                        print(f'the probability of reaching the consensus = {prob}%')

                        print(f'\t\t\t\tBREAK: accuracy {acceptance} reached\t\t\t')
                        break

            print(f'p1 = {p1}, dim = {dim}, n = {n}, p = {p}')
            ##### OUTPUT: Q-Q plot #####
            qq_plot(list(degrees.values()), 'binom', (n - 1, p))
            # degree distributions for G(n,p)
            empirical_counts, _ = np.histogram(list(degrees.values()),
                                               bins=np.arange(max(list(degrees.values())) + 2) - 0.5, density=True)
            k_values = np.arange(max(list(degrees.values())) + 1)
            binomial_pmf = binom.pmf(k_values, n - 1, p)
            expected_counts = binomial_pmf * len(list(degrees.values()))

            ##### OUTPUT 2: Chi-square test for goodness of fit #####
            chi2_stat = np.sum((empirical_counts - expected_counts) ** 2 / expected_counts)
            p_value = chi2.sf(chi2_stat, len(list(degrees.values())) - 1)

            # print(f"\nChi-square statistic: {chi2_stat}")
            alpha = 0.05  # -> significance level = 5%
            test_passed = p_value > alpha
            # print(f'alpha = {alpha}')
            if test_passed:
                print(f"P-value: {p_value}: Chi-square test passed.")
            else:
                print(f"P-value: {p_value}: Chi-square test failed.")
            new_record = {
                "p1": p1,
                "dimension": dim,
                "n": n,
                "p": p,
                "mean avg time (3 batches)": np.mean(avg_times),
                "probability": prob,
                "accuracy": acc_times,
                "chi^2": chi2_stat,
                "p-value": p_value
            }
            # insert the reconrd int he dataframe for a better visualization
            main_df.loc[len(main_df)] = new_record

            print('\n\n')


    else: # dim = 2 and dim = 3
          # -> the loops are the same as before but the qq-plot and the chi^2 statistics are avoided
        ns = {100, 1_000}  # -> let's compare 2 or 3 values of n
        num_of_runs = 5
        end_time = 5_000
        for n in ns:
            print(f'---------------------n = {n}-----------------')

            for p1 in p1s:
                print(f'---------------------p1 = {p1}-----------------')
                avg_times = []
                times = []
                acc_times = 0.0

                while acc_times < acceptance:
                    for batch in range(batches):
                        # print(f'batch = {batch}')

                        degrees, nodes_1, times, prob = simulate_(p1, n, dim, num_of_runs, end_time)
                        p = 10 / n
                        int1, acc_times = confidence_int(times, cl)

                        if acc_times > acceptance:
                            print(f'the accuracy is: {acc_times}')
                            avg_times.append(np.mean(times))
                            print(f'the average time to reach consensus for p1 = {p1} is {np.mean(times)}')
                            print(f'the probability of reaching the consensus = {prob}%')
                            if len(times) == 0:
                                print('consensus not reached')
                                break

                            print(f'\t\t\t\tBREAK: accuracy {acceptance} reached\t\t\t')
                            break

                    new_record = {
                        "p1": p1,
                        "dimension": dim,
                        "n": n,
                        "p": p,
                        "mean avg time (3 batches)": np.mean(avg_times),
                        "probability": prob,
                        "accuracy": acc_times,
                        "chi^2": '-',
                        "p-value": '-'
                    }
                    main_df.loc[len(main_df)] = new_record
        print('\n')
    print('\n')

##### DISPLAY THE RESULTS #####
print(main_df)