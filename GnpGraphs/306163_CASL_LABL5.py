from queue import PriorityQueue
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, probplot, chisquare
import statsmodels.api as sm

# the fes is introduced as a priority queue (type FIFO): an event of the type of 'update' is associate with each node
# the type 'update' is an arrival event: the inter-arrival time is following the exponential distribution with rate lambda
def initialize_fes(lambda_rate, nodes):
    fes = PriorityQueue()
    for node in nodes:
        arrival = random.expovariate(1 / lambda_rate)
        fes.put((arrival, node, 'update'))
    return fes

# a node is characterized by the index, the vote (i.e., the state variable) and neighbors (the nodes connected with it)
class Node:
    def __init__(self, idx):
        self.idx = idx
        self.vote = 0
        self.neighbors = []

    def __str__(self):
        return f'{(self.idx, self.vote)}'

# the following function creates n instances of the class Node with vote chosen randomly between -1 and +1
def create_nodes(n):
    nodes = []
    for node_index in range(n):
        vote = random.choice([1, -1])
        node = Node(node_index)
        node.vote = vote
        nodes.append(node)
    return nodes

# the following function creates a graph with n nodes, probability of being linked p
# the list of the initial nodes (no link yet) is provided as parameter
def create_graph(n, p, nodes):
    graph = {}
    degrees = {}
    for node in nodes:
        graph[node] = []
        k = np.random.poisson(n * p)  # number of edges for each node -> it is chosen from a poisson distribution
                                      # (the explanation is given in the report)
        degrees[node.idx] = k # number of degrees = number of links for each node
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

#
def simulate(graph, fes, end_time):
    while not fes.empty():
        time, node, event_type = fes.get() # take the first element of the priority queue and process it

        if time > end_time:
            break

        if event_type == 'update':
            current_node = node
            # for each event, take the associated node and update the votes of neighbors: if the current node
            # has vote -1, then all the neighbors' votes (if different) are updated to -1
            # the same works if the cutrrent node has vote = +1
            for neighbor in current_node.neighbors:
                if neighbor.vote != current_node.vote:
                    neighbor.vote = current_node.vote
                    # schedule the next update event for the neighbor
                    next_time = time + random.expovariate(1 / lambda_rate)
                    fes.put((next_time, neighbor, 'update'))

    # progressively, the system takes the neighbors and change the state variables, since the system goal is to reach
    # the consensus (i.e., all the linked nodes of a graph have the same state variable)
    # display the number of nodes without any link
    isolated_nodes = 0
    for tupl, lista in graph.items():
        if not lista:
            isolated_nodes += 1
    print(f'the number of nodes that do not have any link is: {isolated_nodes}')

    # display nodes unreached and so whose vote remains unchanged
    unchanged = []
    for tupl, lista in graph.items():
        for neighbor in lista:
            if tupl.vote != neighbor.vote:
                unchanged.append(neighbor)
    print(f'the number of unreached nodes whose vote remain unchanged is {len(unchanged)}')
    print('\n')


def chi_squared_test(observed, expected):
    # firstly, frequencies are normalized
    observed = observed / np.sum(observed)
    expected = expected / np.sum(expected)
    # then, the p-value is computed and displayed
    _, p_value = chisquare(observed, expected)
    return p_value

# it follows the main function (it can take some minutes to finish compiling: please be patient, it worth it!)
if __name__ == "__main__":
    n = 10_000  # tot number of nodes in a graph
    p = 0.001  # probability of being linked
    lambdas = [0.2, 0.6, 1.2, 2]

    # lambdas = [1.2]
    end_time = 1000
    seed = 1322
    random.seed(seed) # introduced for reproducibility purposes

    for lambda_rate in lambdas:
        print(f'WORKING WITH LAMBDA = {lambda_rate}')
        nodes = create_nodes(n) # the nodes are created
        graph, degrees = create_graph(n, p, nodes) # the graph is created

        # the Future Event Set is initialized
        fes = initialize_fes(lambda_rate, nodes)

        simulate(graph, fes, end_time)

        # output 1: the degrees (i.e., number of edges) for each node is displayed
        plt.figure(figsize=(16, 12))
        plt.subplot(2, 1, 1)
        plt.bar(list(degrees.keys()), list(degrees.values()))
        plt.xlabel('Node')
        plt.ylabel('Degree')
        plt.grid(True)
        plt.title(f'Experimental Degrees Distribution with lambda = {lambda_rate}')
        plt.yticks(list(set(degrees.values())))
        # plt.xticks([x for x in list(degrees.keys()) if degrees[x]!=0])

        # output 2: the experimental degrees and their frequencies are displayed
        plt.subplot(2, 1, 2)
        plt.hist(list(degrees.values()), alpha=0.5, label='Experimental Degrees Distribution')

        lambda_poisson = (n - 1) * p
        degrees_analytical = poisson.rvs(lambda_poisson, size=n)
        plt.hist(degrees_analytical, alpha=0.4, color='red', label='Analytical Degrees Distribution')

        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title(f'Empirical vs Analytical Degrees Distribution with lambda = {lambda_rate}')
        plt.grid(True)

        # output 3: qq-plot
        # plt.subplot(3, 1, 3)
        # _, _ = probplot(list(degrees.values()), dist='poisson', sparams=(lambda_rate), fit=True, plot=plt)
        # create Q-Q plot with 45-degree line
        sm.qqplot(np.array(list(set(degrees.values()))), line='45')

        # plt.xlabel('Theoretical Quantiles (Poisson)')
        # plt.ylabel('Ordered Values')
        # plt.title('Q-Q Plot - Experimental vs Analytical Degrees Distribution')

        # output 4: chi squared test
        chi_squared_p_value = chi_squared_test(list(degrees.values()), degrees_analytical)
        print(f'Chi-squared test p-value: {chi_squared_p_value}\n')

        plt.tight_layout()
        plt.show()