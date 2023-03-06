import os
from typing import Dict, Set
from exercises.tick10 import load_graph


def get_node_betweenness(graph: Dict[int, Set[int]]) -> Dict[int, float]:
    # initialize betweenness centrality for all nodes to zero
    betweenness = {node: 0.0 for node in graph}

    # loop over all nodes as the source node
    for s in graph:
        # initialize variables for computing shortest paths and dependencies
        S = []  # stack
        P = {node: [] for node in graph}  # predecessors
        sigma = {node: 0 for node in graph}  # number of shortest paths from s
        sigma[s] = 1
        d = {node: -1 for node in graph}  # distance from s to node
        d[s] = 0

        # perform breadth-first search from s
        Q = [s]  # queue
        while Q:
            v = Q.pop(0)
            S.append(v)
            for w in graph[v]:
                if d[w] == -1: # unvisited
                    Q.append(w) # append into queue
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        # initialize dependencies for all nodes to zero
        delta = {node: 0 for node in graph}

        # back-propagate dependencies along shortest paths
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
    for key in betweenness:
        betweenness[key]/=2
    return betweenness

def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    betweenness = get_node_betweenness(graph)
    print(f"Node betweenness values: {betweenness}")


if __name__ == '__main__':
    main()