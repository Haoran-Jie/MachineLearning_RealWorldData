import os
from typing import Set, Dict, List, Tuple
from exercises.tick10 import load_graph


def get_number_of_edges(graph: Dict[int, Set[int]]) -> int:
    """
    Find the number of edges in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the number of edges
    """
    count = 0
    for node in graph:
        count+=len(graph[node])
    return count/2


def get_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find the number of components in the graph using a DFS.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: list of components for the graph.
    """
    visited = set()
    count = 0
    def dfs(s):
        visited.add(s)
        stackk = [s]
        while stackk:
            nownode = stackk.pop()
            for neibours in graph[nownode]:
                if neibours not in visited:
                    stackk.append(neibours)
                    visited.add(neibours)
    for nodes in graph:
        if nodes not in visited:
            dfs(nodes)
            count+=1
    return count



def get_edge_betweenness(graph: Dict[int, Set[int]]) -> Dict[Tuple[int, int], float]:
    """
    Calculate the edge betweenness.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: betweenness for each pair of vertices in the graph connected to each other by an edge
    """
    pass


def girvan_newman(graph: Dict[int, Set[int]], min_components: int) -> List[Set[int]]:
    """     * Find the number of edges in the graph.
     *
     * @param graph
     *        {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *        loaded graph
     * @return {@link Integer}> Number of edges.
    """
    pass


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'facebook_circle.edges'))

    num_edges = get_number_of_edges(graph)
    print(f"Number of edges: {num_edges}")

    components = get_components(graph)
    print(f"Number of components: {len(components)}")

    edge_betweenness = get_edge_betweenness(graph)
    print(f"Edge betweenness: {edge_betweenness}")

    clusters = girvan_newman(graph, min_components=20)
    print(f"Girvan-Newman for 20 clusters: {clusters}")


if __name__ == '__main__':
    main()