import os
from typing import Dict, Set
from collections import deque
import queue
import numpy as np

def load_graph(filename: str) -> Dict[int, Set[int]]:
    """
    Load the graph file. Each line in the file corresponds to an edge; the first column is the source node and the
    second column is the target. As the graph is undirected, your program should add the source as a neighbour of the
    target as well as the target a neighbour of the source.

    @param filename: The path to the network specification
    @return: a dictionary mapping each node (represented by an integer ID) to a set containing all the nodes it is
        connected to (also represented by integer IDs)
    """
    res = dict()
    with open(filename,'r') as file:
        while True:
            nowline = file.readline().rstrip('\n')
            if not nowline:
                break
            nodeA,nodeB = int(nowline.split(" ")[0]),int(nowline.split(" ")[1])
            if nodeA in res:
                res[nodeA].add(nodeB)
            else:
                res[nodeA]=set()
                res[nodeA].add(nodeB)
            if nodeB in res:
                res[nodeB].add(nodeA)
            else:
                res[nodeB]=set()
                res[nodeB].add(nodeA)
    return res



def get_node_degrees(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Find the number of neighbours of each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to the degree of the node
    """
    res = dict()
    for key in graph:
        res[key] = len(graph[key])
    return res


def get_diameter(graph: Dict[int, Set[int]]) -> int:
    """
    Find the longest shortest path between any two nodes in the network using a breadth-first search.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the length of the longest shortest path between any pair of nodes in the graph
    """
    size = len(graph)
    flag = [False for i in range(size+1)]
    distance = [-1 for i in range(size+1)]
    maxdis = 0
    for s in graph.keys():
        for i in range(size+1):
            flag[i]=False
        nowq = [s]
        flag[s]=True
        distance[s]=0
        while nowq:
            nownode = nowq.pop(0)
            for neibour in graph[nownode]:
                if not flag[neibour]:
                    nowq.append(neibour)
                    distance[neibour]=distance[nownode]+1
                    maxdis = max(maxdis, distance[neibour])
                    flag[neibour]=True

    return maxdis

    # def bfs(start,visited):
    #     nowdis = 0
    #     to_explore = queue.Queue()
    #     to_explore.put(start)
    #     distance = np.zeros(size+1)
    #     distance[start]=0
    #     visited.add(start)
    #     while not to_explore.empty():
    #         nownode = to_explore.get()
    #         nowdis = max(nowdis,distance[nownode])
    #         for neighbour in graph[nownode]:
    #             if neighbour not in visited:
    #                 to_explore.put(neighbour)
    #                 distance[neighbour]=distance[nownode]+1
    #                 visited.add(neighbour)
    #     return nowdis
    # res = 0
    # count = 0
    # visited = set()
    # for key in graph:
    #     res = max(res,bfs(key,visited))
    #     visited.clear()
    #     count +=1
    #     if count%100==0:
    #         print(f"{count}/{size}")
    #
    # return res






def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    degrees = get_node_degrees(graph)
    print(f"Node degrees: {degrees}")

    diameter = get_diameter(graph)
    print(f"Diameter: {diameter}")


if __name__ == '__main__':
    main()