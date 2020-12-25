import sys
import heapq

## Modified from: https://github.com/mburst/dijkstras-algorithm/blob/master/dijkstras.py

def shortest_path(vertices, start, finish):
    distances = {} # Distance from start to node
    previous = {}  # Previous node in optimal path from source
    nodes = [] # Priority queue of all nodes in Graph

    for vertex in vertices:
        if vertex == start: # Set root node as distance of 0
            distances[vertex] = 0
            heapq.heappush(nodes, [0, vertex])
        else:
            distances[vertex] = sys.maxsize
            heapq.heappush(nodes, [sys.maxsize, vertex])
        previous[vertex] = None

    while nodes:
        smallest = heapq.heappop(nodes)[1] # Vertex in nodes with smallest distance in distances
        if smallest == finish: # If the closest node is our target we're done so print the path
            path = []
            while previous[smallest]: # Traverse through nodes til we reach the root which is 0
                path.append(smallest)
                smallest = previous[smallest]
            return path[::-1]
        if distances[smallest] == sys.maxsize: # All remaining vertices are inaccessible from source
            break

        for neighbor in vertices[smallest]: # Look at all the nodes that this vertex is attached to
            alt = distances[smallest] + vertices[smallest][neighbor] # Alternative path distance
            if alt < distances[neighbor]: # If there is a new shortest path update our priority queue (relax)
                distances[neighbor] = alt
                previous[neighbor] = smallest
                for n in nodes:
                    if n[1] == neighbor:
                        n[0] = alt
                        break
                heapq.heapify(nodes)

    return None
