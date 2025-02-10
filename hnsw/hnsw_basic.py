import numpy as np
dataset = np.random.normal(size=(1000, 128))

# BuilD HNSW index
layers = 5;
# use list to represent the index
# the inner lists corresponding to each layer/graph

index = [[] for _ in range(layers)]

# Every element in each graph is a 3-tuple containing:  
# - the vector itself
# - A list of indexes the vector links to within the graph
# - The index of the node that in the layer below it or None for the bottom layer

# Search for the nearest neighbor in the graph


def _searach_layer(graph, entry, query,efforts =1):
    best = np.linalg.norm(graph[entry][0] - query, entry)
    
    nns = [best]
    # set of visited nodes
    visit = set(best) 
     # candidate nodes to insert into nearest neighbors
    nearest_neighbour_candidates = [best]
    
    heapify(nearest_neighbour_candidates)
    
    # find top-k nearest neighbors
    while nearest_neighbour_candidates:
        cv = heappop(nearest_neighbour_candidates)
        
        if(nns[-1][0] < cv[0]):
            break
        
        # loop through all nearest neighbors to the candidate vector
        for e in graph[cv[1]][1]:
            distance = np.linalg.norm(graph[e][0] - query)
            if(distance, e) not in visit:
                visit.add((distance, e))

                # push only "better" vectors into candidate heap
                if distance < nns[-1][0] or len(nns) < efforts:
                    heappush(nearest_neighbour_candidates, (distance, e))
                    insort(nns, (distance, e))
                    if len(nns) > efforts:
                        nns.pop()
    return nns

def search(index, query, effort =1):
    # If the index is empty, return empty list
    if not index[0]:
        return []
    # Initial vertex entry point
    # We first start at the entry point (zeroth element in the uppermost graph), 
    # and search for the nearest neighbor in each index layer until we reach the bottommost layer
    best_vertex = 0
    
    for graph in index:
        best_distance, best_vertex = _searach_layer(graph, best_vertex,query, effort )
        if graph[best_vertex][2]:
            best_vertex = graph[best_vertex][2]
        # Reach to the bottom layer
        else:
            return _searach_layer(graph, best_vertex,query, effort )

# Find the layer to insert a vector into

def _get_insert_layer(L, mL):
    # ml is a multiplicative factor used to normalized the distribution
    l = -int(np.log(np.random.random()) * mL)
    return min(l, L)
  
def insert(self, vec, efc =10):
    # If the index is empty, insert the vector into all layers and return
    if not index[0]:
        i =None
        for graph in index[::-1]:
            graph.append((vec, [], i))
            i=0
        return 
    
    l  = _get_insert_layer(1/np.log(L))
    start_v = 0
    for n, graph in enumerate(index):
        # perform insertion for layers [l, L) only
        if n < l:
            _, start_v = _searach_layer(graph, start_v, vec, efforts=1)[0]
        # For layer l and all those below it, we first find the nearest 
        # neighbors to vec up to a pre-determined number ef. We then create connections from the node to its nearest neighbors and vice versa. 
        else:
            node = (vec, [], len(index[n+1]) if n < (L-1) else None) 
            nns = _searach_layer(graph, start_v, vec, ef = efc)
            for nn in nns:
                # outbound connections to NNss
                node[1].append(nns[1])
                # inboudn connection to node
                graph[nn[1]][1].append(len(graph))
            graph.append(node)
        # set the starting vertex to the nearest neighbor in the next layer
        start_v = graph[start_v][2]