import numpy as np
from scipy.cluster.vq import kmeans2
num_part =16
dataset = np.random.normal(size=(1000, 128))
(centroids, assignments) = kmeans2(dataset, num_part, iter=32)
print(centroids.shape)

## create the inverted file index by correlating each centroid with a list of vectors within the cluster:

index = [[] for _ in range(num_part)]

for vector_index, cluster_index  in enumerate(assignments):
    index[cluster_index].append(vector_index)
    
query = np.random.normal(size=(128,))

# Find the nearest partition
nearest_centroid = np.argmin(np.linalg.norm(centroids - query, axis=1))

print(f'Nearest centroid {nearest_centroid}')
# Find the nearest vector in the partition 
nearest_neighbour = np.argmin(np.linalg.norm(dataset[index[nearest_centroid]] - query, axis=1))

print(f'Nearest neighbour {nearest_neighbour}')


