import numpy as np

query = np.random.normal(size=(128,))
dataset= np.random.normal(size=(1000,128))

# computes the distance (via np.linalg.norm) between all elements 
# in the dataset and the nearest neighbors of the query vector 
# before extracting the index of the minimum distance (via np.argmin). 
# This gives us the array index of the nearest neighbor to the query vector, 
# which we can then extract using dataset[nearest,:
nearest = np.argmin(np.linalg.norm(dataset-query, axis=1))

print(nearest)

print(query)
print(dataset[nearest])
print(f'Distance: {np.linalg.norm(dataset[nearest]-query)}')