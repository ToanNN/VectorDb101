# Product Quantization splits a high-dimensional vector into a lower dimensional subspace, with dimension of the subspace corresponding to multiple dimensions in the original high-dimensional vector

# Steps:

# Given a dataset of N total vectors, we'll first divide each vector into M subvectors (also known as a subspace). These subvectors don't necessarily have to be the same length, but in practice they almost always are.
# We'll then use k-means (or some other clustering algorithm) for all subvectors in the dataset. This will give us a collection of K centroids for each subspace, each of which will be assigned its own unique ID.
# With all centroids computed, we'll replace all subvectors in the original dataset with the ID of its closest centroid.

## Plain language:
# Divide the original vector to smaller parts
# Cluster each part to a centroid
## replace part with centroid ID

## The trade-off is dependent on the parameters used - using more centroids and subvectors will improve search accuracy

# Each 128D vector will be split into 16 subvectors of size 8, 
# with each subvector then being quantized into one of 256 buckets.
(M, K) = (16, 256)

import numpy as np
dataset = np.random.normal(size=(1000, 128))

sub_vector_length = dataset.shape[1] // M
print(sub_vector_length) # 8

first_sub_space = dataset[:, 0: sub_vector_length]
print(first_sub_space.shape) # (1000, 8)

## Calculate centroids
from scipy.cluster.vq import kmeans2

(centroids, assignments) = kmeans2(first_sub_space, K, iter=32)
# Convert to uint8 from int32_t
quantized = np.uint8(assignments)

