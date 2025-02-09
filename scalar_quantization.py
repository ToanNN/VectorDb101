import numpy as np
dataset = np.random.normal(size=(1000, 128))

# Find the maximum and minimum values
ranges = np.vstack((np.min(dataset, axis=0), np.max(dataset, axis=0)))

print(ranges)

starts = ranges[0, :]
# the step size is determined by the number of discrete bins in the integer type that we'll be using. In this case, we'll be 
# using 8-bit unsigned integers uint8_t for a total of 256 bins
steps = (ranges[1,:] - ranges[0, :]) / 255

dataset_quantized = np.uint8((dataset-starts) / steps)

print(dataset_quantized)
print( np.min(dataset_quantized, axis=0))
print( np.max(dataset_quantized, axis=0))
