import faiss
import numpy as np

# Step 1: Create random data
d = 128  # Dimension of the data
nb = 10000  # Number of data points (database size)

# Using np.random.seed() to set the seed for reproducibility
np.random.seed(1234)

# Generate random float32 data (embeddings)
data = np.random.rand(nb, d).astype("float32")

# Step 2: Initialize FAISS GPU resources
res = faiss.StandardGpuResources()

# Create FAISS index using L2 (Euclidean) distance
index = faiss.IndexFlatL2(d)

# Step 3: Move the index to the GPU
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

# Step 4: Add data to the FAISS GPU index
gpu_index.add(data)
print(f"Number of embeddings added to the index: {gpu_index.ntotal}")

# Step 5: Perform a similarity search
query = np.random.rand(1, d).astype("float32")  # Random query vector
k = 5  # Number of nearest neighbors to retrieve

# Perform the search
distances, indices = gpu_index.search(query, k)
print("Distances:", distances)
print("Indices:", indices)

# Step 6: Retrieve and print nearest neighbors
nearest_neighbors = data[indices[0]]
print("Nearest neighbors based on the query:")
print(nearest_neighbors)
