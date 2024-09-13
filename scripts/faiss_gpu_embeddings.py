import numpy as np
import faiss

# Path to the embeddings CSV
embeddings_csv_path = "data/fitbit/activity_embeddings.csv"

# Step 1: Load the CSV directly into numpy arrays
try:
    # Load the CSV file into a numpy array, skipping the header
    embeddings = np.genfromtxt(
        embeddings_csv_path, delimiter=",", skip_header=1, dtype="float32"
    )
    print(f"Loaded numpy array with shape: {embeddings.shape}")
except Exception as e:
    print(f"Error loading CSV into numpy array: {e}")
    exit(1)

# Step 2: Initialize FAISS GPU resources (without GpuResourceConfig)
try:
    res = faiss.StandardGpuResources()  # Initialize GPU resources

    # Initialize the FAISS index for L2 distance
    d = embeddings.shape[1]  # dimension of the embeddings (2 in our case)
    index = faiss.IndexFlatL2(d)

    # Transfer the index to GPU
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

    # Step 3: Add embeddings to the FAISS GPU index
    gpu_index.add(embeddings)
    print(f"Number of embeddings added to index: {gpu_index.ntotal}")

    # Step 4: Perform a similarity search on GPU
    query_embedding = np.array(
        [embeddings[0]]
    )  # Example: using the first embedding as a query
    k = 5  # Number of nearest neighbors to retrieve

    distances, indices = gpu_index.search(query_embedding, k)
    print("Distances:", distances)
    print("Indices:", indices)

    # Retrieve the nearest neighbors
    nearest_neighbors = embeddings[indices[0]]
    print("Nearest neighbors based on query:")
    print(nearest_neighbors)

except Exception as e:
    print(f"Error during FAISS operations: {e}")
    exit(1)
