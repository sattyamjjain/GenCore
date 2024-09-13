import numpy as np
import faiss
import pandas as pd

# File paths
embeddings_csv_path = "data/fitbit/activity_embeddings_with_clusters.csv"
activity_data_path = (
    "data/fitbit/activity_embeddings_with_clusters.csv"  # Activity data with clusters
)

# Load the clustered embeddings
try:
    embeddings_with_clusters = np.genfromtxt(
        embeddings_csv_path, delimiter=",", skip_header=1, dtype="float32"
    )
    print(f"Loaded clustered embeddings with shape: {embeddings_with_clusters.shape}")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    exit(1)

# Separate embeddings and cluster labels
embeddings = embeddings_with_clusters[
    :, :-1
]  # All columns except the last one (embedding data)
cluster_labels = embeddings_with_clusters[:, -1]  # Last column is the cluster label

# Load the activity data
try:
    activity_data = np.genfromtxt(
        activity_data_path, delimiter=",", skip_header=1, dtype="float32"
    )
    print(f"Loaded activity data with shape: {activity_data.shape}")
except Exception as e:
    print(f"Error loading activity data: {e}")
    exit(1)

# Initialize FAISS index
d = embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(d)

# Add embeddings to FAISS index
index.add(embeddings)
print(f"Number of embeddings added to index: {index.ntotal}")


# Function to perform cluster-based FAISS retrieval
def search_within_cluster(
    query_embedding, embeddings, cluster_labels, faiss_index, query_cluster=None
):
    if query_cluster is not None:
        # Filter embeddings by the same cluster
        same_cluster_indices = np.where(cluster_labels == query_cluster)[0]
        same_cluster_embeddings = embeddings[same_cluster_indices]
    else:
        # Use all embeddings for search
        same_cluster_embeddings = embeddings
        same_cluster_indices = np.arange(len(embeddings))

    # Create a FAISS index only for this cluster
    cluster_index = faiss.IndexFlatL2(d)  # Using L2 distance
    cluster_index.add(same_cluster_embeddings)

    # Search in this cluster for the query embedding
    distances, indices = cluster_index.search(np.array([query_embedding]), 5)

    # Map cluster-specific indices back to original indices
    global_indices = same_cluster_indices[indices]

    return distances, global_indices


# Query by index or embedding
query_type = input(
    "Would you like to query by index or custom embedding? (type 'index' or 'embedding'): "
)

if query_type == "index":
    query_index = int(input(f"Enter the index (0 to {embeddings.shape[0] - 1}): "))
    query_embedding = embeddings[query_index]
    query_cluster = cluster_labels[query_index]

    # Perform search within the same cluster
    distances, indices = search_within_cluster(
        query_embedding, embeddings, cluster_labels, index, query_cluster
    )
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    # Display the most similar activity data entries
    for i, idx in enumerate(indices[0]):
        print(f"\nEntry {i + 1}:")
        entry = activity_data[idx]
        print(f"  Id: {entry[0]}")
        print(f"  TotalSteps: {entry[1]}")
        print(f"  TotalDistance: {entry[2]}")
        # Add other fields here from the activity data

elif query_type == "embedding":
    query_embedding = [
        float(x)
        for x in input(
            "Enter custom embedding vector (comma-separated values): "
        ).split(",")
    ]
    query_embedding = np.array(query_embedding, dtype="float32")

    # Perform search for the closest embeddings
    distances, indices = search_within_cluster(
        query_embedding, embeddings, cluster_labels, index
    )

    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    # Display the most similar activity data entries
    for i, idx in enumerate(indices[0]):
        print(f"\nEntry {i + 1}:")
        entry = activity_data[idx]
        print(f"  Id: {entry[0]}")
        print(f"  TotalSteps: {entry[1]}")
        print(f"  TotalDistance: {entry[2]}")
        # Add other fields here from the activity data
