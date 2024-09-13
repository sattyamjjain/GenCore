import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the clustered embeddings
embeddings_csv_path = "data/fitbit/activity_embeddings_with_clusters.csv"

# Using pd.read_csv to load CSV data
try:
    embeddings_with_clusters = pd.read_csv(embeddings_csv_path).values
    print(f"Loaded clustered embeddings with shape: {embeddings_with_clusters.shape}")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    exit(1)

# Separate embeddings and cluster labels
embeddings = embeddings_with_clusters[:, :-1]  # All columns except the last one
cluster_labels = embeddings_with_clusters[:, -1].astype(
    int
)  # Last column is the cluster label

# Convert cluster_labels to a pandas Series
cluster_labels_series = pd.Series(cluster_labels, name="Cluster")
print(f"Cluster labels dtype: {cluster_labels_series.dtype}")

# Count points per cluster
unique_clusters, counts_per_cluster = np.unique(cluster_labels, return_counts=True)
print(f"Clusters and their counts: {list(zip(unique_clusters, counts_per_cluster))}")

# Load activity data using pd.read_csv
activity_data_path = "data/fitbit/activity_embeddings_with_clusters.csv"  # Placeholder

try:
    activity_data = pd.read_csv(activity_data_path).values
    print(f"Loaded activity data with shape: {activity_data.shape}")
except Exception as e:
    print(f"Error loading activity data: {e}")
    exit(1)

# Convert activity data into a pandas DataFrame
df_activity = pd.DataFrame(activity_data)

# Check if the DataFrame has compatible data types
print(f"Activity DataFrame dtypes:\n{df_activity.dtypes}")

# Ensure cluster_labels_series is the same length as df_activity
if len(cluster_labels_series) == len(df_activity):
    df_activity = df_activity.copy()  # Copy to avoid modification warnings
    # Insert cluster labels as a new column, ensuring data type compatibility
    df_activity["Cluster"] = cluster_labels_series
    print("Cluster labels added to activity data.")
else:
    print(
        f"Length mismatch: cluster_labels({len(cluster_labels_series)}) vs activity_data({len(df_activity)})"
    )
    exit(1)

# Analyze statistics per cluster
cluster_stats = df_activity.groupby("Cluster").agg(["mean", "std"])
print(cluster_stats)

# Plot the distribution of clusters
plt.bar(unique_clusters, counts_per_cluster)
plt.title("Cluster Distribution")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Points")
plt.show()
