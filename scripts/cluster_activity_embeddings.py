import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define the path to the saved embeddings CSV
embeddings_csv_path = "data/fitbit/activity_embeddings.csv"

# Load the embeddings
embeddings_df = pd.read_csv(embeddings_csv_path)

# Step 1: Apply K-Means Clustering
kmeans = KMeans(
    n_clusters=3, random_state=42
)  # You can adjust the number of clusters (n_clusters)
clusters = kmeans.fit_predict(embeddings_df)

# Add the cluster labels to the DataFrame
embeddings_df["Cluster"] = clusters

# Step 2: Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(
    embeddings_df["Component1"],
    embeddings_df["Component2"],
    c=embeddings_df["Cluster"],
    cmap="viridis",
    label="Clusters",
    alpha=0.7,
)

# Adding labels and title
plt.title("K-Means Clustering of Activity Embeddings", fontsize=14)
plt.xlabel("Component 1", fontsize=12)
plt.ylabel("Component 2", fontsize=12)

# Display the plot
plt.legend()
plt.grid(True)
plt.show()

# Save the clustered embeddings to a CSV file
output_clustered_embeddings_path = "data/fitbit/activity_embeddings_with_clusters.csv"
embeddings_df.to_csv(output_clustered_embeddings_path, index=False)
print(f"Clustered embeddings saved to {output_clustered_embeddings_path}")
