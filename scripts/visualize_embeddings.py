import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the saved embeddings CSV
embeddings_csv_path = "data/fitbit/activity_embeddings.csv"

# Load the embeddings
embeddings_df = pd.read_csv(embeddings_csv_path)

# Plot the embeddings
plt.figure(figsize=(10, 6))
plt.scatter(
    embeddings_df["Component1"],
    embeddings_df["Component2"],
    c="blue",
    label="Activity Embeddings",
    alpha=0.7,
)

# Adding labels and title
plt.title("2D Visualization of Activity Embeddings", fontsize=14)
plt.xlabel("Component 1", fontsize=12)
plt.ylabel("Component 2", fontsize=12)

# Display the plot
plt.legend()
plt.grid(True)
plt.show()
