import numpy as np
import faiss
import os

# Path to the embeddings CSV
embeddings_csv_path = os.path.join("data", "fitbit", "activity_embeddings.csv")

# Load embeddings using numpy
try:
    embeddings = np.genfromtxt(
        embeddings_csv_path, delimiter=",", skip_header=1, dtype="float32"
    )
    print(f"Loaded numpy array with shape: {embeddings.shape}")
except Exception as e:
    print(f"Error loading CSV into numpy array: {e}")
    exit(1)

# Load the actual activity data using numpy (similar to embeddings)
activity_data_path = os.path.join(
    "dataset",
    "mturkfitbit_export_3.12.16-4.11.16",
    "Fitabase Data 3.12.16-4.11.16",
    "dailyActivity_merged.csv",
)

try:
    activity_data = np.genfromtxt(
        activity_data_path, delimiter=",", skip_header=1, dtype="float32"
    )
    print(f"Loaded activity data with shape: {activity_data.shape}")
except Exception as e:
    print(f"Error loading activity data into numpy array: {e}")
    exit(1)

# Activity data column names for better interpretation
activity_columns = [
    "Id",
    "ActivityDate",
    "TotalSteps",
    "TotalDistance",
    "TrackerDistance",
    "LoggedActivitiesDistance",
    "VeryActiveDistance",
    "ModeratelyActiveDistance",
    "LightActiveDistance",
    "SedentaryActiveDistance",
    "VeryActiveMinutes",
    "FairlyActiveMinutes",
    "LightlyActiveMinutes",
    "SedentaryMinutes",
    "Calories",
]

# Step 1: Set up FAISS index and add embeddings
d = embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(d)

# Add embeddings to the FAISS index
index.add(np.array(embeddings, dtype=np.float32))


# Step 2: Perform a similarity search
def retrieve_similar_entries(query_embedding, k=5):
    query_embedding = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_embedding, k)
    return distances, indices


# Step 3: Handle user input
query_type = input(
    "Would you like to query by index or custom embedding? (type 'index' or 'embedding'): "
)

if query_type.lower() == "index":
    query_index = int(input(f"Enter the index (0 to {len(embeddings) - 1}): "))
    query_embedding = embeddings[query_index]
else:
    try:
        query_embedding = [
            float(x)
            for x in input(
                "Enter custom embedding vector (comma-separated values): "
            ).split(",")
        ]
        if len(query_embedding) != d:
            raise ValueError(f"Please enter exactly {d} values for the embedding.")
    except ValueError as ve:
        print(f"Invalid input: {ve}")
        exit(1)

# Retrieve similar entries
distances, indices = retrieve_similar_entries(query_embedding)

# Handle NaN values in the activity data by replacing them with 0 or a placeholder
clean_activity_data = np.nan_to_num(activity_data, nan=0.0)

# Show the retrieved data (similar entries from the original dataset)
similar_data = clean_activity_data[indices[0]]  # Use indices from FAISS search

# Print labeled activity data for each entry
print("Distances:", distances)
print("Indices:", indices)
print("Similar entries from the activity data:")
for i, entry in enumerate(similar_data):
    print(f"\nEntry {i + 1}:")
    for col_name, value in zip(activity_columns, entry):
        print(f"  {col_name}: {value}")
