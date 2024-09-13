import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Define paths for the input JSON and output CSV files
input_json_path = "data/fitbit/daily_activity.json"  # Path to daily_activity.json
output_embeddings_path = (
    "data/fitbit/activity_embeddings.csv"  # Path to save activity embeddings
)

# Load the processed JSON data (daily activity data)
daily_activity_df = pd.read_json(input_json_path)

# Select the features for embeddings (TotalSteps, TotalDistance, Calories, etc.)
features = [
    "TotalSteps",
    "TotalDistance",
    "VeryActiveMinutes",
    "FairlyActiveMinutes",
    "LightlyActiveMinutes",
    "Calories",
]

# Check if all features are present
missing_features = [f for f in features if f not in daily_activity_df.columns]
if missing_features:
    print(
        f"Warning: The following features are missing from the dataset: {missing_features}"
    )

# Extract the relevant features
activity_data = daily_activity_df[features].dropna()  # Drop rows with missing values

# Step 1: Standardize the data (scale it to mean 0, variance 1)
scaler = StandardScaler()
scaled_activity_data = scaler.fit_transform(activity_data)

# Step 2: Reduce dimensionality (optional, for visualization or efficiency)
# You can use PCA to reduce the dimensions of the embedding
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization purposes
activity_embeddings = pca.fit_transform(scaled_activity_data)

# Step 3: Store or analyze the embeddings
print("Generated Activity Embeddings:")
print(activity_embeddings)

# Step 4: Save the embeddings for further use
output_dir = os.path.dirname(output_embeddings_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pd.DataFrame(activity_embeddings, columns=["Component1", "Component2"]).to_csv(
    output_embeddings_path, index=False
)
print(f"Embeddings successfully saved to {output_embeddings_path}")
