import pandas as pd
import json
import os


# Function to load a CSV file into a pandas DataFrame
def load_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# Function to convert a DataFrame to structured JSON format
def convert_to_json(df):
    try:
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Error converting DataFrame to JSON: {e}")
        return None


# Function to save JSON data to a file
def save_json(data, output_path):
    try:
        with open(output_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving JSON to {output_path}: {e}")


# Function to process Fitbit data and convert to JSON
def process_fitbit_data(file_paths, output_paths):
    for data_type, file_path in file_paths.items():
        print(f"Processing {data_type} data...")

        # Step 1: Load the CSV file
        df = load_csv_file(file_path)
        if df is not None:
            # Step 2: Convert to JSON format
            json_data = convert_to_json(df)
            if json_data:
                # Step 3: Save the JSON data to a file
                save_json(json_data, output_paths[data_type])


# Define the paths to your downloaded Fitbit CSV files (Update paths accordingly)
fitbit_file_paths = {
    "daily_activity": "dataset/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/dailyActivity_merged.csv",
    "heartrate": "dataset/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/heartrate_seconds_merged.csv",
    "sleep": "dataset/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/minuteSleep_merged.csv",
    "calories": "dataset/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16/hourlyCalories_merged.csv",
}

# Define the paths where you want to save the JSON files (Update paths accordingly)
output_paths = {
    "daily_activity": "data/fitbit/daily_activity.json",
    "heartrate": "data/fitbit/heartrate.json",
    "sleep": "data/fitbit/sleep.json",
    "calories": "data/fitbit/calories.json",
}

# Process and save all Fitbit data
process_fitbit_data(fitbit_file_paths, output_paths)
