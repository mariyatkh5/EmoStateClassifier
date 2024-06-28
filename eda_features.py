import os
import pandas as pd
import numpy as np
import neurokit2 as nk
import warnings
from scipy.signal import resample
from sklearn.decomposition import PCA

# Function to process EDA files
def process_eda_file(file_path):
    try:
        # Load data
        data = pd.read_csv(file_path, sep=";")
        
        # Convert timestamps to seconds
        data["TimeStamp"] = data["TimeStamp"].astype(str)
        data["TimeStamp"] = data["TimeStamp"].apply(lambda x: float(x.replace(",", ".")))
        
        # Extract and clean EDA data as an array of numeric values
        eda_values = []
        for value in data["EDA"]:
            try:
                if isinstance(value, str):  # Check if the value is already a string
                    eda_values.append(float(value.replace(",", ".")))
                else:
                    # If the value is not a string, assume it is already a number
                    eda_values.append(value)
            except Exception as e:
                print(f"Error extracting and cleaning EDA values: {e}")
                eda_values.append(np.nan)
        
        # Check if the length of EDA data is sufficient
        if len(eda_values) < 1:  # Example threshold
            print("Error: The length of EDA data is too short.")
            return None
        
        # Data processing
        df, info = nk.eda_process(eda_values, sampling_rate=4)  # Assume sampling rate is 4 Hz
        
        # Analysis
        analyze_df = nk.eda_analyze(df, sampling_rate=4)
        
        return analyze_df
    
    except Exception as e:
        print(f"Error loading file or general error: {e}")
        return None

warnings.filterwarnings("ignore")

# Helper function to rename columns for compatibility with the rest of the pipeline
def transform_eda(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to rename the columns of a data frame such that they work with the rest of the pipeline.

    :param raw_data: Raw data frame which needs to be renamed.
    :return raw_data: Renamed data frame.
    """

    # Rename columns
    for k in raw_data.keys():
        if "time" in k.lower():
            raw_data = raw_data.rename(columns={k: "LocalTimestamp"})
        elif "ea" in k.lower():
            raw_data = raw_data.rename(columns={k: "EDA"})

    return raw_data

# Function to extract information from the file path
def extract_info_from_path(file_path):
    parts = file_path.split(os.path.sep)
    person = parts[-3].split("_")[1]
    speed, robots = parts[-2].split("_")
    return person, speed, robots

# List to store the results
all_analyze_dfs = []

# Iterate through all EDA files
eda_files = []
for person in range(1,26):
    for robot_speed in [1, 2]:
        for num_robots in [1, 2, 3]:
            file_path = os.path.join("Measurements_fixed", f"p_{person}", f"{robot_speed}_{num_robots}", "EDA.csv")
            if os.path.exists(file_path):
                eda_files.append(file_path)

# Perform analysis for each EDA file
for file_path in eda_files:
    analyze_df = process_eda_file(file_path)
    if analyze_df is not None:
        all_analyze_dfs.append((file_path, analyze_df))
    else:
        print(f"The EDA data in the file '{file_path}' is too short.")

# Create folder for results
output_folder = "analysis_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save results in separate CSV files
for idx, (file_path, result) in enumerate(all_analyze_dfs):
    person, speed, robots = extract_info_from_path(file_path)
    print(f"Filename: {file_path}")
    print(f"Person: {person}, Robot speed: {speed}, Number of robots: {robots}")
    
    # Create the full path for the CSV file
    output_folder_person = os.path.join(output_folder, f"Person_{person}")
    os.makedirs(output_folder_person, exist_ok=True)  # Create directory for the person if it does not exist
    
    output_file = os.path.join(output_folder_person, f"Analysis_{speed}_{robots}.csv")
    
    result.to_csv(output_file, index=False)  # Save DataFrame to CSV file

    print(f"Results for Person {person}, Robot speed {speed}, Number of robots {robots} saved.")
