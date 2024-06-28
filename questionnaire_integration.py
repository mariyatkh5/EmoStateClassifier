import pandas as pd
import os

# Path to the Excel file containing the questionnaire data
excel_file_path = "Questionnaire_data1.xlsx"

# Load the Excel file
questionnaire_data = pd.read_excel(excel_file_path)

# Function to extract information from the Index column
def extract_info_from_index(index):
    row = questionnaire_data[questionnaire_data['Index'] == index]
    return row

# Iterate through all CSV files
for person in range(1, 26):  # Assuming person numbering from 1 to 25
    for speed in [1, 2]:  # Assuming speeds 1 and 2
        for robots in [1, 2, 3]:  # Assuming number of robots 1, 2, and 3
            # Path to the CSV file
            file_path = f"analysis_results/Person_{person}/Analysis_{speed}_{robots}.csv"
            
            # Check if the file exists
            if os.path.exists(file_path):
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Extract the relevant information from the Index column
                index_info = extract_info_from_index(person)
                x_y_3_value = index_info[f"{speed}.{robots}.3"].iloc[0]
                x_y_4_value = index_info[f"{speed}.{robots}.4"].iloc[0]
                
                # Add new columns to the DataFrame
                df[f"{speed}.{robots}.3"] = x_y_3_value
                df[f"{speed}.{robots}.4"] = x_y_4_value
                
                # Save the updated DataFrame
                df.to_csv(file_path, index=False)
