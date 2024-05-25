import csv
import json
import pandas as pd

# Define the input and output file paths
input_csv = './sarah.csv'
output_json = 'output.json'
df = pd.read_csv(input_csv)

# Initialize an empty list to hold the medication dictionaries
medications_list = []

# Open and read the CSV file
with open(input_csv, mode='r', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.DictReader(csvfile)
    
    # Iterate over each row in the CSV
    for row in csvreader:
        # Create a dictionary for each medication
        medication_dict = {
            "name": row["drug_name"],
            "medical_condition": row["medical_condition"],
            "side_effects": row["side_effects"].split(", "),  # Assuming side effects are comma-separated
            "generic_name": row["generic_name"],
            "stock": int(row["stock"])  # Assuming stock is in a column and is either 0 or 1
        }
        medications_list.append(medication_dict)
        
# Wrap the list in a dictionary with the key "medications"
data_dict = {"medications": medications_list}

# Write the dictionary to a JSON file
with open(output_json, mode='w', encoding='utf-8') as jsonfile:
    json.dump(data_dict, jsonfile, indent=4)

print(f'CSV data has been converted to JSON and saved to {output_json}')
