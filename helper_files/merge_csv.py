import os
import glob
import pandas as pd
from collections import defaultdict


# Connet spine csv for each image type into one big df (N=number of patients, M=number of features)
main_folder = r"D:\DP\DATA_Myel"

# Find all CSVs in all subfolders
all_csvs = glob.glob(os.path.join(main_folder, "*", "*.csv"))

# Group CSVs by filename
csv_dict = defaultdict(list)
for file in all_csvs:
    csv_name = os.path.basename(file)  # filename, e.g., featureA.csv
    csv_dict[csv_name].append(file)

# For each CSV name, merge all rows from different patients
for csv_name, files in csv_dict.items():
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        patient_name = os.path.basename(os.path.dirname(file))  # folder name, e.g., myel_001
        df.insert(0, "patient", patient_name)
        dfs.append(df)

    # Concatenate all rows into one big DataFrame
    big_df = pd.concat(dfs, ignore_index=True)

    # Save merged CSV
    output_file = os.path.join(main_folder, f"merged_{csv_name}")
    big_df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")