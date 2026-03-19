import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

from scipy.stats import kendalltau
from statsmodels.stats.multitest import multipletests


# CONFIG
@dataclass
class Config:
    main_folder_path: str
    image_types: List[str]
    patient_dates: Dict[str, List[str]]


# DATA LOADER
class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self._patients_structure = None
        self._patient_dfs = None

    def load_structure(self):
        if self._patients_structure is not None:
            return self._patients_structure

        patients = {img: defaultdict(list) for img in self.config.image_types}
        pattern = r"Myel_FollowUp_(\d+)_(\d+)"

        for folder in os.listdir(self.config.main_folder_path):
            match = re.match(pattern, folder)
            if not match:
                continue

            patient_id = match.group(1)
            followup_number = int(match.group(2))

            folder_path = os.path.join(self.config.main_folder_path, folder)
            for file in os.listdir(folder_path):

                if file.endswith(".csv") and "spine_lesions" in file:
                    for img in self.config.image_types:
                        if img in file:
                            csv_path = os.path.join(folder_path, file)
                            patients[img][patient_id].append((followup_number, csv_path))

        self._patients_structure = patients
        return patients

    def load_dataframes(self):
        if self._patient_dfs is not None:
            return self._patient_dfs

        structure = self.load_structure()
        patient_dfs = {}

        for img_type, patient_data in structure.items():
            patient_dfs[img_type] = {}
            for patient, scans in patient_data.items():
                scans = sorted(scans, key=lambda x: x[0])

                dfs = [pd.read_csv(path) for _, path in scans]
                df = pd.concat(dfs, ignore_index=True)
                df.insert(0, "Date", self.config.patient_dates[patient])

                patient_dfs[img_type][patient] = df

        self._patient_dfs = patient_dfs
        return patient_dfs


# ANALYZER
class Analyzer:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self._kendall_results = None

    def run_kendall(self):
        if self._kendall_results is not None:
            return self._kendall_results

        data = self.data_loader.load_dataframes()
        results = {}

        for img_type, patient_data in data.items():
            results[img_type] = {}
            for patient, df in patient_data.items():

                df = df.copy()
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
                df = df.sort_values("Date")

                features = df.columns[1:]
                time_index = range(len(df))

                rows = []
                for feature in features:
                    values = df[feature].values
                    if len(values) < 3:
                        continue

                    tau, p = kendalltau(time_index, values)
                    rows.append({
                        "Feature": feature,
                        "Tau": tau,
                        "P_value": p
                    })

                res_df = pd.DataFrame(rows)

                if not res_df.empty:
                    res_df["FDR_p"] = multipletests(res_df["P_value"], method="fdr_bh")[1]
                    res_df = res_df.sort_values("FDR_p")

                results[img_type][patient] = res_df

        self._kendall_results = results
        return results

    def get_significant_features(self, img_type, patient):
        df = self.run_kendall()[img_type][patient].copy()
        sig = df[df["P_value"] < 0.05].copy()
        sig["Trend"] = sig["Tau"].apply(lambda x: "Increasing" if x > 0 else "Decreasing")

        return sig.sort_values("P_value").head(10)

    def plot_feature(self, img_type, feature):
        data = self.data_loader.load_dataframes()
        for patient, df in data[img_type].items():

            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
            df = df.sort_values("Date")

            plt.figure(figsize=(8, 5))
            plt.plot(df["Date"], df[feature], marker="o")
            plt.xticks(df["Date"], df["Date"].dt.strftime("%Y-%m-%d"))

            plt.xlabel("Date")
            plt.ylabel(feature)
            plt.title(f"Patient: {patient} - {feature} ({img_type})")

            plt.grid(True)
            plt.tight_layout()
            plt.show()


# MAIN
if __name__ == "__main__":
    config_setup = Config(
        main_folder_path=r"E:\DATA_FollowUp",
        image_types=["CaSupp_25", "monoe_40kev", "konv"],
        patient_dates={
            "001": ["12-05-2022", "10-05-2024", "07-10-2025"],
            "002": ["07-10-2022", "21-06-2023", "28-06-2024", "14-02-2025", "04-07-2025"],
            "003": ["20-01-2023", "07-11-2023", "22-11-2024", "14-03-2025"],
            "004": ["29-11-2022", "20-11-2023", "22-02-2024", "06-06-2024", "24-06-2025"],
            "005": ["09-06-2022", "03-04-2023", "07-05-2024", "05-05-2025"],
            "006": ["27-03-2023", "26-06-2023", "08-11-2023", "27-02-2024"],
            "007": ["04-10-2023", "21-10-2024", "10-04-2025", "21-10-2025"],
            # "008": [],
        }
    )

    loader = DataLoader(config_setup)
    analyzer = Analyzer(loader)

    analyzer.plot_feature("monoe_40kev", "gradient_firstorder_10Percentile")

    significant = analyzer.get_significant_features("monoe_40kev", "002")
    print(significant.to_string())
