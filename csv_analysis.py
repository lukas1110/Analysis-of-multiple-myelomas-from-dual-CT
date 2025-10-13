import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CompareDatasets:
    def __init__(self, raw_datas: dict):
        self.datasets = self._transform_datasets(raw_datas)

    @staticmethod
    def _transform_datasets(input_datasets: dict):
        # Dictionary to hold the *transformed* 1-row DataFrames
        datasets = {}

        for name, df_raw in input_datasets.items():
            if df_raw.shape[0] < 2:
                df = df_raw.iloc[0].to_frame().T
                df.columns = df_raw.columns.tolist()
            else:
                df = df_raw.iloc[1].to_frame().T
                df.columns = df_raw.iloc[0].tolist()

            # Convert values to numeric, coercing non-numeric to NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            datasets[name] = df
        return datasets

    def print_comparison(self):
        comparison_data = []

        for name, df in self.datasets.items():
            # Flatten the 1-row DataFrame into a 1D array of 200 values
            values = df.values.flatten()

            overall_stats = {
                'Dataset': name,  # <-- Using the variable name here
                'Mean_Value_Across_200_Features': values.mean(),
                'Std_Dev_Across_200_Features': values.std(),
                'Min_Value_Across_200_Features': values.min(),
                'Max_Value_Across_200_Features': values.max(),
                'Median_Value_Across_200_Features': np.median(values),
            }
            comparison_data.append(overall_stats)

        # Create the final comparison table
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_markdown(index=False, floatfmt=".3f"))

    def kernel_density_plot(self):
        # Stack the data from the single-row DataFrames for plotting
        all_values = pd.DataFrame()
        for name, df in self.datasets.items():
            temp_df = df.T.rename(columns={df.index[0]: 'Value'})
            temp_df['Dataset'] = name
            all_values = pd.concat([all_values, temp_df])

        # Density Plot (KDE) - Remains Linear Scale for clarity
        plt.figure(figsize=(12, 7))
        for name in self.datasets.keys():
            sns.kdeplot(self.datasets[name].values.flatten(), label=name, fill=True, alpha=0.3, linewidth=1.5)
        plt.title('Density Plot (Distribution) of Feature Values')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend(title='Dataset Variable Name')
        plt.show()


# vmi_40 = pd.read_csv(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\monoe_40kev_radiomics_spine_vertebrae_features.csv")
# vmi_80 = pd.read_csv(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\monoe_80kev_radiomics_spine_vertebrae_features.csv")
# vmi_120 = pd.read_csv(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\monoe_120kev_radiomics_spine_vertebrae_features.csv")

# raw_datasets = {
#     'vmi_40': vmi_40,
#     'vmi_80': vmi_80,
#     'vmi_120': vmi_120
# }



c_25 = pd.read_csv(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\CaSupp_25_radiomics_spine_lesions_features.csv")
c_50 = pd.read_csv(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\CaSupp_50_radiomics_spine_lesions_features.csv")
c_75 = pd.read_csv(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\CaSupp_75_radiomics_spine_lesions_features.csv")
c_100 = pd.read_csv(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\CaSupp_100_radiomics_spine_lesions_features.csv")

raw_datasets = {
    'c_25': c_25,
    'c_50': c_50,
    'c_75': c_75,
    'c_100': c_100
}


CompareDatasets(raw_datasets).kernel_density_plot()