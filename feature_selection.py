import os
import glob
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr, kruskal
from sklearn.preprocessing import StandardScaler


base_dir_path = r"D:\DATA_Myelomy"
clinical_biomarkers_path = r"D:\Clinical_data\Table_clinical_data.csv"


def return_numeric_features(df):
    return df.select_dtypes(include=['number']).columns


def return_significant_features(df):
    return df.loc[df['p_value'] < 0.05, 'Feature']


def standardize_df(df):
    scaler = StandardScaler()
    numeric_features = return_numeric_features(df)
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_features]),
                                   columns=df[numeric_features].columns,
                                   index=df[numeric_features].index)
    return df_scaled


def return_merged_lesions_csv(dir_path):
    all_csvs = glob.glob(os.path.join(dir_path, "*", "*spine_lesions*.csv"))
    csv_dict = defaultdict(list)
    result_dict = {}
    for file in all_csvs:
        csv_name = os.path.basename(file)
        csv_dict[csv_name].append(file)

    for csv_name, files in csv_dict.items():
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            patient_name = os.path.basename(os.path.dirname(file))
            df.insert(0, "patient", patient_name)
            dfs.append(df)

        key_name = csv_name.replace("_radiomics_spine_lesions_features.csv", "")
        big_df = pd.concat(dfs, ignore_index=True)
        result_dict[key_name] = big_df
    return result_dict


def return_spearman_csv(merged_csv, clinical_path, clinical_column_name):
    clinical_biomarkers_df = pd.read_csv(clinical_path, encoding="cp1252")

    result_dict = {}
    for csv_name, csv_df in merged_csv.items():
        spearman_corr = {}
        for col in return_numeric_features(csv_df):
            valid_idx = standardize_df(csv_df)[col].notna() & clinical_biomarkers_df[clinical_column_name].notna()
            corr, p_value = spearmanr(standardize_df(csv_df)[col][valid_idx], clinical_biomarkers_df[clinical_column_name][valid_idx])
            spearman_corr[col] = {'spearman_corr': corr, 'p_value': p_value}

        spearman_df = pd.DataFrame(spearman_corr).T
        spearman_df = spearman_df.reset_index().rename(columns={'index': 'Feature'})
        result_dict[csv_name] = spearman_df
    return result_dict


def return_kruskal_wallis_csv(merged_csv, clinical_path):
    pd.set_option('future.no_silent_downcasting', True)
    clinical_df = pd.read_csv(clinical_path, encoding="cp1252")
    clinical_df['Stage'] = clinical_df['ISS classification'].replace({'Stage 1': 1, 'Stage 2': 2, 'Stage 3': 3})

    results_dict = {}
    for csv_name, csv_df in merged_csv.items():
        results = []
        for feature in return_numeric_features(csv_df):

            data = pd.concat([csv_df[feature], clinical_df['Stage']], axis=1).dropna()
            groups = [data.loc[data['Stage'] == i, feature] for i in [1, 2, 3]]

            stat_all, p_value = kruskal(*groups)
            p12 = kruskal(groups[0], groups[1]).pvalue
            p13 = kruskal(groups[0], groups[2]).pvalue
            p23 = kruskal(groups[1], groups[2]).pvalue

            results.append({
                "Feature": feature,
                "H_all": stat_all,
                "p_value": p_value,
                "p(1-2)": p12,
                "p(1-3)": p13,
                "p(2-3)": p23,
            })
        kruskal_wallis_df = pd.DataFrame(results)
        results_dict[csv_name] = kruskal_wallis_df
    return results_dict





