import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr, kruskal
from sklearn.preprocessing import StandardScaler


base_dir_path = r"D:\DATA_Myelomy"
clinical_biomarkers_path = r"D:\Clinical_data\Table_clinical_data.csv"


def return_numeric_features(df) -> list[str]:
    return df.select_dtypes(include=['number']).columns.tolist()


def return_significant_features(df) -> list[str]:
    return df.loc[df['p_value'] < 0.05, 'Feature'].tolist()


def standardize_df(df) -> pd.DataFrame:
    scaler = StandardScaler()
    numeric_features = return_numeric_features(df)
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_features]),
                                   columns=df[numeric_features].columns,
                                   index=df[numeric_features].index)
    return df_scaled


def merged_lesions_csv(dir_path) -> dict[str, pd.DataFrame]:
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


def get_spearman_csv(merged_csv, clinical_path,
                        clinical_column_name='Beta2 microglobulin (mg/l)') -> dict[str, pd.DataFrame]:
    clinical_biomarkers_df = pd.read_csv(clinical_path, encoding="cp1252")

    result_dict = {}
    for csv_name, csv_df in merged_csv.items():
        spearman_corr = {}
        for col in return_numeric_features(csv_df):
            valid_idx = csv_df[col].notna() & clinical_biomarkers_df[clinical_column_name].notna()
            corr, p_value = spearmanr(standardize_df(csv_df)[col][valid_idx],
                                      standardize_df(clinical_biomarkers_df)[clinical_column_name][valid_idx])
            spearman_corr[col] = {'spearman_corr': corr, 'p_value': p_value}

        spearman_df = pd.DataFrame(spearman_corr).T
        spearman_df = spearman_df.reset_index().rename(columns={'index': 'Feature'})
        result_dict[csv_name] = spearman_df
    return result_dict


def get_kruskal_wallis_csv(merged_csv, clinical_path) -> dict[str, pd.DataFrame]:
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


def filtered_features_spearman(merged_csv, spearman_csv,
                                      image_name=None, plot=False, threshold=0.75) -> dict[str, pd.DataFrame]:
    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}
        spearman_csv = {image_name: spearman_csv[image_name]}
    result_dict = {}
    for csv_name, (merged_df, spearman_df) in zip(merged_csv.keys(), zip(merged_csv.values(),
                                                                         spearman_csv.values())):
        significant_features = return_significant_features(spearman_df)
        remaining_features = list(significant_features)

        while True:
            corr_matrix = standardize_df(merged_df)[remaining_features].corr(method='spearman').abs()
            np.fill_diagonal(corr_matrix.values, 0)
            max_corr = corr_matrix.values.max()
            if max_corr <= threshold:
                break

            max_pos = np.unravel_index(np.argmax(corr_matrix.values), corr_matrix.shape)
            feat1 = corr_matrix.columns[max_pos[0]]
            feat2 = corr_matrix.columns[max_pos[1]]
            spearman1 = spearman_df.loc[spearman_df['Feature'] == feat1, 'spearman_corr'].values[0]
            spearman2 = spearman_df.loc[spearman_df['Feature'] == feat2, 'spearman_corr'].values[0]

            if abs(spearman1) >= abs(spearman2):
                to_remove = feat2
            else:
                to_remove = feat1
            remaining_features.remove(to_remove)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Filtration of Significant features based on Spearman "
                         f"for {csv_name} image with threshold {threshold}", fontsize=16, fontweight='bold')

            corr_matrix = standardize_df(merged_df).corr(method='spearman')
            im1 = axes[0].imshow(corr_matrix, cmap='gray', vmin=-1, vmax=1)
            axes[0].set_title(f"Correlation matrix\n(all significant features n={len(significant_features)})")

            final_corr_matrix = standardize_df(merged_df)[remaining_features].corr(method='spearman')
            _ = axes[1].imshow(final_corr_matrix, cmap='gray', vmin=-1, vmax=1)
            axes[1].set_title(f"Final correlation matrix\n(filtered features n={len(remaining_features)})")

            fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, label='Spearman correlation')
            plt.show()

        result_dict[csv_name] = remaining_features
    return result_dict


def filtered_features_kruskal_wallis(merged_csv, kruskal_wallis_csv,
                                            image_name=None, plot=False, threshold=0.75) -> dict[str, pd.DataFrame]:
    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}
        kruskal_wallis_csv = {image_name: kruskal_wallis_csv[image_name]}
    result_dict = {}
    for csv_name, (merged_df, kruskal_wallis_df) in zip(merged_csv.keys(), zip(merged_csv.values(),
                                                                               kruskal_wallis_csv.values())):
        significant_features = return_significant_features(kruskal_wallis_df)
        remaining_features = list(significant_features)

        while True:
            corr_matrix = standardize_df(merged_df)[remaining_features].corr(method='spearman').abs()
            np.fill_diagonal(corr_matrix.values, 0)
            max_corr = corr_matrix.values.max()
            if max_corr <= threshold:
                break

            max_pos = np.unravel_index(np.argmax(corr_matrix.values), corr_matrix.shape)
            feat1 = corr_matrix.columns[max_pos[0]]
            feat2 = corr_matrix.columns[max_pos[1]]
            kw1 = kruskal_wallis_df.loc[kruskal_wallis_df['Feature'] == feat1, 'H_all'].values[0]
            kw2 = kruskal_wallis_df.loc[kruskal_wallis_df['Feature'] == feat2, 'H_all'].values[0]

            if abs(kw1) >= abs(kw2):
                to_remove = feat2
            else:
                to_remove = feat1
            remaining_features.remove(to_remove)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Filtration of Significant features based on Kruskal Wallis "
                         f"for {csv_name} image with threshold {threshold}", fontsize=16, fontweight='bold')

            corr_matrix = standardize_df(merged_df).corr(method='spearman')
            im1 = axes[0].imshow(corr_matrix, cmap='gray', vmin=-1, vmax=1)
            axes[0].set_title(f"Correlation matrix\n(all significant features n={len(significant_features)})")

            final_corr_matrix = standardize_df(merged_df)[remaining_features].corr(method='spearman')
            _ = axes[1].imshow(final_corr_matrix, cmap='gray', vmin=-1, vmax=1)
            axes[1].set_title(f"Final correlation matrix\n(filtered features n={len(remaining_features)})")

            fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, label='Spearman correlation')
            plt.show()

        result_dict[csv_name] = remaining_features
    return result_dict
