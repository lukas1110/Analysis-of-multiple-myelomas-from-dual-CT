import re
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from feature_engine.selection import MRMR
from scipy.stats import spearmanr, kruskal
from collections import defaultdict
from mrmr import mrmr_classif, mrmr_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error


base_dir_path = r"D:\DATA_Myelomy"
clinical_biomarkers_path = r"D:\Clinical_data\Table_clinical_data.csv"

### DONE ###
def return_numeric_features(df) -> list[str]:
    return df.select_dtypes(include=['number']).columns.tolist()

### DONE ###
def return_significant_features(df) -> list[str]:
    return df.loc[df['p_value'] < 0.05, 'Feature'].tolist()

### DONE ###
def standardize_df(df) -> pd.DataFrame:
    scaler = StandardScaler()
    numeric_features = return_numeric_features(df)
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_features]),
                                   columns=df[numeric_features].columns,
                                   index=df[numeric_features].index)
    return df_scaled

### DONE ###
def extract_group(feature_name: str) -> str:
    patterns = [r"^gradient_firstorder", r"^gradient_glcm", r"^gradient_glrlm",
                r"^gradient_glszm", r"^gradient_gldm", r"^gradient_ngtdm",
                r"^original_firstorder", r"^original_glcm", r"^original_glrlm",
                r"^original_glszm", r"^original_gldm", r"^original_ngtdm", r"^shape"]

    for p in patterns:
        if re.match(p, feature_name):
            return p.replace("^", "")
    return "other"

### DONE ###
def add_stage_in_clinical_df(clinical_path) -> pd.DataFrame:
    pd.set_option('future.no_silent_downcasting', True)
    clinical_df = pd.read_csv(clinical_path, encoding="cp1252")
    clinical_df['Stage'] = clinical_df['ISS classification'].replace({'Stage 1': 1, 'Stage 2': 2, 'Stage 3': 3})
    clinical_df = clinical_df.dropna(subset=['Stage'])
    clinical_df['Stage'] = clinical_df['Stage'].astype(int)
    return clinical_df

### DONE ###
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

### DONE ###
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

### DONE ###
def get_kruskal_wallis_csv(merged_csv, clinical_path) -> dict[str, pd.DataFrame]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

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

### DONE ###
def selected_spearman_features(spearman_csv, plot=False) -> dict[str, list]:
    result_dict = {}
    for csv_name, spearman_df in spearman_csv.items():
        significant_features = return_significant_features(spearman_df)

        importance = np.sort(
            np.abs(spearman_df.loc[spearman_df['Feature'].isin(significant_features), 'spearman_corr'].values)
        )[::-1]
        x = np.arange(1, len(importance) + 1)

        knee = KneeLocator(x, importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        selected_features = spearman_df[spearman_df['spearman_corr'].abs() > threshold]['Feature'].tolist()

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(x, importance, marker='o', label='Feature spearman_corr')
            if threshold is not None:
                plt.axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
                plt.axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
            plt.title(f"Selected Features for {csv_name} based on spearman_corr")
            plt.xlabel("Feature rank (sorted by spearman_corr)")
            plt.ylabel("spearman_corr")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict

### DONE ###
def selected_kruskal_wallis_features(kruskal_wallis_csv, plot=False) -> dict[str, list]:
    result_dict = {}
    for csv_name, kruskal_wallis_df in kruskal_wallis_csv.items():
        significant_features = return_significant_features(kruskal_wallis_df)

        importance = np.sort(
            np.abs(kruskal_wallis_df.loc[kruskal_wallis_df['Feature'].isin(significant_features), 'H_all'].values)
        )[::-1]
        x = np.arange(1, len(importance) + 1)

        knee = KneeLocator(x, importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        selected_features = kruskal_wallis_df[kruskal_wallis_df['H_all'].abs() > threshold]['Feature'].tolist()

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(x, importance, marker='o', label='Feature H_all')
            if threshold is not None:
                plt.axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
                plt.axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
            plt.title(f"Selected Features for {csv_name} based on H_all")
            plt.xlabel("Feature rank (sorted by H_all)")
            plt.ylabel("H_all")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict

### DONE ###
def best_feature_from_each_feature_group(significant_csv):
    filtered_dfs = []
    for csv_name, significant_df in significant_csv.items():
        temp = significant_df[significant_df['p_value'] < 0.05][['Feature', 'p_value']].copy()
        temp.rename(columns={'p_value': csv_name}, inplace=True)
        filtered_dfs.append(temp)

    merged_df = filtered_dfs[0]
    for temp in filtered_dfs[1:]:
        merged_df = pd.merge(merged_df, temp, on='Feature', how='outer')

    merged_df = merged_df.sort_values(by='Feature').reset_index(drop=True)
    merged_df['Group'] = merged_df['Feature'].apply(extract_group)
    merged_df = merged_df.sort_values(by=['Group', 'Feature']).reset_index(drop=True)

    image_columns = [col for col in merged_df.columns if col not in ["Feature", "Group"]]

    result_dict = {}
    for image in image_columns:
        selected_features = []
        for group, group_df in merged_df.groupby("Group"):

            valid_rows = group_df.dropna(subset=[image])
            if valid_rows.empty:
                continue

            best_feature = valid_rows.loc[valid_rows[image].idxmin(), "Feature"]
            selected_features.append(best_feature)
        result_dict[image] = selected_features
    return result_dict


def filtered_features_spearman(merged_csv, spearman_csv,
                                      image_name=None, plot=False, threshold=0.75) -> dict[str, list]:
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

            corr_matrix = standardize_df(merged_df)[significant_features].corr(method='spearman')
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
                                     image_name=None, plot=False, threshold=0.75) -> dict[str, list]:
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

            corr_matrix = standardize_df(merged_df)[significant_features].corr(method='spearman')
            im1 = axes[0].imshow(corr_matrix, cmap='gray', vmin=-1, vmax=1)
            axes[0].set_title(f"Correlation matrix\n(all significant features n={len(significant_features)})")

            final_corr_matrix = standardize_df(merged_df)[remaining_features].corr(method='spearman')
            _ = axes[1].imshow(final_corr_matrix, cmap='gray', vmin=-1, vmax=1)
            axes[1].set_title(f"Final correlation matrix\n(filtered features n={len(remaining_features)})")

            fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, label='Spearman correlation')
            plt.show()

        result_dict[csv_name] = remaining_features
    return result_dict


def random_forest_classifier_selected_features(merged_csv, clinical_path,
                                    image_name=None, plot=False, test_size=0.2, n_trees=200) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}

    result_dict = {}
    for csv_name, merged_df in merged_csv.items():
        features = merged_df.select_dtypes(include=['number'])
        labels = clinical_df['Stage']

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features = features[valid_idx]
        labels = labels[valid_idx]

        features_train, features_test, labels_train, labels_test = (
            train_test_split(features, labels, test_size=test_size,random_state=42, stratify=labels))

        rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf.fit(features_train, labels_train)

        labels_pred = rf.predict(features_test)
        acc = accuracy_score(labels_test, labels_pred)

        feature_importance_df = (pd.DataFrame({'Feature': features.columns, 'Importance': rf.feature_importances_}
                                              ).sort_values(by='Importance', ascending=False))

        importance = np.sort(feature_importance_df['Importance'].values)[::-1]
        x = np.arange(1, len(importance) + 1)

        knee = KneeLocator(x, importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Selected features from all possible features based on Random Forest "
                         f"for {csv_name} with accuracy {acc:.3f}", fontsize=16, fontweight='bold')

            cm = confusion_matrix(labels_test, labels_pred)

            class_accuracies = []
            for i in range(len(cm)):
                total = cm[i].sum()
                correct = cm[i, i]
                acc = correct / total if total > 0 else 0
                class_accuracies.append(acc)

            accuracy_text = " | ".join([
                f"Stage {i + 1}: {class_accuracies[i] * 100:.1f}%"
                for i in range(len(class_accuracies))
            ])

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[1, 2, 3], yticklabels=[1, 2, 3], ax=axes[0], cbar=False)
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("True label")
            axes[0].set_title(f"Confusion Matrix\n{accuracy_text}")

            axes[1].plot(x, importance, marker='o', label='Feature importance', color='tab:blue')
            if threshold is not None:
                axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
                axes[1].axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
            axes[1].set_title("Feature Importance Curve")
            axes[1].set_xlabel("Feature rank (sorted by importance)")
            axes[1].set_ylabel("Importance")
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def random_forest_regressor_selected_features(merged_csv, clinical_path,
                                    image_name=None, plot=False, test_size=0.2, n_trees=200) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}

    result_dict = {}
    for csv_name, merged_df in merged_csv.items():
        features = merged_df.select_dtypes(include=['number'])
        labels = clinical_df['Beta2 microglobulin (mg/l)']

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features = features[valid_idx]
        labels = labels[valid_idx]

        features_train, features_test, labels_train, labels_test = (
            train_test_split(features, labels, test_size=test_size,random_state=42))

        rf = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        rf.fit(features_train, labels_train)

        labels_pred = rf.predict(features_test)
        r2 = r2_score(labels_test, labels_pred)
        mse = mean_squared_error(labels_test, labels_pred)
        mae = mean_absolute_error(labels_test, labels_pred)

        feature_importance_df = (pd.DataFrame({'Feature': features.columns, 'Importance': rf.feature_importances_}
                                              ).sort_values(by='Importance', ascending=False))

        importance = np.sort(feature_importance_df['Importance'].values)[::-1]
        x = np.arange(1, len(importance) + 1)

        knee = KneeLocator(x, importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Selected features from all possible features based on Random Forest\n"
                         f"for {csv_name} with R²: {r2:.3f} || MSE: {mse:.3f} || MAE: {mae:.3f}", fontsize=16,
                         fontweight='bold')

            axes[0].scatter(labels_test, labels_pred, alpha=0.7)
            axes[0].plot([labels_test.min(), labels_test.max()],
                         [labels_test.min(), labels_test.max()],
                         'r--', lw=2)
            axes[0].set_xlabel("True values")
            axes[0].set_ylabel("Predicted values")
            axes[0].set_title("True vs Predicted (Regression)")

            axes[1].plot(x, importance, marker='o', label='Feature importance', color='tab:blue')
            if threshold is not None:
                axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
                axes[1].axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
            axes[1].set_title("Feature Importance Curve")
            axes[1].set_xlabel("Feature rank (sorted by importance)")
            axes[1].set_ylabel("Importance")
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def random_forest_classifier_significant_selected_features(merged_csv, kruskal_wallis_csv, clinical_path, image_name=None,
                                                plot=False, test_size=0.2, n_trees=200) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}
        kruskal_wallis_csv = {image_name: kruskal_wallis_csv[image_name]}

    result_dict = {}
    for csv_name, (merged_df, kruskal_wallis_df) in zip(merged_csv.keys(), zip(merged_csv.values(),
                                                                               kruskal_wallis_csv.values())):
        significant_features = return_significant_features(kruskal_wallis_df)
        features = merged_df[significant_features].select_dtypes(include=['number'])
        labels = clinical_df['Stage']

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features = features[valid_idx]
        labels = labels[valid_idx]

        features_train, features_test, labels_train, labels_test = (
            train_test_split(features, labels, test_size=test_size, random_state=42, stratify=labels))

        rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf.fit(features_train, labels_train)

        labels_pred = rf.predict(features_test)
        acc = accuracy_score(labels_test, labels_pred)

        feature_importance_df = (pd.DataFrame({'Feature': features.columns, 'Importance': rf.feature_importances_}
                                              ).sort_values(by='Importance', ascending=False))

        importance = np.sort(feature_importance_df['Importance'].values)[::-1]
        x = np.arange(1, len(importance) + 1)

        knee = KneeLocator(x, importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Selected features from Significant features (K-W test) based on Random Forest "
                         f"for {csv_name} with accuracy {acc:.3f}", fontsize=16, fontweight='bold')

            cm = confusion_matrix(labels_test, labels_pred)

            class_accuracies = []
            for i in range(len(cm)):
                total = cm[i].sum()
                correct = cm[i, i]
                acc = correct / total if total > 0 else 0
                class_accuracies.append(acc)

            accuracy_text = " | ".join([
                f"Stage {i + 1}: {class_accuracies[i] * 100:.1f}%"
                for i in range(len(class_accuracies))
            ])

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[1, 2, 3], yticklabels=[1, 2, 3], ax=axes[0], cbar=False)
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("True label")
            axes[0].set_title(f"Confusion Matrix\n{accuracy_text}")

            axes[1].plot(x, importance, marker='o', label='Feature importance', color='tab:blue')
            if threshold is not None:
                axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
                axes[1].axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
            axes[1].set_title("Feature Importance Curve")
            axes[1].set_xlabel("Feature rank (sorted by importance)")
            axes[1].set_ylabel("Importance")
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def random_forest_regressor_significant_selected_features(merged_csv, spearman_csv, clinical_path, image_name=None,
                                                plot=False, n_trees=200) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}
        spearman_csv = {image_name: spearman_csv[image_name]}

    result_dict = {}
    for csv_name, (merged_df, spearman_df) in zip(merged_csv.keys(), zip(merged_csv.values(),
                                                                               spearman_csv.values())):
        significant_features = return_significant_features(spearman_df)
        features = merged_df[significant_features].select_dtypes(include=['number'])
        labels = clinical_df['Beta2 microglobulin (mg/l)']

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features = features[valid_idx]
        labels = labels[valid_idx]

        rf = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        rf.fit(features, labels)

        labels_pred = rf.predict(features)
        r2 = r2_score(labels, labels_pred)
        mse = mean_squared_error(labels, labels_pred)
        mae = mean_absolute_error(labels, labels_pred)

        feature_importance_df = (pd.DataFrame({'Feature': features.columns, 'Importance': rf.feature_importances_}
                                              ).sort_values(by='Importance', ascending=False))

        importance = np.sort(feature_importance_df['Importance'].values)[::-1]
        x = np.arange(1, len(importance) + 1)

        knee = KneeLocator(x, importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(
                f"Selected features from Significant features (Spearman) based on Random Forest\n"
                f"for {csv_name} with R²: {r2:.3f}  |  MSE: {mse:.3f}  |  MAE: {mae:.3f}", fontsize=16,
                fontweight='bold')

            axes[0].scatter(labels, labels_pred, alpha=0.7)
            axes[0].plot([labels.min(), labels.max()],
                         [labels.min(), labels.max()],
                         'r--', lw=2)
            axes[0].set_xlabel("True values")
            axes[0].set_ylabel("Predicted values")
            axes[0].set_title("True vs Predicted (Regression)")

            axes[1].plot(x, importance, marker='o', label='Feature importance', color='tab:blue')
            if threshold is not None:
                axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
                axes[1].axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
            axes[1].set_title("Feature Importance Curve")
            axes[1].set_xlabel("Feature rank (sorted by importance)")
            axes[1].set_ylabel("Importance")
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def mutual_information_selected_features(merged_csv, clinical_path, clinical_column_name='Beta2 microglobulin (mg/l)',
                                image_name=None, plot=False) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}

    result_dict = {}
    for csv_name, merged_df in merged_csv.items():
        numeric_features = return_numeric_features(merged_df)

        mutual_info_list = []
        for col in numeric_features:
            valid_idx = merged_df[col].notna() & clinical_df[clinical_column_name].notna()

            feature = merged_df[[col]][valid_idx]
            labels = clinical_df[clinical_column_name][valid_idx]

            if clinical_column_name == 'Stage':
                mi = mutual_info_classif(feature, labels, random_state=42)[0]
            else:
                mi = mutual_info_regression(feature, labels, random_state=42)[0]
            mutual_info_list.append({'Feature': col, 'MI': mi})
        mutual_information_df = pd.DataFrame(mutual_info_list).sort_values(by='MI',
                                                                           ascending=False).reset_index(drop=True)

        importance = np.sort(mutual_information_df['MI'].values)[::-1]
        x = np.arange(1, len(importance) + 1)

        knee = KneeLocator(x, importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        selected_features = mutual_information_df[mutual_information_df['MI'] > threshold]['Feature'].tolist()

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(x, importance, marker='o', label='Feature MI')
            if threshold is not None:
                plt.axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
                plt.axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
            plt.title(f"Mutual information curve for {csv_name} based on clinical: {clinical_column_name}")
            plt.xlabel("Feature rank (sorted by MI)")
            plt.ylabel("MI")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def lasso_regression_selected_features(merged_csv, clinical_path, clinical_column_name='Beta2 microglobulin (mg/l)',
                                       alpha=0.01, image_name=None, plot=False) -> dict[str, list]:
    clinical_df = pd.read_csv(clinical_path, encoding="cp1252")
    clinical_scaled = standardize_df(clinical_df)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}

    result_dict = {}
    for csv_name, merged_df in merged_csv.items():
        numeric_features = return_numeric_features(merged_df)
        features_scaled = standardize_df(merged_df)

        valid_idx = clinical_scaled[clinical_column_name].notna()
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=5000)

        features = features_scaled[numeric_features][valid_idx]
        labels = clinical_scaled[clinical_column_name][valid_idx]

        lasso.fit(features, labels)
        labels_pred = lasso.predict(features)

        r2 = r2_score(labels, labels_pred)
        mse = mean_squared_error(labels, labels_pred)
        mae = mean_absolute_error(labels, labels_pred)

        feature_weights_df = pd.DataFrame({'Feature': features.columns, 'Weight': lasso.coef_}
                                          ).sort_values(by='Weight',ascending=False).reset_index(drop=True)

        weights = np.sort(np.abs(feature_weights_df['Weight'].values))[::-1]
        x = np.arange(1, len(weights) + 1)

        knee = KneeLocator(x, weights, curve='convex', direction='decreasing')
        threshold = weights[knee.knee] if knee.knee is not None else None
        selected_features = feature_weights_df[feature_weights_df['Weight'].abs() > threshold]['Feature'].tolist()

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(
                f"Selected features from Lasso for {csv_name} with \n"
                f"R²: {r2:.3f}  |  MSE: {mse:.3f}  |  MAE: {mae:.3f}", fontsize=16, fontweight='bold')

            axes[0].scatter(labels, labels_pred, alpha=0.7)
            axes[0].plot([labels.min(), labels.max()],
                         [labels.min(), labels.max()],
                         'r--', lw=2)
            axes[0].set_xlabel("True values")
            axes[0].set_ylabel("Predicted values")
            axes[0].set_title("True vs Predicted (Regression)")

            axes[1].plot(x, weights, marker='o', label='Feature weights', color='tab:blue')
            if threshold is not None:
                axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
                axes[1].axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
            axes[1].set_title("Feature Weights Curve")
            axes[1].set_xlabel("Feature rank (sorted by importance)")
            axes[1].set_ylabel("Weights")
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def lasso_regression_significant_selected_features(merged_csv, spearman_csv, clinical_path,
                                                   clinical_column_name='Beta2 microglobulin (mg/l)',
                                                   alpha=0.01, image_name=None, plot=False) -> dict[str, list]:
    clinical_df = pd.read_csv(clinical_path, encoding="cp1252")
    clinical_scaled = standardize_df(clinical_df)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}
        spearman_csv = {image_name: spearman_csv[image_name]}

    result_dict = {}
    for csv_name, (merged_df, spearman_df) in zip(merged_csv.keys(), zip(merged_csv.values(),
                                                                         spearman_csv.values())):
        significant_features = return_significant_features(spearman_df)
        features_scaled = standardize_df(merged_df)

        valid_idx = clinical_scaled[clinical_column_name].notna()
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=5000)

        features = features_scaled[significant_features][valid_idx]
        labels = clinical_scaled[clinical_column_name][valid_idx]

        lasso.fit(features, labels)
        labels_pred = lasso.predict(features)

        r2 = r2_score(labels, labels_pred)
        mse = mean_squared_error(labels, labels_pred)
        mae = mean_absolute_error(labels, labels_pred)

        feature_weights_df = pd.DataFrame({'Feature': features.columns, 'Weight': lasso.coef_}
                                          ).sort_values(by='Weight', ascending=False).reset_index(drop=True)

        weights = np.sort(np.abs(feature_weights_df['Weight'].values))[::-1]
        x = np.arange(1, len(weights) + 1)

        knee = KneeLocator(x, weights, curve='convex', direction='decreasing')
        threshold = weights[knee.knee] if knee.knee is not None else None
        selected_features = feature_weights_df[feature_weights_df['Weight'].abs() > threshold]['Feature'].tolist()

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(
                f"Significant (Spearman) Selected features from Lasso for {csv_name} with \n"
                f"R²: {r2:.3f}  |  MSE: {mse:.3f}  |  MAE: {mae:.3f}", fontsize=16, fontweight='bold')

            axes[0].scatter(labels, labels_pred, alpha=0.7)
            axes[0].plot([labels.min(), labels.max()],
                         [labels.min(), labels.max()],
                         'r--', lw=2)
            axes[0].set_xlabel("True values")
            axes[0].set_ylabel("Predicted values")
            axes[0].set_title("True vs Predicted (Regression)")

            axes[1].plot(x, weights, marker='o', label='Feature weights', color='tab:blue')
            if threshold is not None:
                axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
                axes[1].axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
            axes[1].set_title("Feature Weights Curve")
            axes[1].set_xlabel("Feature rank (sorted by importance)")
            axes[1].set_ylabel("Weights")
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def mrmr_classification_selected_features(merged_csv, clinical_path, n_top_features=20,
                                          plot=False, image_name=None) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}

    result_dict = {}
    for csv_name, merged_df in merged_csv.items():
        temp_df = pd.merge(merged_df, clinical_df[['Patient ID', 'Stage']],
                             left_on='patient', right_on='Patient ID', how='inner')

        numeric_features = return_numeric_features(temp_df)
        features = temp_df[numeric_features].drop(columns=['Stage'], errors='ignore')
        labels = temp_df['Stage']

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features, labels = features[valid_idx], labels[valid_idx]

        df_mrmr = standardize_df(features).copy()
        df_mrmr['Stage'] = labels.values
        selected_features = mrmr_classif(X=df_mrmr.drop(columns='Stage'), y=df_mrmr['Stage'], K=n_top_features)

        if plot:
            selected_df = standardize_df(features)[selected_features]
            corr_matrix = selected_df.corr(method='spearman')
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
            plt.title(f"Spearman correlation between MRMR-selected features for {csv_name}")
            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def mrmr_classification_significant_selected_features(merged_csv, kruskal_wallis_csv, clinical_path,
                                                      n_top_features=20, plot=False, image_name=None) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}
        kruskal_wallis_csv = {image_name: kruskal_wallis_csv[image_name]}

    result_dict = {}
    for csv_name, (merged_df, kruskal_wallis_df) in zip(merged_csv.keys(), zip(merged_csv.values(),
                                                                         kruskal_wallis_csv.values())):
        temp_df = pd.merge(merged_df, clinical_df[['Patient ID', 'Stage']],
                           left_on='patient', right_on='Patient ID', how='inner')

        significant_features = return_significant_features(kruskal_wallis_df)
        features = temp_df[significant_features].drop(columns=['Stage'], errors='ignore')
        labels = temp_df['Stage']

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features, labels = features[valid_idx], labels[valid_idx]

        df_mrmr = standardize_df(features).copy()
        df_mrmr['Stage'] = labels.values
        selected_features = mrmr_classif(X=df_mrmr.drop(columns='Stage'), y=df_mrmr['Stage'], K=n_top_features)

        if plot:
            selected_df = standardize_df(features)[selected_features]
            corr_matrix = selected_df.corr(method='spearman')
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
            plt.title(f"Spearman correlation between MRMR-selected Significant features for {csv_name}")
            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def mrmr_fe_classification_selected_features(merged_csv, clinical_path, n_top_features=20,
                                             plot=False, image_name=None) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}

    result_dict = {}
    for csv_name, merged_df in merged_csv.items():
        temp_df = pd.merge(merged_df, clinical_df[['Patient ID', 'Stage']],
                           left_on='patient', right_on='Patient ID', how='inner')

        numeric_features = return_numeric_features(temp_df)
        features = temp_df[numeric_features].drop(columns=['Stage'], errors='ignore')
        labels = temp_df['Stage']

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features, labels = features[valid_idx], labels[valid_idx]

        selector = MRMR(method='MIQ', max_features=n_top_features)
        selector.fit(standardize_df(features), labels)
        selected_features = selector.transform(standardize_df(features)).columns.tolist()

        if plot:
            selected_df = standardize_df(features)[selected_features]
            corr_matrix = selected_df.corr(method='spearman')
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
            plt.title(f"Spearman correlation between MRMR-selected features for {csv_name}")
            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def mrmr_fe_classification_significant_selected_features(merged_csv, kruskal_wallis_csv, clinical_path,
                                                         n_top_features=20, plot=False, image_name=None) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}
        kruskal_wallis_csv = {image_name: kruskal_wallis_csv[image_name]}

    result_dict = {}
    for csv_name, (merged_df, kruskal_wallis_df) in zip(merged_csv.keys(), zip(merged_csv.values(),
                                                                               kruskal_wallis_csv.values())):
        temp_df = pd.merge(merged_df, clinical_df[['Patient ID', 'Stage']],
                           left_on='patient', right_on='Patient ID', how='inner')

        significant_features = return_significant_features(kruskal_wallis_df)
        features = temp_df[significant_features].drop(columns=['Stage'], errors='ignore')
        labels = temp_df['Stage']

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features, labels = features[valid_idx], labels[valid_idx]

        selector = MRMR(method='MIQ', max_features=n_top_features)
        selector.fit(standardize_df(features), labels)
        selected_features = selector.transform(standardize_df(features)).columns.tolist()

        if plot:
            selected_df = standardize_df(features)[selected_features]
            corr_matrix = selected_df.corr(method='spearman')
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
            plt.title(f"Spearman correlation between MRMR-selected Significant features for {csv_name}")
            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def mrmr_regression_selected_features(merged_csv, clinical_path, clinical_column_name='Beta2 microglobulin (mg/l)',
                                      n_top_features=20, plot=False, image_name=None) -> dict[str, list]:
    clinical_df = pd.read_csv(clinical_path, encoding="cp1252")

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}

    result_dict = {}
    for csv_name, merged_df in merged_csv.items():
        temp_df = pd.merge(merged_df, clinical_df[['Patient ID', clinical_column_name]],
                           left_on='patient', right_on='Patient ID', how='inner')

        numeric_features = return_numeric_features(temp_df)
        features = temp_df[numeric_features].drop(columns=[clinical_column_name], errors='ignore')
        labels = temp_df[clinical_column_name]

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features, labels = features[valid_idx], labels[valid_idx]

        df_mrmr = standardize_df(features).copy()
        df_mrmr[clinical_column_name] = labels.values
        selected_features = mrmr_regression(X=df_mrmr.drop(columns=clinical_column_name),
                                            y=df_mrmr[clinical_column_name], K=n_top_features)

        if plot:
            selected_df = standardize_df(features)[selected_features]
            corr_matrix = selected_df.corr(method='spearman')
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
            plt.title(f"Spearman correlation between MRMR-selected features for {csv_name}")
            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict


def mrmr_regression_significant_selected_features(merged_csv, spearman_csv, clinical_path,
                                                  clinical_column_name='Beta2 microglobulin (mg/l)',
                                                  n_top_features=20, plot=False, image_name=None) -> dict[str, list]:
    clinical_df = add_stage_in_clinical_df(clinical_path)

    if image_name is not None:
        merged_csv = {image_name: merged_csv[image_name]}
        spearman_csv = {image_name: spearman_csv[image_name]}

    result_dict = {}
    for csv_name, (merged_df, spearman_df) in zip(merged_csv.keys(), zip(merged_csv.values(),
                                                                         spearman_csv.values())):
        temp_df = pd.merge(merged_df, clinical_df[['Patient ID', clinical_column_name]],
                           left_on='patient', right_on='Patient ID', how='inner')

        significant_features = return_significant_features(spearman_df)
        features = temp_df[significant_features].drop(columns=[clinical_column_name], errors='ignore')
        labels = temp_df[clinical_column_name]

        valid_idx = features.notna().all(axis=1) & labels.notna()
        features, labels = features[valid_idx], labels[valid_idx]

        df_mrmr = standardize_df(features).copy()
        df_mrmr[clinical_column_name] = labels.values
        selected_features = mrmr_regression(X=df_mrmr.drop(columns=clinical_column_name),
                                            y=df_mrmr[clinical_column_name], K=n_top_features)

        if plot:
            selected_df = standardize_df(features)[selected_features]
            corr_matrix = selected_df.corr(method='spearman')
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False)
            plt.title(f"Spearman correlation between MRMR-selected Significant features for {csv_name}")
            plt.tight_layout()
            plt.show()

        result_dict[csv_name] = selected_features
    return result_dict
