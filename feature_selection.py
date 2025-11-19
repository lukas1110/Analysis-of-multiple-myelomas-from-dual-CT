import re
import os
import glob
import numpy as np
import pandas as pd
# import seaborn as sns
from functools import reduce
from kneed import KneeLocator
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Lasso
# from feature_engine.selection import MRMR
from scipy.stats import spearmanr, kruskal
from collections import defaultdict
# from mrmr import mrmr_classif, mrmr_regression
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
# from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error


class Dataset:
    def __init__(self, base_path: str, clinical_path: str) -> None:
        self.base_path = base_path
        self.clinical_path = clinical_path
        self.clinical_df = self._clinical_stage()

    def _read_clinical(self) -> pd.DataFrame:
        return pd.read_csv(self.clinical_path, encoding='cp1252')

    def _clinical_stage(self) -> pd.DataFrame:
        pd.set_option('future.no_silent_downcasting', True)
        clinical_df = self._read_clinical()
        clinical_df['Stage'] = clinical_df['ISS classification'].replace(
            {'Stage 1': int(1), 'Stage 2': int(2), 'Stage 3': int(3)})
        return clinical_df.dropna(subset=['Stage'])

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=['number']).columns.tolist()

    @staticmethod
    def _significant_features(df: pd.DataFrame) -> list[str]:
        return df.loc[df['p_value'] < 0.05, 'Feature'].tolist()

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df[self._numeric_features(df)]),
                            columns=df[self._numeric_features(df)].columns,
                            index=df[self._numeric_features(df)].index)

    @staticmethod
    def _extract_group(feature_name: str) -> str:
        patterns = [r"^gradient_firstorder", r"^gradient_glcm", r"^gradient_glrlm",
                    r"^gradient_glszm", r"^gradient_gldm", r"^gradient_ngtdm",
                    r"^original_firstorder", r"^original_glcm", r"^original_glrlm",
                    r"^original_glszm", r"^original_gldm", r"^original_ngtdm", r"^shape"]

        for p in patterns:
            if re.match(p, feature_name):
                return p.replace("^", "")
        return "other"

    def _all_csv(self) -> list[str]:
        return glob.glob(os.path.join(self.base_path, "*", "*spine_lesions*.csv"))

    def merged_dfs(self) -> dict[str, pd.DataFrame]:
        csv_dict, result_dict = defaultdict(list), {}
        [csv_dict[os.path.basename(f)].append(f) for f in self._all_csv()]

        for csv_name, files in csv_dict.items():
            dfs = []
            for f in files:
                df = pd.read_csv(f)
                patient_name = os.path.basename(os.path.dirname(f))
                df.insert(0, "patient", patient_name)
                dfs.append(df)

            key_name = csv_name.replace("_radiomics_spine_lesions_features.csv", "")
            result_dict[key_name] = pd.concat(dfs, ignore_index=True)
        return result_dict


class Spearman(Dataset):
    def __init__(self, base_path: str, clinical_path: str) -> None:
        super().__init__(base_path, clinical_path)
        self.merged = self.merged_dfs()
        self.statistic_dfs = self._spearman_dfs()

    def _spearman_dfs(self, clinical_marker: str='Beta2 microglobulin (mg/l)')-> dict[str, pd.DataFrame]:
        result_dict = {}
        for csv_name, df in self.merged.items():
            spearman_corr = {}
            for col in self._numeric_features(df):
                valid_idx = df[col].notna() & self.clinical_df[clinical_marker].notna()
                corr, p_value = spearmanr(self._standardize(df)[col][valid_idx],
                                          self._standardize(self.clinical_df)[clinical_marker][valid_idx])
                spearman_corr[col] = {'stat': corr, 'p_value': p_value}

            spearman_df = pd.DataFrame(spearman_corr).T
            result_dict[csv_name] = spearman_df.reset_index().rename(columns={'index': 'Feature'})
        return result_dict

    def _knee_threshold_selection(self, df: pd.DataFrame) -> tuple[KneeLocator, float | None, list]:
        importance = np.sort(np.abs(df.loc[df['Feature'].isin(self._significant_features(df)), 'stat'].values))[::-1]
        knee = KneeLocator(np.arange(1, len(importance) + 1), importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        best_features = df.loc[df['stat'].abs() > threshold, 'Feature'].tolist()

        return knee, threshold, best_features

    def _filtered_by_group(self) -> pd.DataFrame:
        filtered_dfs = [df[df['p_value'] < 0.05][['Feature', 'p_value']].rename(
            columns={'p_value': csv_name}) for csv_name, df in self.statistic_dfs.items()]

        merged_df = reduce(lambda left, right: pd.merge(
            left, right, on='Feature', how='outer'), filtered_dfs)
        merged_df['Group'] = merged_df['Feature'].apply(self._extract_group)

        return merged_df.sort_values(['Group', 'Feature']).reset_index(drop=True)

    def best_features(self, plot: bool=False) -> dict[str, list]:
        result_dict = {}
        for csv_name, df in self.statistic_dfs.items():
            knee, threshold, best_features = self._knee_threshold_selection(df)
            if plot: pass
            result_dict[csv_name] = best_features
        return result_dict

    def best_feature_from_group(self) -> dict[str, list]:
        image_columns = [col for col in self._filtered_by_group().columns if col not in ["Feature", "Group"]]

        result_dict = {}
        for image in image_columns:
            selected_features = []
            for group, group_df in self._filtered_by_group().groupby("Group"):
                valid_rows = group_df.dropna(subset=[image])
                if valid_rows.empty:
                    continue
                best_feature = valid_rows.loc[valid_rows[image].idxmin(), "Feature"]
                selected_features.append(best_feature)
            result_dict[image] = selected_features
        return result_dict


class KruskalWallis(Spearman):
    def __init__(self, base_path: str, clinical_path: str) -> None:
        super().__init__(base_path, clinical_path)
        self.merged = self.merged_dfs()
        self.statistic_dfs = self._kw_dfs()

    def _kw_dfs(self):
        results_dict = {}
        for csv_name, df in self.merged.items():
            results = []
            for feature in self._numeric_features(df):
                data = pd.concat([df[feature], self.clinical_df['Stage']], axis=1).dropna()
                groups = [data.loc[data['Stage'] == i, feature] for i in [1, 2, 3]]

                stat_all, p_value = kruskal(*groups)
                results.append({"Feature": feature, "stat": stat_all, "p_value": p_value,})
            results_dict[csv_name] = pd.DataFrame(results)
        return results_dict











base_dir_path = r"D:\DATA_Myelomy"
clinical_biomarkers_path = r"D:\Clinical_data\Table_clinical_data.csv"

