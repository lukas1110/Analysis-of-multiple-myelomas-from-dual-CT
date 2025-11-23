import re
import os
import glob
import numpy as np
import pandas as pd
# import seaborn as sns
from functools import reduce
from kneed import KneeLocator
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
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



class VisualizationManager:
    # TODO: Visualization of filtered features based on statistical value
    # TODO: Visualization of RF
    # TODO: Visualization of LASSO
    # TODO: Visualization of MI
    # TODO: Visualization of MRMR

    @staticmethod
    def plot_stat_selection(df: pd.DataFrame, name: str) -> None:
        selector = ThresholdSelectionFeatures(df)
        knee, threshold, _, importance = selector.select_features_by_knee()

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(importance) + 1), importance, marker='o', label='Feature importance')

        if threshold is not None:
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold: {threshold:.4f}')
            plt.axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature: {knee.knee}')

        plt.title(f"Selected Features for {name}")
        plt.xlabel("Feature rank")
        plt.ylabel("Importance")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_filtered_features(
            merged_df: pd.DataFrame,
            significant_features: list[str],
            remaining_features: list[str],
            name: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Filtration of Significant features for {name}", fontsize=16, fontweight='bold')

        corr_matrix = (merged_df[significant_features].corr(method='spearman'))
        im1 = axes[0].imshow(corr_matrix, cmap='gray', vmin=-1, vmax=1)
        axes[0].set_title(f"All significant features (n={len(significant_features)})")

        final_corr_matrix = merged_df[remaining_features].corr(method='spearman')
        axes[1].imshow(final_corr_matrix, cmap='gray', vmin=-1, vmax=1)
        axes[1].set_title(f"Filtered features (n={len(remaining_features)})")

        fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, label='Spearman correlation')
        plt.show()


class ThresholdSelectionFeatures:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def _get_significant_features(self) -> list[str]:
        return self.df.loc[self.df['p_value'] < 0.05, 'Feature'].tolist()

    def _compute_importance(self) -> np.ndarray:
        sig_features = self._get_significant_features()
        importance = np.abs(self.df.loc[self.df['Feature'].isin(sig_features), 'stat'].values)
        return np.sort(importance)[::-1]

    @staticmethod
    def _find_knee(importance: np.ndarray) -> tuple[KneeLocator, float | None]:
        x = np.arange(1, len(importance) + 1)
        knee = KneeLocator(x, importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        return knee, threshold

    def select_features_by_knee(self) -> tuple[KneeLocator, float | None, list[str], np.ndarray]:
        importance = self._compute_importance()
        knee, threshold = self._find_knee(importance)
        best_features = self.df.loc[
            self.df['stat'].abs() > (threshold if threshold is not None else 0), 'Feature'].tolist()
        return knee, threshold, best_features, importance


class DatasetHelper:
    def __init__(self, base_path: str, clinical_path: str) -> None:
        self.base_path = base_path
        self.clinical_path = clinical_path

    def _read_clinical(self) -> pd.DataFrame:
        return pd.read_csv(self.clinical_path, encoding='cp1252')

    def _add_stage(self) -> pd.DataFrame:
        df = self._read_clinical()
        df['Stage'] = df['ISS classification'].replace(
            {'Stage 1': int(1), 'Stage 2': int(2), 'Stage 3': int(3)})
        return df.dropna(subset=['Stage'])

    @staticmethod
    def _extract_feature_group(feature_name: str) -> str:
        patterns = [
            r"^gradient_firstorder", r"^gradient_glcm", r"^gradient_glrlm",
            r"^gradient_glszm", r"^gradient_gldm", r"^gradient_ngtdm",
            r"^original_firstorder", r"^original_glcm", r"^original_glrlm",
            r"^original_glszm", r"^original_gldm", r"^original_ngtdm",
            r"^shape"]
        for p in patterns:
            if re.match(p, feature_name):
                return p.replace("^", "")
        return "other"

    def _all_csv(self) -> list[str]:
        return glob.glob(os.path.join(self.base_path, "*", "*spine_lesions*.csv"))


class Dataset(DatasetHelper):
    def __init__(self, base_path: str, clinical_path: str) -> None:
        super().__init__(base_path, clinical_path)
        self.clinical_df = self._add_stage()
        self.merged = self.merged_csvs()

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=['number']).columns.tolist()

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df[self._numeric_features(df)]),
                            columns=self._numeric_features(df),
                            index=df.index)

    def merged_csvs(self) -> dict[str, pd.DataFrame]:
        csv_dict = defaultdict(list)
        for f in self._all_csv():
            csv_dict[os.path.basename(f)].append(f)

        result_dict = {}
        for csv_name, files in csv_dict.items():
            dfs = []
            for path in files:
                df = pd.read_csv(path)
                df.insert(0, "patient", os.path.basename(os.path.dirname(path)))
                dfs.append(df)

            key = csv_name.replace("_radiomics_spine_lesions_features.csv", "")
            result_dict[key] = pd.concat(dfs, ignore_index=True)
        return result_dict


class StatisticHelper(DatasetHelper, ABC):
    @property
    @abstractmethod
    def statistic_dfs(self) -> dict[str, pd.DataFrame]:
        pass

    @staticmethod
    def _significant_features(df: pd.DataFrame) -> list[str]:
        return df.loc[df['p_value'] < 0.05, 'Feature'].tolist()

    def _filtered_by_group(self) -> pd.DataFrame:
        filtered_dfs = [
            df[df['Feature'].isin(self._significant_features(df))][['Feature', 'p_value']]
            .rename(columns={'p_value': csv_name})
            for csv_name, df in self.statistic_dfs.items()
        ]

        merged = reduce(lambda l, r: pd.merge(l, r, on='Feature', how='outer'), filtered_dfs)
        merged["Group"] = merged["Feature"].apply(self._extract_feature_group)
        return merged.sort_values(["Group", "Feature"]).reset_index(drop=True)

    def best_features_by_stat(self, plot: bool = False) -> dict[str, list]:
        result_dict = {}
        for csv_name, df in self.statistic_dfs.items():
            df_copy = df[['Feature', 'stat', 'p_value']].copy()
            selector = ThresholdSelectionFeatures(df_copy)

            knee, threshold, selected_features, importance = selector.select_features_by_knee()
            result_dict[csv_name] = selected_features
            if plot: VisualizationManager.plot_stat_selection(df, csv_name)
        return result_dict

    def best_feature_from_group(self) -> dict[str, list]:
        merged_df = pd.concat([
            df[df['Feature'].isin(self._significant_features(df))][['Feature', 'p_value']]
            .rename(columns={'p_value': name})
            for name, df in self.statistic_dfs.items()
        ]).groupby('Feature', as_index=False).first()

        merged_df['Group'] = merged_df['Feature'].apply(self._extract_feature_group)
        image_columns = [c for c in merged_df.columns if c not in ["Feature", "Group"]]

        return {image: [
                group_df.loc[group_df[image].idxmin(), 'Feature']
                for _, group_df in merged_df.groupby('Group')
                if group_df[image].notna().any()] for image in image_columns}


class Spearman(Dataset, StatisticHelper):
    def __init__(self, base_path: str, clinical_path: str,
                 biomarker_name: str='Beta2 microglobulin (mg/l)') -> None:
        super().__init__(base_path, clinical_path)
        self.biomarker_name = biomarker_name

    @property
    def statistic_dfs(self):
        return self._spearman_dfs()

    def _spearman_dfs(self) -> dict[str, pd.DataFrame]:
        result = {}
        for csv_name, df in self.merged.items():
            stats = []
            for col in self._numeric_features(df):
                valid = df[col].notna() & self.clinical_df[self.biomarker_name].notna()
                x = self._standardize(df)[col][valid]
                y = self._standardize(self.clinical_df)[self.biomarker_name][valid]

                corr, p = spearmanr(x, y)
                stats.append({"Feature": col, "stat": corr, "p_value": p})

            result[csv_name] = pd.DataFrame(stats)
        return result


class KruskalWallis(Dataset, StatisticHelper):
    @property
    def statistic_dfs(self):
        return self._kw_dfs()

    def _kw_dfs(self) -> dict[str, pd.DataFrame]:
        results = {}
        for csv_name, df in self.merged.items():
            rows = []
            for feature in self._numeric_features(df):
                data = pd.concat([df[feature], self.clinical_df["Stage"]], axis=1).dropna()
                groups = [data[data["Stage"] == s][feature] for s in [1, 2, 3]]
                stat, p = kruskal(*groups)
                rows.append({"Feature": feature, "stat": stat, "p_value": p})
            results[csv_name] = pd.DataFrame(rows)
        return results


class RandomForest:
    pass


class Lasso:
    pass


class MutualInformation:
    pass


class MRMR:
    pass







# if __name__ == "__main__":
#     base_dir_path = r"D:\DATA_Myelomy"
#     clinical_biomarkers_path = r"D:\Clinical_data\Table_clinical_data.csv"
