import re
import os
import glob
import numpy as np
import pandas as pd
# import seaborn as sns
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
    # TODO: Visualization of RF
    # TODO: Visualization of LASSO
    # TODO: Visualization of MI
    # TODO: Visualization of MRMR

    @staticmethod
    def plot_stat_selection(knee: KneeLocator, importance: np.ndarray, threshold: float, name: str) -> None:
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
    def plot_filtered_features(merged_df: pd.DataFrame, significant_features: list[str],
                               remaining_features: list[str], name: str) -> None:
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


class Dataset:
    def __init__(self, base_path: str, clinical_path: str) -> None:
        self.base_path = base_path
        self.clinical_path = clinical_path
        self.clinical_df = self._add_stage()

    def _read_clinical(self) -> pd.DataFrame:
        return pd.read_csv(self.clinical_path, encoding='cp1252')

    def _add_stage(self) -> pd.DataFrame:
        pd.set_option('future.no_silent_downcasting', True)
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

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=['number']).columns.tolist()

    @staticmethod
    def _significant_features(df: pd.DataFrame) -> list[str]:
        return df.loc[df['p_value'] < 0.05, 'Feature'].tolist()

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df[self._numeric_features(df)]),
                            columns=self._numeric_features(df),
                            index=df.index)

    def _merged_csvs(self) -> dict[str, pd.DataFrame]:
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


class StatisticHelper(Dataset, ABC):
    @property
    @abstractmethod
    def statistic(self):
        pass

    def _select_features_by_knee(self, df:pd.DataFrame) -> tuple[KneeLocator, float | None, list[str], np.ndarray]:
        importance = np.sort(np.abs(df.loc[df['Feature'].isin(self._significant_features(df)), 'stat'].values))[::-1]
        knee = KneeLocator(np.arange(1, len(importance) + 1), importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else None
        best_features = df[df['stat'].abs() > threshold]['Feature'].tolist()
        return knee, threshold, best_features, importance

    def best_features_by_stat(self, image_name: str | None = None, plot: bool = False) -> dict[str, list]:
        statistic_csv = self.statistic if image_name is None else {image_name: self.statistic[image_name]}
        result_dict = {}
        for csv_name, df in statistic_csv.items():
            knee, threshold, selected_features, importance = self._select_features_by_knee(df)
            result_dict[csv_name] = selected_features
            if plot: VisualizationManager.plot_stat_selection(knee, importance, threshold, csv_name)
        return result_dict

    def best_feature_from_group(self, image_name: str | None = None) -> dict[str, list]:
        statistic_csv = self.statistic if image_name is None else {image_name: self.statistic[image_name]}
        merged_df = pd.concat([
            df[df['Feature'].isin(self._significant_features(df))][['Feature', 'p_value']]
            .rename(columns={'p_value': name})
            for name, df in statistic_csv.items()]).groupby('Feature', as_index=False).first()

        merged_df['Group'] = merged_df['Feature'].apply(self._extract_feature_group)
        image_columns = [c for c in merged_df.columns if c not in ["Feature", "Group"]]

        return {image: [group_df.loc[group_df[image].idxmin(), 'Feature']
                        for _, group_df in merged_df.groupby('Group')
                        if group_df[image].notna().any()] for image in image_columns}

    def relevant_features_by_corr(self, image_name: str | None = None, plot: bool = False,
                                  threshold: float = 0.75) -> dict[str, list]:
        merged_csv = self._merged_csvs() if image_name is None else {image_name: self._merged_csvs()[image_name]}
        statistic_csv = self.statistic if image_name is None else {image_name: self.statistic[image_name]}

        result_dict = {}
        for csv_name, (merged_df, stat_df) in zip(merged_csv.keys(), zip(merged_csv.values(), statistic_csv.values())):
            sig_features = self._significant_features(stat_df)
            remaining_features = list(sig_features)

            while True:
                corr_matrix = self._standardize(merged_df)[remaining_features].corr(method='spearman').abs()
                np.fill_diagonal(corr_matrix.values, 0)
                max_corr = corr_matrix.values.max()
                if max_corr <= threshold:
                    break

                idx = np.unravel_index(np.argmax(corr_matrix.values), corr_matrix.shape)
                row_idx, col_idx = map(int, idx)
                feat1, feat2 = corr_matrix.columns[row_idx], corr_matrix.columns[col_idx]
                stat1 = stat_df.loc[stat_df['Feature'] == feat1, 'stat'].values[0]
                stat2 = stat_df.loc[stat_df['Feature'] == feat2, 'stat'].values[0]
                to_remove = feat2 if abs(stat1) >= abs(stat2) else feat1
                remaining_features.remove(to_remove)

            result_dict[csv_name] = remaining_features
            if plot: VisualizationManager.plot_filtered_features(merged_df, sig_features, remaining_features, csv_name)
        return result_dict


class Spearman(StatisticHelper):
    def __init__(self, base_path: str, clinical_path: str, biomarker_name: str='Beta2 microglobulin (mg/l)') -> None:
        super().__init__(base_path, clinical_path)
        self.biomarker_name = biomarker_name

    @property
    def statistic(self) -> dict[str, pd.DataFrame]:
        return self._spearman_dfs()

    def _spearman_dfs(self) -> dict[str, pd.DataFrame]:
        result = {}
        for csv_name, df in self._merged_csvs().items():
            stats = []
            for col in self._numeric_features(df):
                valid = df[col].notna() & self.clinical_df[self.biomarker_name].notna()
                x = self._standardize(df)[col][valid]
                y = self._standardize(self.clinical_df)[self.biomarker_name][valid]

                corr, p = spearmanr(x, y)
                stats.append({"Feature": col, "stat": corr, "p_value": p})

            result[csv_name] = pd.DataFrame(stats)
        return result


class KruskalWallis(StatisticHelper):
    def __init__(self, base_path: str, clinical_path: str) -> None:
        super().__init__(base_path, clinical_path)

    @property
    def statistic(self) -> dict[str, pd.DataFrame]:
        return self._kw_dfs()

    def _kw_dfs(self) -> dict[str, pd.DataFrame]:
        results = {}
        for csv_name, df in self._merged_csvs().items():
            rows = []
            for feature in self._numeric_features(df):
                data = pd.concat([df[feature], self.clinical_df["Stage"]], axis=1).dropna()
                groups = [data[data["Stage"] == s][feature] for s in [1, 2, 3]]
                stat, p = kruskal(*groups)
                rows.append({"Feature": feature, "stat": stat, "p_value": p})
            results[csv_name] = pd.DataFrame(rows)
        return results










if __name__ == "__main__":
    base_dir_path = r"D:\DATA_Myelomy"
    clinical_biomarkers_path = r"D:\Clinical_data\Table_clinical_data.csv"
    print(KruskalWallis(base_dir_path, clinical_biomarkers_path).relevant_features_by_corr(plot=True))
