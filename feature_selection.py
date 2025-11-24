import re
import os
import glob
import numpy as np
import pandas as pd
from kneed import KneeLocator
from abc import ABC, abstractmethod
# from sklearn.linear_model import Lasso
# from feature_engine.selection import MRMR
from scipy.stats import spearmanr, kruskal
from collections import defaultdict
# from collections import Counter
# from mrmr import mrmr_classif, mrmr_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from visualization_manager import VisualizationManager
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class DatasetHelper:
    @staticmethod
    def _read_clinical() -> pd.DataFrame:
        return pd.read_csv(Settings.clinical_path, encoding='cp1252')

    @staticmethod
    def _all_csv() -> list[str]:
        return glob.glob(os.path.join(Settings.base_dir_path, "*", "*spine_lesions*.csv"))

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


class Dataset(DatasetHelper):
    def __init__(self) -> None:
        self.clinical_df = self._add_stage()

    def _add_stage(self) -> pd.DataFrame:
        pd.set_option('future.no_silent_downcasting', True)
        df = self._read_clinical()
        df['Stage'] = df['ISS classification'].replace({'Stage 1': 1, 'Stage 2': 2, 'Stage 3': 3})
        df = df.dropna(subset=['Stage'])
        df['Stage'] = df['Stage'].astype(int)
        return df

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

    def best_features_by_stat(self) -> dict[str, list]:
        statistic_csv = self.statistic if Settings.image_name is None else {Settings.image_name: self.statistic[Settings.image_name]}
        result_dict = {}
        for csv_name, df in statistic_csv.items():
            knee, threshold, selected_features, importance = self._select_features_by_knee(df)
            result_dict[csv_name] = selected_features
            if Settings.show_visualization: VisualizationManager.plot_stat_selection(knee, importance, threshold, csv_name)
        return result_dict

    def best_feature_from_group(self) -> dict[str, list]:
        statistic_csv = self.statistic if Settings.image_name is None else {Settings.image_name: self.statistic[Settings.image_name]}
        merged_df = pd.concat([
            df[df['Feature'].isin(self._significant_features(df))][['Feature', 'p_value']]
            .rename(columns={'p_value': name})
            for name, df in statistic_csv.items()]).groupby('Feature', as_index=False).first()

        merged_df['Group'] = merged_df['Feature'].apply(self._extract_feature_group)
        image_columns = [c for c in merged_df.columns if c not in ["Feature", "Group"]]

        return {image: [group_df.loc[group_df[image].idxmin(), 'Feature']
                        for _, group_df in merged_df.groupby('Group')
                        if group_df[image].notna().any()] for image in image_columns}

    def relevant_features_by_corr(self, threshold: float = 0.75) -> dict[str, list]:
        merged_csv = self._merged_csvs() if Settings.image_name is None else {Settings.image_name: self._merged_csvs()[Settings.image_name]}
        statistic_csv = self.statistic if Settings.image_name is None else {Settings.image_name: self.statistic[Settings.image_name]}

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
            if Settings.show_visualization: VisualizationManager.plot_filtered_features(merged_df, sig_features, remaining_features, csv_name)
        return result_dict


class Spearman(StatisticHelper):
    @property
    def statistic(self) -> dict[str, pd.DataFrame]:
        return self._spearman_dfs()

    def _spearman_dfs(self) -> dict[str, pd.DataFrame]:
        result = {}
        for csv_name, df in self._merged_csvs().items():
            stats = []
            for col in self._numeric_features(df):
                valid = df[col].notna() & self.clinical_df[Settings.clinical_biomarker_name].notna()
                x = self._standardize(df)[col][valid]
                y = self._standardize(self.clinical_df)[Settings.clinical_biomarker_name][valid]

                corr, p = spearmanr(x, y)
                stats.append({"Feature": col, "stat": corr, "p_value": p})

            result[csv_name] = pd.DataFrame(stats)
        return result


class KruskalWallis(StatisticHelper):
    @property
    def statistic(self) -> dict[str, pd.DataFrame]:
        return self._kw_dfs()

    def _kw_dfs(self) -> dict[str, pd.DataFrame]:
        results = {}
        for csv_name, df in self._merged_csvs().items():
            rows = []
            for feature in self._numeric_features(df):
                data = pd.concat([df[feature], self.clinical_df[Settings.clinical_group_name]], axis=1).dropna()
                groups = [data[data[Settings.clinical_group_name] == s][feature] for s in [1, 2, 3]]
                stat, p = kruskal(*groups)
                rows.append({"Feature": feature, "stat": stat, "p_value": p})
            results[csv_name] = pd.DataFrame(rows)
        return results


class RandomForestHelper:
    def __init__(self, features: pd.DataFrame, labels: pd.Series, task: str = "classification",
                 n_trees: int = 200, test_size: float = 0.2):
        self.features = features
        self.labels = labels
        self.task = task
        self.n_trees = n_trees
        self.test_size = test_size

        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.model = None
        self.feature_importance_df = None

    def split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=self.test_size,
            stratify=self.labels if self.task == "classification" else None, random_state=42)

    def fit(self):
        self.split()
        if self.task == "classification":
            self.model = RandomForestClassifier(n_estimators=self.n_trees, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=self.n_trees, random_state=42)
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        return self.model.predict(self.x_test)

    def feature_importance(self):
        self.feature_importance_df = pd.DataFrame({
            "Feature": self.features.columns,
            "Importance": self.model.feature_importances_
        }).sort_values("Importance", ascending=False)
        return self.feature_importance_df

    @staticmethod
    def select_features_by_knee(feature_importance_df: pd.DataFrame) -> tuple[KneeLocator, float | None, list[str], np.ndarray]:
        importance = np.sort(feature_importance_df['Importance'].values)[::-1]
        knee = KneeLocator(np.arange(1, len(importance) + 1), importance, curve='convex', direction='decreasing')
        threshold = importance[knee.knee] if knee.knee is not None else 0
        best_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()
        return knee, threshold, best_features, importance


class RandomForest(Dataset):
    def classification(self, significant: bool = False) -> dict[str, list]:

        merged = self._merged_csvs() if Settings.image_name is None else {Settings.image_name: self._merged_csvs()[Settings.image_name]}
        statistic = KruskalWallis().statistic
        statistic = statistic if Settings.image_name is None else {Settings.image_name: statistic[Settings.image_name]}
        labels = self.clinical_df[Settings.clinical_group_name]

        result_dict = {}
        for name, df in merged.items():
            if significant and name in statistic:
                features = df[self._significant_features(statistic[name])]
            else:
                features = df[self._numeric_features(df)]

            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels = features[valid_idx], labels[valid_idx]

            rf_helper = RandomForestHelper(pd.DataFrame(features), labels)
            rf_helper.fit()
            selected_features = rf_helper.select_features_by_knee(rf_helper.feature_importance())[2]
            result_dict[name] = selected_features

            if Settings.show_visualization: VisualizationManager.plot_rf_classification(
                rf_helper.y_test, rf_helper.predict(), rf_helper.feature_importance(), name)
        return result_dict

    def regression(self, significant: bool = False) -> dict[str, list]:

        merged = self._merged_csvs() if Settings.image_name is None else {Settings.image_name: self._merged_csvs()[Settings.image_name]}
        statistic = Spearman().statistic
        statistic = statistic if Settings.image_name is None else {Settings.image_name: statistic[Settings.image_name]}
        labels = self.clinical_df[Settings.clinical_biomarker_name]

        result_dict = {}
        for name, df in merged.items():
            if significant and name in statistic:
                features = df[self._significant_features(statistic[name])]
            else:
                features = df[self._numeric_features(df)]

            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels = features[valid_idx], labels[valid_idx]

            rf_helper = RandomForestHelper(pd.DataFrame(features), labels, task="regression")
            rf_helper.fit()
            selected_features = rf_helper.select_features_by_knee(rf_helper.feature_importance())[2]
            result_dict[name] = selected_features

            if Settings.show_visualization: VisualizationManager.plot_rf_regression(
                rf_helper.y_test, rf_helper.predict(), rf_helper.feature_importance(), name)
        return result_dict


class Settings:
    base_dir_path: str = r"D:\DATA_Myelomy"
    clinical_path: str = r"D:\Clinical_data\Table_clinical_data.csv"

    clinical_biomarker_name: str | None = 'Beta2 microglobulin (mg/l)'
    clinical_group_name: str | None = 'Stage'

    image_name: str | None = None
    show_visualization: bool = False

if __name__ == "__main__":
    pass
