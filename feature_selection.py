import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from visualization_manager import VisualizationManager


class Settings:
    base_dir_path: str = r"G:\DATA_Myelomy"
    clinical_path: str = r"G:\Clinical_data\Table_clinical_data.csv"

    clinical_biomarker_name: str | None = 'Beta2 microglobulin (mg/l)'    # 'Creatinine level (µmol/l)'    # 'Serum M-protein quantity (g/l)'
    clinical_group_name: str | None = 'ISS classification'

    image_name: str | None = None
    show_visualization: bool = False


class DatasetHelper:
    @staticmethod
    def _read_clinical() -> pd.DataFrame:
        return pd.read_csv(Settings.clinical_path, encoding='cp1252')

    @staticmethod
    def _all_csv() -> list[str]:
        import glob
        return glob.glob(os.path.join(Settings.base_dir_path, "*", "*spine_lesions*.csv"))

    @staticmethod
    def _extract_feature_group(feature_name: str) -> str:
        import re
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

    def _categorical_df(self) -> pd.DataFrame:
        pd.set_option('future.no_silent_downcasting', True)

        df = self._read_clinical()
        df = df.dropna(subset=[Settings.clinical_group_name])

        unique_values = sorted(df[Settings.clinical_group_name].unique())
        mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}

        df['Category'] = df[Settings.clinical_group_name].map(mapping).astype(int)
        return df


class Dataset(DatasetHelper):
    def __init__(self) -> None:
        self.clinical_df = self._categorical_df()

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=['number']).columns.tolist()

    @staticmethod
    def _significant_features(df: pd.DataFrame) -> list[str]:
        return list(df.loc[df['p_value'] < 0.05, 'Feature'].values)

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        num_features = self._numeric_features(df)
        return pd.DataFrame(scaler.fit_transform(df[num_features]), columns=num_features, index=df.index)

    def _merged_csvs(self) -> dict[str, pd.DataFrame]:
        from collections import defaultdict
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


class ThresholdSelectionFeatures:
    def __init__(self, importance: pd.DataFrame) -> None:
        self.importance = importance

    def select_features_by_knee(self) -> tuple:
        from kneed import KneeLocator
        importance = np.sort(np.abs(self.importance['importance'].values))[::-1]

        knee = KneeLocator(
            np.arange(1, len(importance) + 1),
            importance,
            curve='convex',
            direction='decreasing')

        threshold = importance[knee.knee] if knee.knee is not None else 0
        best_features = list(self.importance.loc[self.importance['importance'].abs() > threshold, 'Feature'])

        return knee, threshold, best_features, importance


class StatisticHelper(Dataset, ABC):
    @property
    @abstractmethod
    def statistic(self) -> dict[str, pd.DataFrame]:
        pass

    def best_features_by_stat(self) -> dict[str, list]:
        if Settings.image_name is None:
            statistic_csv = self.statistic
        else:
            statistic_csv = {Settings.image_name: self.statistic[Settings.image_name]}

        result_dict = {}
        for csv_name, df in statistic_csv.items():
            knee, threshold, selected_features, importance = ThresholdSelectionFeatures(df).select_features_by_knee()
            result_dict[csv_name] = selected_features

            if Settings.show_visualization:
                VisualizationManager.plot_stat_selection(knee, importance, threshold, csv_name)

        return result_dict

    def best_feature_from_group(self) -> dict[str, list]:
        if Settings.image_name is None:
            statistic_csv = self.statistic
        else:
            statistic_csv = {Settings.image_name: self.statistic[Settings.image_name]}

        dfs = []
        for image_name, df in statistic_csv.items():
            sig_features = self._significant_features(df)

            tmp = df.loc[df['Feature'].isin(sig_features), ['Feature', 'p_value']].copy()
            tmp.rename(columns={'p_value': image_name}, inplace=True)
            dfs.append(tmp)

        merged_df = (pd.concat(dfs).groupby('Feature', as_index=False).first())
        merged_df['Group'] = merged_df['Feature'].apply(self._extract_feature_group)
        image_columns = [c for c in merged_df.columns if c not in ["Feature", "Group"]]

        result = {}
        for image in image_columns:
            selected = []
            for _, group_df in merged_df.groupby('Group'):
                valid = group_df.dropna(subset=[image])
                if valid.empty:
                    continue

                best_feature = valid.loc[valid[image].idxmin(), 'Feature']
                selected.append(best_feature)
            result[image] = selected
        return result

    def relevant_features_by_corr(self, threshold: float=0.75) -> dict[str, list]:
        if Settings.image_name is None:
            merged_csv = self._merged_csvs()
            statistic_csv = self.statistic
        else:
            merged_csv = {Settings.image_name: self._merged_csvs()[Settings.image_name]}
            statistic_csv = {Settings.image_name: self.statistic[Settings.image_name]}

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
                feat1, feat2 = str(corr_matrix.columns[row_idx]), str(corr_matrix.columns[col_idx])

                stat1 = stat_df.loc[stat_df['Feature'] == feat1, 'importance'].values[0]
                stat2 = stat_df.loc[stat_df['Feature'] == feat2, 'importance'].values[0]
                to_remove = feat2 if abs(stat1) >= abs(stat2) else feat1
                remaining_features.remove(to_remove)

            result_dict[csv_name] = remaining_features
            if Settings.show_visualization:
                VisualizationManager.plot_filtered_features(merged_df, sig_features, remaining_features, csv_name)

        return result_dict


class Spearman(StatisticHelper):
    @property
    def statistic(self) -> dict[str, pd.DataFrame]:
        return self._spearman_dfs()

    def _spearman_dfs(self) -> dict[str, pd.DataFrame]:
        from scipy.stats import spearmanr
        result = {}

        for csv_name, df in self._merged_csvs().items():
            stats = []
            for col in self._numeric_features(df):
                valid = df[col].notna() & self.clinical_df[Settings.clinical_biomarker_name].notna()
                x = self._standardize(df)[col][valid]
                y = self._standardize(self.clinical_df)[Settings.clinical_biomarker_name][valid]

                corr, p = spearmanr(x, y)
                stats.append({"Feature": col, "importance": corr, "p_value": p})

            result[csv_name] = pd.DataFrame(stats)
        return result


class KruskalWallis(StatisticHelper):
    @property
    def statistic(self) -> dict[str, pd.DataFrame]:
        return self._kw_dfs()

    def _kw_dfs(self) -> dict[str, pd.DataFrame]:
        from scipy.stats import kruskal
        results = {}

        for csv_name, df in self._merged_csvs().items():
            rows = []
            for feature in self._numeric_features(df):
                data = pd.concat([df[feature], self.clinical_df["Category"]], axis=1).dropna()
                categories = np.sort(data["Category"].dropna().unique())
                groups = [data[data["Category"] == c][feature] for c in categories]

                stat, p = kruskal(*groups)
                rows.append({"Feature": feature, "importance": stat, "p_value": p})

            results[csv_name] = pd.DataFrame(rows)
        return results


class LassoHelper:
    def __init__(self, features: pd.DataFrame, labels: pd.Series,
                 alpha: float = 0.01, test_size: float = 0.2) -> None:
        self.features = features
        self.labels = labels
        self.alpha = alpha
        self.test_size = test_size

        self.x_train = self.x_test = None
        self.y_train = self.y_test = None
        self.x_train_scaled = self.x_test_scaled = None

        self.model = None
        self.scaler = None

    def split(self) -> None:
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=self.test_size, random_state=42)

    def fit(self) -> None:
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler

        self.split()
        self.scaler = StandardScaler()

        self.x_train_scaled = pd.DataFrame(self.scaler.fit_transform(self.x_train),
                                           columns=self.x_train.columns, index=self.x_train.index)

        self.x_test_scaled = pd.DataFrame(self.scaler.transform(self.x_test),
                                          columns=self.x_test.columns, index=self.x_test.index)

        self.model = Lasso(alpha=self.alpha, random_state=42)
        self.model.fit(self.x_train_scaled, self.y_train)

    def predict(self):
        return self.model.predict(self.x_test_scaled)

    def feature_importance(self) -> pd.DataFrame:
        return (pd.DataFrame({
                "Feature": self.x_train.columns,
                "importance": self.model.coef_})
            .assign(importance=lambda df: df["importance"].abs())
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))


class Lasso(Dataset):
    def regression(self, significant: bool = False) -> dict[str, list]:
        merged = self._merged_csvs()
        statistic = Spearman().statistic

        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        labels = self.clinical_df[Settings.clinical_biomarker_name]

        result_dict = {}
        for name, df in merged.items():
            if significant and name in statistic:
                features = df[self._significant_features(statistic[name])]
            else:
                features = df[self._numeric_features(df)]

            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels_ = features[valid_idx], labels[valid_idx]

            lasso_helper = LassoHelper(features, labels_, alpha=0.01)
            lasso_helper.fit()

            knee, threshold, selected_features, importance = (ThresholdSelectionFeatures(lasso_helper.feature_importance()).select_features_by_knee())
            result_dict[name] = selected_features

            if Settings.show_visualization:
                VisualizationManager.plot_lasso_regression(lasso_helper.y_test, lasso_helper.predict(), knee, importance, threshold, name)

        return result_dict


class RandomForestHelper:
    def __init__(self, features: pd.DataFrame, labels: pd.Series, task: str="classification",
                 n_trees: int=200, test_size: float=0.2) -> None:
        self.features = features
        self.labels = labels
        self.task = task
        self.n_trees = n_trees
        self.test_size = test_size

        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.model = None
        self.feature_importance_df = None

    def split(self) -> None:
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=self.test_size,
            stratify=self.labels if self.task == "classification" else None, random_state=42)

    def fit(self) -> None:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        self.split()
        if self.task == "classification":
            self.model = RandomForestClassifier(n_estimators=self.n_trees, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=self.n_trees, random_state=42)
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        return self.model.predict(self.x_test)

    def feature_importance(self) -> pd.DataFrame:
        self.feature_importance_df = pd.DataFrame({
            "Feature": self.features.columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        return self.feature_importance_df


class RandomForest(Dataset):
    def classification(self, significant: bool=False) -> dict[str, list]:
        merged = self._merged_csvs()
        statistic = KruskalWallis().statistic

        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}
        labels = self.clinical_df["Category"]

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

            knee, threshold, selected_features, importance = ThresholdSelectionFeatures(rf_helper.feature_importance()).select_features_by_knee()
            result_dict[name] = selected_features

            if Settings.show_visualization:
                VisualizationManager.plot_rf_classification(rf_helper.y_test, rf_helper.predict(), knee, importance, threshold, name)

        return result_dict

    def regression(self, significant: bool=False) -> dict[str, list]:
        merged = self._merged_csvs()
        statistic = Spearman().statistic

        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}
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

            knee, threshold, selected_features, importance = ThresholdSelectionFeatures(rf_helper.feature_importance()).select_features_by_knee()
            result_dict[name] = selected_features

            if Settings.show_visualization:
                VisualizationManager.plot_rf_regression(rf_helper.y_test, rf_helper.predict(), knee, importance, threshold, name)

        return result_dict


class MutualInformation(Dataset):
    def classification(self) -> dict[str, list]:
        from sklearn.feature_selection import mutual_info_classif
        if Settings.image_name is None:
            merged = self._merged_csvs()
        else:
            merged = {Settings.image_name: self._merged_csvs()[Settings.image_name]}
        labels = self.clinical_df["Category"]

        result_dict = {}
        for name, df in merged.items():
            mutual_info_list = []
            for feat in self._numeric_features(df):
                valid_idx = df[feat].notna() & labels.notna()
                x, y = df[[feat]][valid_idx], labels[valid_idx]

                mi = mutual_info_classif(x, y, random_state=42)[0]
                mutual_info_list.append({"Feature": feat, "importance": mi})

            mi_df = pd.DataFrame(mutual_info_list).sort_values("importance", ascending=False)
            knee, threshold, selected, importance = ThresholdSelectionFeatures(mi_df).select_features_by_knee()
            result_dict[name] = selected

            if Settings.show_visualization:
                VisualizationManager.plot_mutual_information(knee, importance, threshold, name)

        return result_dict

    def regression(self) -> dict[str, list]:
        from sklearn.feature_selection import mutual_info_regression
        if Settings.image_name is None:
            merged = self._merged_csvs()
        else:
            merged = {Settings.image_name: self._merged_csvs()[Settings.image_name]}
        labels = self.clinical_df[Settings.clinical_biomarker_name]

        result_dict = {}
        for name, df in merged.items():
            mutual_info_list = []
            for feat in self._numeric_features(df):
                valid_idx = df[feat].notna() & labels.notna()
                x, y = df[[feat]][valid_idx], labels[valid_idx]

                mi = mutual_info_regression(x, y, random_state=42)[0]
                mutual_info_list.append({"Feature": feat, "importance": mi})

            mi_df = pd.DataFrame(mutual_info_list).sort_values("importance", ascending=False)
            knee, threshold, selected, importance = ThresholdSelectionFeatures(mi_df).select_features_by_knee()
            result_dict[name] = selected

            if Settings.show_visualization:
                VisualizationManager.plot_mutual_information(knee, importance, threshold, name)

        return result_dict


class MRMR(Dataset):
    def classification_type1(self, significant: bool=False, n_top_features: int=20) -> dict[str, list]:
        from mrmr import mrmr_classif
        merged = self._merged_csvs()
        statistic = KruskalWallis().statistic

        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        result_dict = {}
        for name, df in merged.items():
            temp_df = pd.merge(df, self.clinical_df[['Patient ID', 'Category']],
                               left_on='patient', right_on='Patient ID', how='inner')

            if significant and name in statistic:
                features = temp_df[self._significant_features(statistic[name])]
            else:
                features = temp_df[self._numeric_features(df)].drop(columns=['Category'], errors='ignore')
            labels = temp_df['Category']

            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels = features[valid_idx], labels[valid_idx]

            df_mrmr = self._standardize(features).copy()
            df_mrmr['Category'] = labels.values
            selected_features = mrmr_classif(X=df_mrmr.drop(columns='Category'), y=df_mrmr['Category'], K=n_top_features)

            if Settings.show_visualization:
                VisualizationManager.plot_mrmr(self._standardize(features), selected_features, name)

            result_dict[name] = selected_features
        return result_dict

    def classification_type2(self, significant: bool=False, n_top_features: int=20) -> dict[str, list]:
        from feature_engine.selection import MRMR
        merged = self._merged_csvs()
        statistic = KruskalWallis().statistic

        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        result_dict = {}
        for name, df in merged.items():
            temp_df = pd.merge(df, self.clinical_df[['Patient ID', 'Category']],
                               left_on='patient', right_on='Patient ID', how='inner')

            if significant and name in statistic:
                features = temp_df[self._significant_features(statistic[name])]
            else:
                features = temp_df[self._numeric_features(df)].drop(columns=['Category'], errors='ignore')
            labels = temp_df['Category']

            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels = features[valid_idx], labels[valid_idx]

            selector = MRMR(method='MIQ', max_features=n_top_features)
            selector.fit(self._standardize(features), labels)
            selected_features = selector.transform(self._standardize(features)).columns.tolist()

            if Settings.show_visualization:
                VisualizationManager.plot_mrmr(self._standardize(features), selected_features, name)

            result_dict[name] = selected_features
        return result_dict

    def regression(self, significant: bool=False, n_top_features: int=20) -> dict[str, list]:
        from mrmr import mrmr_regression
        merged = self._merged_csvs()
        statistic = Spearman().statistic

        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        result_dict = {}
        for name, df in merged.items():
            temp_df = pd.merge(df, self.clinical_df[['Patient ID', Settings.clinical_biomarker_name]],
                               left_on='patient', right_on='Patient ID', how='inner')

            if significant and name in statistic:
                features = temp_df[self._significant_features(statistic[name])]
            else:
                features = temp_df[self._numeric_features(df)].drop(columns=[Settings.clinical_biomarker_name], errors='ignore')
            labels = temp_df[Settings.clinical_biomarker_name]

            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels = features[valid_idx], labels[valid_idx]

            df_mrmr = self._standardize(features).copy()
            df_mrmr[Settings.clinical_biomarker_name] = labels.values
            selected_features = mrmr_regression(X=df_mrmr.drop(columns=Settings.clinical_biomarker_name), y=df_mrmr[Settings.clinical_biomarker_name], K=n_top_features)

            if Settings.show_visualization:
                VisualizationManager.plot_mrmr(self._standardize(features), selected_features, name)

            result_dict[name] = selected_features
        return result_dict


class FeatureSelection:
    # WITH CORRELATED IMAGE FEATURES (NOR MRMR)
    selected_spearman = Spearman().best_features_by_stat()
    selected_kw = KruskalWallis().best_features_by_stat()

    # best_one_group_spearman = Spearman().best_feature_from_group()
    # best_one_group_kw = KruskalWallis().best_feature_from_group()

    mi_classification = MutualInformation().classification()
    mi_regression = MutualInformation().regression()

    rf_classification_all = RandomForest().classification()
    rf_classification_significant = RandomForest().classification(significant=True)

    rf_regression_all = RandomForest().regression()
    rf_regression_significant = RandomForest().regression(significant=True)

    lasso_regression_all = Lasso().regression()
    lasso_regression_significant = Lasso().regression(significant=True)

    # WITHOUT CORRELATED IMAGE FEATURES (MRMR)
    filtered_spearman = Spearman().relevant_features_by_corr()
    filtered_kw = KruskalWallis().relevant_features_by_corr()

    mrmr_classification_1_all = MRMR().classification_type1()
    mrmr_classification_2_all = MRMR().classification_type2()

    mrmr_classification_1_significant = MRMR().classification_type1(significant=True)
    mrmr_classification_2_significant = MRMR().classification_type2(significant=True)

    mrmr_regression_all = MRMR().regression()
    mrmr_regression_significant = MRMR().regression(significant=True)


    all_dicts = [
                 selected_kw, selected_spearman,
                 rf_classification_all, rf_classification_significant,
                 rf_regression_all, rf_regression_significant,
                 lasso_regression_all, lasso_regression_significant,
                 mi_classification, mi_regression,
                 filtered_spearman, filtered_kw,
                 mrmr_classification_1_all, mrmr_classification_1_significant,
                 mrmr_classification_2_all, mrmr_classification_2_significant,
                 mrmr_regression_all, mrmr_regression_significant,
                ]

    @staticmethod
    def _print_count_tables(tables: dict[str, pd.DataFrame]) -> None:
        for name, df in tables.items():
            print(f"\n-------------------------------------{name}-----------------------------------------")
            print(df[df["Count"] > 1])

    def count_tables(self) -> None:
        from collections import Counter
        dataset_names = self.all_dicts[0].keys()
        result = {ds: Counter() for ds in dataset_names}

        for d in self.all_dicts:
            for dataset_name, feature_list in d.items():
                result[dataset_name].update(feature_list)

        tables = {}
        for dataset_name, counter in result.items():
            df = pd.DataFrame(counter.items(), columns=["Feature", "Count"])
            df = df.sort_values("Count", ascending=False).reset_index(drop=True)

            total = len(self.all_dicts)
            df["Ratio"] = df["Count"].astype(str) + "/" + str(total)
            df["Ratio[%]"] = round((df["Count"] / total) * 100, 1)
            tables[dataset_name] = df

        self._print_count_tables(tables)


if __name__ == "__main__":
    FeatureSelection().count_tables()
