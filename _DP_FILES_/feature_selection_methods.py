import os
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from visualization_manager import VisualizationManager


class Settings:
    """
    Configuration class storing all global paths,
    clinical target names, and visualization settings.
    """
    base_dir_path: str = r"E:\DATA_Myelomy"
    clinical_path: str = r"E:\Clinical_data\Table_clinical_data.csv"

    # Name of the clinical biomarker to be used for regression
    clinical_biomarker_name: str | None = 'Beta2 microglobulin (mg/l)'
    # Name of the clinical group
    clinical_group_name: str | None = 'ISS classification'

    # Optional: specify a single image name for analysis
    image_name: str | None = None
    # Whether to show visualization plots
    show_visualization: bool = False


class DatasetHelper:
    """
    Helper class providing static methods for reading clinical data,
    listing CSV files, and extracting feature groups from feature names.
    """
    @staticmethod
    def _read_clinical() -> pd.DataFrame:
        """
        Reads the clinical CSV file defined in Settings.
        Returns: pd.DataFrame: Clinical data.
        """
        return pd.read_csv(Settings.clinical_path, encoding='cp1252')

    @staticmethod
    def _all_csv() -> list[str]:
        """
        Returns a list of all CSV files matching the spine lesions pattern in the base directory.
        Returns: list[str]: Paths to CSV files.
        """
        import glob
        return glob.glob(os.path.join(Settings.base_dir_path, "*", "*spine_lesions*.csv"))

    @staticmethod
    def _extract_feature_group(feature_name: str) -> str:
        """
        Extracts the radiomic feature group based on regex matching of the feature name.
        Args: feature_name (str): Name of the feature.
        Returns: str: Feature group name (e.g., 'original_glcm').
        """
        import re
        patterns = [
            r"^gradient_firstorder", r"^gradient_glcm", r"^gradient_glrlm",
            r"^gradient_glszm", r"^gradient_gldm", r"^gradient_ngtdm",
            r"^original_firstorder", r"^original_glcm", r"^original_glrlm",
            r"^original_glszm", r"^original_gldm", r"^original_ngtdm",
            r"^shape"
        ]

        for p in patterns:
            if re.match(p, feature_name):
                return p.replace("^", "")
        return "other"

    def _categorical_df(self) -> pd.DataFrame:
        """
        Returns a categorical version of the clinical group variable.
        The group is mapped to integers starting from 1.
        Returns: pd.DataFrame: Clinical data with a new 'Category' column.
        """
        # Ensure future pandas behavior does not silently downcast
        pd.set_option('future.no_silent_downcasting', True)

        df = self._read_clinical()

        # Drop rows where the group variable is missing
        df = df.dropna(subset=[Settings.clinical_group_name])

        # Map unique values of the group to consecutive integers
        unique_values = sorted(df[Settings.clinical_group_name].unique())
        mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}

        # Add categorical column
        df['Category'] = df[Settings.clinical_group_name].map(mapping).astype(int)

        return df


class Dataset(DatasetHelper):
    """
    Provides methods to access clinical data, extract numeric or significant features,
    standardize feature matrices, and merge CSV files from different patients.
    """
    def __init__(self) -> None:
        """
        Initializes the Dataset by reading and categorizing clinical data.
        """
        self.clinical_df = self._categorical_df()

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> list[str]:
        """
        Returns the list of numeric columns in a DataFrame.
        Args: df (pd.DataFrame): Input DataFrame.
        Returns: list[str]: List of numeric column names.
        """
        return df.select_dtypes(include=['number']).columns.tolist()

    @staticmethod
    def _significant_features(df: pd.DataFrame) -> list[str]:
        """
        Returns the list of features with p-value < 0.05.
        Args: df (pd.DataFrame): DataFrame with 'Feature' and 'p_value' columns.
        Returns: list[str]: List of significant feature names.
        """
        return list(df.loc[df['p_value'] < 0.05, 'Feature'].values)

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes numeric features in the DataFrame to mean=0, std=1.
        Args: df (pd.DataFrame): Input DataFrame.
        Returns: pd.DataFrame: Standardized DataFrame with the same columns and index.
        """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        num_features = self._numeric_features(df)

        return pd.DataFrame(scaler.fit_transform(df[num_features]), columns=num_features, index=df.index)

    def _merged_csvs(self) -> dict[str, pd.DataFrame]:
        """
        Reads all CSVs matching the spine lesions pattern, merges files by CSV name,
        and adds a 'patient' column from the folder name.
        Returns: dict[str, pd.DataFrame]: Keys are CSV names (without suffix), values are merged DataFrames.
        """
        from collections import defaultdict

        csv_dict = defaultdict(list)

        for f in self._all_csv():
            csv_dict[os.path.basename(f)].append(f)

        result_dict = {}
        for csv_name, files in csv_dict.items():
            dfs = []
            for path in files:
                df = pd.read_csv(path)
                # Add patient identifier column from folder name
                df.insert(0, "patient", os.path.basename(os.path.dirname(path)))
                dfs.append(df)

            key = csv_name.replace("_radiomics_spine_lesions_features.csv", "")
            result_dict[key] = pd.concat(dfs, ignore_index=True)
        return result_dict


class ThresholdSelectionFeatures:
    """
    Selects the most important features using the
    "knee/elbow" method on feature importance values.
    """
    def __init__(self, importance: pd.DataFrame) -> None:
        """
        Initializes the selector with a DataFrame containing feature names and their importance.
        Args: importance (pd.DataFrame): DataFrame with columns 'Feature' and 'importance'.
        """
        self.importance = importance

    def select_features_by_knee(self) -> tuple:
        """
        Identifies the 'knee' point in the sorted importance curve
        and selects features above the threshold.
        Returns: tuple: (knee object, threshold, list of selected features, sorted importance values)
        """
        from kneed import KneeLocator

        # Sort absolute importance values descending
        importance = np.sort(np.abs(self.importance['importance'].values))[::-1]

        # Detect knee/elbow in the sorted importance curve
        knee = KneeLocator(np.arange(1, len(importance) + 1), importance,
                           curve='convex', direction='decreasing')

        # Determine threshold at the knee point
        threshold = importance[knee.knee] if knee.knee is not None else 0

        best_features = list(
            self.importance.loc[self.importance['importance'].abs() > threshold, 'Feature'])

        return knee, threshold, best_features, importance


class StatisticHelper(Dataset, ABC):
    """
    Abstract base class for computing statistical feature selection on datasets.
    Provides methods for selecting best features based on statistics, groups, and correlation filtering.
    """
    @property
    @abstractmethod
    def statistic(self) -> dict[str, pd.DataFrame]:
        """
        Abstract property that must return a dictionary of DataFrames containing feature statistics.
        Keys are CSV/image names, values are DataFrames with at least 'Feature' and 'p_value' columns.
        """
        pass

    def best_features_by_stat(self) -> dict[str, list]:
        """
        Selects the best features per CSV/image based on the "knee" method applied to the statistic.
        Returns: dict[str, list]: Keys are CSV/image names, values are lists of selected features.
        """
        if Settings.image_name is None:
            statistic_csv = self.statistic
        else:
            statistic_csv = {Settings.image_name: self.statistic[Settings.image_name]}

        result_dict = {}
        for csv_name, df in statistic_csv.items():
            # Use ThresholdSelectionFeatures to detect knee and select features
            knee, threshold, selected_features, importance = ThresholdSelectionFeatures(df).select_features_by_knee()
            result_dict[csv_name] = selected_features

            if Settings.show_visualization:
                VisualizationManager.plot_stat_selection(knee, importance, threshold, csv_name)

        return result_dict

    def best_feature_from_group(self) -> dict[str, list]:
        """
        Selects the best feature per feature group for each CSV/image.
        Within each group, the feature with the lowest p-value is selected.
        Returns: dict[str, list]: Keys are CSV/image names, values are lists of selected features.
        """
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

        # Merge all images into one DataFrame, keep first occurrence of each feature
        merged_df = pd.concat(dfs).groupby('Feature', as_index=False).first()
        # Assign feature groups
        merged_df['Group'] = merged_df['Feature'].apply(self._extract_feature_group)
        image_columns = [c for c in merged_df.columns if c not in ["Feature", "Group"]]

        result = {}
        for image in image_columns:
            selected = []
            for _, group_df in merged_df.groupby('Group'):
                valid = group_df.dropna(subset=[image])
                if valid.empty:
                    continue
                # Select feature with lowest p-value
                best_feature = valid.loc[valid[image].idxmin(), 'Feature']
                selected.append(best_feature)
            result[image] = selected
        return result

    def relevant_features_by_corr(self, threshold: float = 0.75) -> dict[str, list]:
        """
        Removes highly correlated features (Spearman correlation) iteratively, keeping the one with higher importance.
        Args: threshold (float): Maximum allowed correlation; features exceeding this will be filtered.
        Returns: dict[str, list]: Keys are CSV/image names, values are lists of remaining relevant features.
        """
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
                # Compute Spearman correlation matrix
                corr_matrix = self._standardize(merged_df)[remaining_features].corr(method='spearman').abs()
                np.fill_diagonal(corr_matrix.values, 0)
                max_corr = corr_matrix.values.max()
                if max_corr <= threshold:
                    break

                # Find pair with the highest correlation
                idx = np.unravel_index(np.argmax(corr_matrix.values), corr_matrix.shape)
                row_idx, col_idx = map(int, idx)
                feat1, feat2 = str(corr_matrix.columns[row_idx]), str(corr_matrix.columns[col_idx])

                # Compare their importance and remove the less important one
                stat1 = stat_df.loc[stat_df['Feature'] == feat1, 'importance'].values[0]
                stat2 = stat_df.loc[stat_df['Feature'] == feat2, 'importance'].values[0]
                to_remove = feat2 if abs(stat1) >= abs(stat2) else feat1
                remaining_features.remove(to_remove)

            result_dict[csv_name] = remaining_features
            if Settings.show_visualization:
                VisualizationManager.plot_filtered_features(merged_df, sig_features, remaining_features, csv_name)

        return result_dict


class Spearman(StatisticHelper):
    """
    Computes Spearman correlation between numeric features and a clinical biomarker
    for each dataset/CSV. Provides the resulting statistics as DataFrames.
    """
    @property
    def statistic(self) -> dict[str, pd.DataFrame]:
        """
        Returns the computed Spearman statistics per CSV/image.
        Returns: dict[str, pd.DataFrame].
        """
        return self._spearman_dfs()

    def _spearman_dfs(self) -> dict[str, pd.DataFrame]:
        """
        Computes Spearman correlation and p-values for all numeric features
        against the clinical biomarker, per CSV/image.
        Returns: dict[str, pd.DataFrame].
        """
        from scipy.stats import spearmanr

        result = {}

        for csv_name, df in self._merged_csvs().items():
            stats = []

            for col in self._numeric_features(df):
                # Only consider rows where both feature and biomarker are not NaN
                valid = df[col].notna() & self.clinical_df[Settings.clinical_biomarker_name].notna()

                # Standardize both feature column and clinical biomarker
                x = self._standardize(df)[col][valid]
                y = self._standardize(self.clinical_df)[Settings.clinical_biomarker_name][valid]

                # Compute Spearman correlation and p-value
                corr, p = spearmanr(x, y)
                stats.append({"Feature": col, "importance": corr, "p_value": p})

            # Convert to DataFrame for this CSV
            result[csv_name] = pd.DataFrame(stats)

        return result


class KruskalWallis(StatisticHelper):
    """
    Computes Kruskal-Wallis H-test for each numeric feature across clinical groups.
    Provides a statistical ranking of features based on group differences.
    """
    @property
    def statistic(self) -> dict[str, pd.DataFrame]:
        """
        Returns the computed Kruskal-Wallis statistics per CSV/image.
        Returns: dict[str, pd.DataFrame].
        """
        return self._kw_dfs()

    def _kw_dfs(self) -> dict[str, pd.DataFrame]:
        """
        Computes Kruskal-Wallis H-test for all numeric features, grouping by clinical category.
        Returns: dict[str, pd.DataFrame].
        """
        from scipy.stats import kruskal

        results = {}

        for csv_name, df in self._merged_csvs().items():
            rows = []

            for feature in self._numeric_features(df):
                # Combine feature values with clinical category, drop NaNs
                data = pd.concat([df[feature], self.clinical_df["Category"]], axis=1).dropna()

                # Determine unique categories
                categories = np.sort(data["Category"].dropna().unique())

                # Prepare list of groups per category
                groups = [data[data["Category"] == c][feature] for c in categories]

                # Compute Kruskal-Wallis test
                stat, p = kruskal(*groups)
                rows.append({"Feature": feature, "importance": stat, "p_value": p})

            # Store results as DataFrame per CSV
            results[csv_name] = pd.DataFrame(rows)

        return results


class LassoHelper:
    """
    Helper class for performing Lasso regression with feature standardization.
    Provides train/test splitting, model fitting, prediction, and feature importance extraction.
    """
    def __init__(self, features: pd.DataFrame, labels: pd.Series,
                 alpha: float = 0.01, test_size: float = 0.2) -> None:
        """
        Initializes the LassoHelper.
        Args:
            features (pd.DataFrame): Feature matrix.
            labels (pd.Series): Target variable.
            alpha (float): Regularization strength for Lasso.
            test_size (float): Fraction of data to use for the test set.
        """
        self.features = features
        self.labels = labels
        self.alpha = alpha
        self.test_size = test_size

        # Train/test split placeholders
        self.x_train = self.x_test = None
        self.y_train = self.y_test = None

        # Scaled feature placeholders
        self.x_train_scaled = self.x_test_scaled = None

        self.model = None
        self.scaler = None

    def split(self) -> None:
        """
        Splits the features and labels into training and testing sets.
        """
        from sklearn.model_selection import train_test_split

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=self.test_size, random_state=42)

    def fit(self) -> None:
        """
        Standardizes features and fits the Lasso regression model on the training set.
        """
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler

        self.split()
        self.scaler = StandardScaler()

        # Fit scaler on training data and transform both train and test sets
        self.x_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.x_train),
            columns=self.x_train.columns,
            index=self.x_train.index)

        self.x_test_scaled = pd.DataFrame(
            self.scaler.transform(self.x_test),
            columns=self.x_test.columns,
            index=self.x_test.index)

        # Fit Lasso regression model
        self.model = Lasso(alpha=self.alpha, random_state=42, max_iter=10000)
        self.model.fit(self.x_train_scaled, self.y_train)

    def predict(self):
        """
        Predicts target values for the test set.
        Returns: np.ndarray: Predicted values.
        """
        return self.model.predict(self.x_test_scaled)

    def feature_importance(self) -> pd.DataFrame:
        """
        Returns the absolute Lasso coefficients as feature importance.
        Returns: pd.DataFrame: DataFrame with columns 'Feature' and 'importance', sorted descending.
        """
        return (
            pd.DataFrame({
                "Feature": self.x_train.columns,
                "importance": self.model.coef_})
            # Take absolute value because Lasso coefficients can be negative
            .assign(importance=lambda df: df["importance"].abs())
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))  # Reset index for clean numbering


class Lasso(Dataset):
    """
    Performs Lasso regression on radiomics features to predict a clinical biomarker.
    Can optionally filter features based on significance and visualize results.
    """
    def regression(self, significant: bool = False) -> dict[str, list]:
        """
        Performs Lasso regression for each CSV/image dataset and selects important features
        using the knee/elbow method on Lasso coefficients.
        Args: significant (bool): If True, only use features that are statistically significant.
        Returns: dict[str, list]: Keys are CSV/image names, values are lists of selected feature names.
        """
        # Load all merged CSVs
        merged = self._merged_csvs()
        # Compute Spearman statistics for all features
        statistic = Spearman().statistic

        # Optionally filter to a single image
        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        labels = self.clinical_df[Settings.clinical_biomarker_name]

        result_dict = {}
        for name, df in merged.items():
            # Optionally select only statistically significant features
            if significant and name in statistic:
                features = df[self._significant_features(statistic[name])]
            else:
                features = df[self._numeric_features(df)]

            # Remove rows with NaNs in features or labels
            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels_ = features[valid_idx], labels[valid_idx]

            # Fit Lasso regression with standardized features
            lasso_helper = LassoHelper(features, labels_)
            lasso_helper.fit()

            # Select features above knee/elbow point of absolute Lasso coefficients
            knee, threshold, selected_features, importance = ThresholdSelectionFeatures(
                lasso_helper.feature_importance()).select_features_by_knee()
            result_dict[name] = selected_features

            # Optional visualization of regression and feature selection
            if Settings.show_visualization:
                VisualizationManager.plot_lasso_regression(lasso_helper.y_test, lasso_helper.predict(),
                                                           knee, importance, threshold, name)

        return result_dict


class RandomForestHelper:
    """
    Helper class for performing Random Forest classification or regression.
    Provides train/test splitting, model fitting, prediction, and feature importance extraction.
    """
    def __init__(self, features: pd.DataFrame, labels: pd.Series, task: str = "classification",
                 n_trees: int = 200, test_size: float = 0.2) -> None:
        """
        Initializes the RandomForestHelper.
        Args:
            features (pd.DataFrame): Feature matrix.
            labels (pd.Series): Target variable.
            task (str): "classification" or "regression".
            n_trees (int): Number of trees in the Random Forest.
            test_size (float): Fraction of data to use for the test set.
        """
        self.features = features
        self.labels = labels
        self.task = task
        self.n_trees = n_trees
        self.test_size = test_size

        # Placeholders for train/test splits
        self.x_train = self.x_test = self.y_train = self.y_test = None

        self.model = None
        self.feature_importance_df = None

    def split(self) -> None:
        """
        Splits the features and labels into training and testing sets.
        For classification, maintains stratified class distribution.
        """
        from sklearn.model_selection import train_test_split

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=self.test_size,
            stratify=self.labels if self.task == "classification" else None, random_state=42)

    def fit(self) -> None:
        """
        Fits a Random Forest model (classification or regression) on the training data.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        self.split()
        if self.task == "classification":
            self.model = RandomForestClassifier(n_estimators=self.n_trees, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=self.n_trees, random_state=42)

        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        """
        Predicts target values for the test set.
        Returns: np.ndarray: Predicted values.
        """
        return self.model.predict(self.x_test)

    def feature_importance(self) -> pd.DataFrame:
        """
        Returns the feature importance values from the Random Forest model.
        Returns: pd.DataFrame: DataFrame with columns 'Feature' and 'importance', sorted descending.
        """
        self.feature_importance_df = pd.DataFrame({
            "Feature": self.features.columns,
            "importance": self.model.feature_importances_}).sort_values("importance", ascending=False)

        return self.feature_importance_df


class RandomForest(Dataset):
    """
    Performs Random Forest classification or regression on radiomics features.
    Can optionally filter features based on significance and visualize results.
    """
    def classification(self, significant: bool = False) -> dict[str, list]:
        """
        Performs Random Forest classification for each CSV/image dataset and selects
        important features using the knee/elbow method on feature importance values.
        Args: significant (bool): If True, only use features that are statistically significant.
        Returns: dict[str, list]: Keys are CSV/image names, values are lists of selected feature names.
        """
        merged = self._merged_csvs()
        statistic = KruskalWallis().statistic

        # Optionally filter to a single image
        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        labels = self.clinical_df["Category"]
        result_dict = {}

        for name, df in merged.items():
            # Optionally select only statistically significant features
            if significant and name in statistic:
                features = df[self._significant_features(statistic[name])]
            else:
                features = df[self._numeric_features(df)]

            # Remove rows with NaNs in features or labels
            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels_ = features[valid_idx], labels[valid_idx]

            # Fit Random Forest classifier
            rf_helper = RandomForestHelper(pd.DataFrame(features), labels_)
            rf_helper.fit()

            # Select features above knee/elbow point of importance values
            knee, threshold, selected_features, importance = ThresholdSelectionFeatures(
                rf_helper.feature_importance()).select_features_by_knee()
            result_dict[name] = selected_features

            # Optional visualization
            if Settings.show_visualization:
                VisualizationManager.plot_rf_classification(rf_helper.y_test, rf_helper.predict(),
                                                            knee, importance, threshold, name)

        return result_dict

    def regression(self, significant: bool = False) -> dict[str, list]:
        """
        Performs Random Forest regression for each CSV/image dataset and selects
        important features using the knee/elbow method on feature importance values.
        Args: significant (bool): If True, only use features that are statistically significant.
        Returns: dict[str, list]: Keys are CSV/image names, values are lists of selected feature names.
        """
        merged = self._merged_csvs()
        statistic = Spearman().statistic

        # Optionally filter to a single image
        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        labels = self.clinical_df[Settings.clinical_biomarker_name]
        result_dict = {}

        for name, df in merged.items():
            # Optionally select only statistically significant features
            if significant and name in statistic:
                features = df[self._significant_features(statistic[name])]
            else:
                features = df[self._numeric_features(df)]

            # Remove rows with NaNs in features or labels
            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels_ = features[valid_idx], labels[valid_idx]

            # Fit Random Forest regressor
            rf_helper = RandomForestHelper(pd.DataFrame(features), labels_, task="regression")
            rf_helper.fit()

            # Select features above knee/elbow point of importance values
            knee, threshold, selected_features, importance = ThresholdSelectionFeatures(
                rf_helper.feature_importance()).select_features_by_knee()
            result_dict[name] = selected_features

            # Optional visualization
            if Settings.show_visualization:
                VisualizationManager.plot_rf_regression(rf_helper.y_test, rf_helper.predict(),
                                                        knee, importance, threshold, name)

        return result_dict


class MutualInformation(Dataset):
    """
    Mutual Information based feature selection for classification and regression.
    Features are ranked and selected using the knee/elbow method.
    """
    def _run(self, mi_function, labels: pd.Series) -> dict[str, list]:
        """
        General function for running classification or regression MI.
        Args:
            mi_function: classification or regression.
            labels: pandas Series of labels.
        Returns: dict[str, list]: Keys are CSV/image names, values are lists of selected feature names.
        """
        merged = self._merged_csvs()
        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}

        result_dict = {}

        for name, df in merged.items():
            features = df[self._numeric_features(df)]

            # NaN filter
            valid_idx = features.notna().all(axis=1) & labels.notna()
            x = features[valid_idx]
            y = labels[valid_idx]

            # calculation of MI
            mi_values = mi_function(x, y, random_state=42)

            mi_df = pd.DataFrame({
                "Feature": x.columns,
                "importance": mi_values
            }).sort_values("importance", ascending=False)

            # knee / elbow selection
            knee, threshold, selected_features, importance = (
                ThresholdSelectionFeatures(mi_df)
                .select_features_by_knee())

            result_dict[name] = selected_features

            if Settings.show_visualization:
                VisualizationManager.plot_mutual_information(knee, importance, threshold, name)

        return result_dict

    def classification(self) -> dict[str, list]:
        """
        Mutual Information feature selection for classification tasks.
        """
        from sklearn.feature_selection import mutual_info_classif

        labels = self.clinical_df["Category"]
        return self._run(mutual_info_classif, labels)

    def regression(self) -> dict[str, list]:
        """
        Mutual Information feature selection for regression tasks.
        """
        from sklearn.feature_selection import mutual_info_regression

        labels = self.clinical_df[Settings.clinical_biomarker_name]
        return self._run(mutual_info_regression, labels)


class MRMR(Dataset):
    """
    Helper class for feature selection using the mRMR (Minimum Redundancy
    Maximum Relevance) criterion.

    Supports:
    - classification using two different mRMR implementations
    - regression using continuous clinical biomarkers

    The class is designed to be consistent with the overall Dataset-based
    feature selection framework.
    """
    def classification_type1(self, significant: bool = False, n_top_features: int = 20) -> dict[str, list]:
        """
        Perform mRMR-based feature selection for classification using the
        `mrmr` package implementation.

        Parameters
        ----------
        significant : bool, optional
            If True, only statistically significant features are considered.
        n_top_features : int, optional
            Number of top features to select via mRMR.

        Returns
        -------
        dict[str, list]
            Dictionary mapping image name to selected feature names.
        """
        from mrmr import mrmr_classif

        merged = self._merged_csvs()
        statistic = KruskalWallis().statistic

        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        result_dict = {}

        for name, df in merged.items():
            # Merge radiomics features with clinical labels
            temp_df = pd.merge(df, self.clinical_df[['Patient ID', 'Category']],
                               left_on='patient', right_on='Patient ID')

            # Feature preselection
            if significant and name in statistic:
                features = temp_df[self._significant_features(statistic[name])]
            else:
                features = temp_df[self._numeric_features(df)]

            labels = temp_df['Category']

            # Keep only fully valid samples
            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels = features[valid_idx], labels[valid_idx]

            # mRMR expects a single DataFrame with labels included
            df_mrmr = self._standardize(features).copy()
            df_mrmr['Category'] = labels.values

            selected_features = mrmr_classif(
                X=df_mrmr.drop(columns='Category'),
                y=df_mrmr['Category'],
                K=n_top_features)

            if Settings.show_visualization:
                VisualizationManager.plot_mrmr(self._standardize(features),
                                               selected_features, name)

            result_dict[name] = selected_features

        return result_dict

    def classification_type2(self, significant: bool = False, n_top_features: int = 20) -> dict[str, list]:
        """
        Perform mRMR-based feature selection for classification using the
        `feature_engine` MRMR selector (MIQ criterion).

        Parameters
        ----------
        significant : bool, optional
            If True, only statistically significant features are considered.
        n_top_features : int, optional
            Maximum number of selected features.

        Returns
        -------
        dict[str, list]
            Dictionary mapping image name to selected feature names.
        """
        from feature_engine.selection import MRMR

        merged = self._merged_csvs()
        statistic = KruskalWallis().statistic

        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        result_dict = {}

        for name, df in merged.items():
            temp_df = pd.merge(df, self.clinical_df[['Patient ID', 'Category']],
                               left_on='patient', right_on='Patient ID')

            if significant and name in statistic:
                features = temp_df[self._significant_features(statistic[name])]
            else:
                features = temp_df[self._numeric_features(df)]

            labels = temp_df['Category']

            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels = features[valid_idx], labels[valid_idx]

            selector = MRMR(max_features=n_top_features)
            selector.fit(self._standardize(features), labels)

            selected_features = (selector.transform(self._standardize(features)).columns.tolist())

            if Settings.show_visualization:
                VisualizationManager.plot_mrmr(self._standardize(features),
                                               selected_features, name)

            result_dict[name] = selected_features

        return result_dict

    def regression(self, significant: bool = False, n_top_features: int = 20) -> dict[str, list]:
        """
        Perform mRMR-based feature selection for regression tasks using
        continuous clinical biomarkers.

        Parameters
        ----------
        significant : bool, optional
            If True, only statistically significant features are considered.
        n_top_features : int, optional
            Number of top features to select.

        Returns
        -------
        dict[str, list]
            Dictionary mapping image name to selected feature names.
        """
        from mrmr import mrmr_regression

        merged = self._merged_csvs()
        statistic = Spearman().statistic

        if Settings.image_name:
            merged = {Settings.image_name: merged[Settings.image_name]}
            statistic = {Settings.image_name: statistic[Settings.image_name]}

        result_dict = {}

        for name, df in merged.items():
            temp_df = pd.merge(df, self.clinical_df[['Patient ID', Settings.clinical_biomarker_name]],
                               left_on='patient', right_on='Patient ID')

            if significant and name in statistic:
                features = temp_df[self._significant_features(statistic[name])]
            else:
                features = temp_df[self._numeric_features(df)]

            labels = temp_df[Settings.clinical_biomarker_name]

            valid_idx = features.notna().all(axis=1) & labels.notna()
            features, labels = features[valid_idx], labels[valid_idx]

            df_mrmr = self._standardize(features).copy()
            df_mrmr[Settings.clinical_biomarker_name] = labels.values

            selected_features = mrmr_regression(
                X=df_mrmr.drop(columns=Settings.clinical_biomarker_name),
                y=df_mrmr[Settings.clinical_biomarker_name],
                K=n_top_features)

            if Settings.show_visualization:
                VisualizationManager.plot_mrmr(self._standardize(features),
                                               selected_features, name)

            result_dict[name] = selected_features

        return result_dict


class FeatureSelection:
    """
    Central orchestration class aggregating feature selection results
    from multiple statistical and machine-learning based methods.
    """
    def __init__(self):
        # ------------------------------------------------------------------
        # FEATURE SELECTION WITH CORRELATED IMAGE FEATURES (NO mRMR)
        # ------------------------------------------------------------------

        # Statistical tests selection
        self.selected_spearman = Spearman().best_features_by_stat()
        self.selected_kw = KruskalWallis().best_features_by_stat()

        # Best feature from each group
        self.best_one_group_spearman = Spearman().best_feature_from_group()
        self.best_one_group_kw = KruskalWallis().best_feature_from_group()

        # Mutual information selection
        self.mi_classification = MutualInformation().classification()
        self.mi_regression = MutualInformation().regression()

        # Random Forest
        self.rf_classification_all = RandomForest().classification()
        self.rf_classification_significant = RandomForest().classification(significant=True)
        self.rf_regression_all = RandomForest().regression()
        self.rf_regression_significant = RandomForest().regression(significant=True)

        # LASSO regression
        self.lasso_regression_all = Lasso().regression()
        self.lasso_regression_significant = Lasso().regression(significant=True)

        # ------------------------------------------------------------------
        # FEATURE SELECTION WITHOUT CORRELATED IMAGE FEATURES (mRMR)
        # ------------------------------------------------------------------

        # Correlation-filtered statistical features
        self.filtered_spearman = Spearman().relevant_features_by_corr()
        self.filtered_kw = KruskalWallis().relevant_features_by_corr()

        # mRMR classification
        self.mrmr_classification_1_all = MRMR().classification_type1()
        self.mrmr_classification_1_significant = MRMR().classification_type1(significant=True)
        self.mrmr_classification_2_all = MRMR().classification_type2()
        self.mrmr_classification_2_significant = MRMR().classification_type2(significant=True)

        # mRMR regression
        self.mrmr_regression_all = MRMR().regression()
        self.mrmr_regression_significant = MRMR().regression(significant=True)

        # Collection of all feature selection outputs
        self.all_dicts = [
            self.selected_kw, self.selected_spearman,
            self.best_one_group_spearman, self.best_one_group_kw,
            self.rf_classification_all, self.rf_classification_significant,
            self.rf_regression_all, self.rf_regression_significant,
            self.lasso_regression_all, self.lasso_regression_significant,
            self.mi_classification, self.mi_regression,
            self.filtered_spearman, self.filtered_kw,
            self.mrmr_classification_1_all, self.mrmr_classification_1_significant,
            self.mrmr_classification_2_all, self.mrmr_classification_2_significant,
            self.mrmr_regression_all, self.mrmr_regression_significant,
        ]

    @staticmethod
    def _print_count_tables(tables: dict[str, pd.DataFrame]) -> None:
        """Print feature occurrence tables filtered to features selected more than once."""
        for name, df in tables.items():
            print(f"\n{'-' * 40} {name} {'-' * 40}")
            print(df[df["Count"] > 1])

    def count_tables(self) -> None:
        """
        Count how frequently each feature is selected across all
        feature selection strategies and datasets.
        """
        from collections import Counter

        # All methods are expected to share identical dataset keys
        dataset_names = self.all_dicts[0].keys()
        result = {ds: Counter() for ds in dataset_names}

        # Aggregate feature selections across methods
        for selection_dict in self.all_dicts:
            for dataset_name, feature_list in selection_dict.items():
                result[dataset_name].update(feature_list)

        tables = {}
        total_methods = len(self.all_dicts)

        for dataset_name, counter in result.items():
            df = pd.DataFrame(counter.items(), columns=["Feature", "Count"])
            df = df.sort_values("Count", ascending=False).reset_index(drop=True)

            # Relative representation across all methods
            df["Ratio"] = df["Count"].astype(str) + f"/{total_methods}"
            df["Ratio[%]"] = round((df["Count"] / total_methods) * 100, 1)

            tables[dataset_name] = df

        self._print_count_tables(tables)


if __name__ == "__main__":
    fs_orchestrator = FeatureSelection()
    fs_orchestrator.count_tables()
