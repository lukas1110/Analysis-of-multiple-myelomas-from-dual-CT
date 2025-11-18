from feature_selection_helper import *


class Dataset:
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    @staticmethod
    def _read_clinical(path: str) -> pd.DataFrame:
        return pd.read_csv(path, encoding='cp1252')

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=['number']).columns.tolist()

    @staticmethod
    def _significant_features(df: pd.DataFrame) -> list[str]:
        return df.loc[df['p_value'] < 0.05, 'Feature'].tolist()

    @staticmethod
    def _clinical_stage(clinical_df: pd.DataFrame) -> pd.DataFrame:
        pd.set_option('future.no_silent_downcasting', True)
        clinical_df['Stage'] = clinical_df['ISS classification'].replace(
            {'Stage 1': int(1), 'Stage 2': int(2), 'Stage 3': int(3)})
        return clinical_df.dropna(subset=['Stage'])

    @staticmethod
    def extract_group(feature_name: str) -> str:
        patterns = [r"^gradient_firstorder", r"^gradient_glcm", r"^gradient_glrlm",
                    r"^gradient_glszm", r"^gradient_gldm", r"^gradient_ngtdm",
                    r"^original_firstorder", r"^original_glcm", r"^original_glrlm",
                    r"^original_glszm", r"^original_gldm", r"^original_ngtdm", r"^shape"]

        for p in patterns:
            if re.match(p, feature_name):
                return p.replace("^", "")
        return "other"

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df[self._numeric_features(df)]),
                            columns=df[self._numeric_features(df)].columns,
                            index=df[self._numeric_features(df)].index)

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
    def __init__(self, base_path: str) -> None:
        super().__init__(base_path)


