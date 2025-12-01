import pandas as pd

class DataCleaner:
    def clean(self, df: pd.DataFrame):
        result = {}

        # Missing values
        missing = df.isnull().sum()
        result["missing_values"] = missing.to_dict()

        # Fill missing with median
        df_cleaned = df.fillna(df.median(numeric_only=True))

        # Remove duplicates
        duplicates = df.duplicated().sum()
        df_cleaned = df_cleaned.drop_duplicates()

        result["duplicates_removed"] = int(duplicates)
        result["cleaned_df"] = df_cleaned

        return result
