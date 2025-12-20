from typing import Tuple, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class ColumnCategories:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    subtype: List[str]
    
def categorize_columns(
    df, max_unique_count: int = 2, max_unique_ratio: float = 0.05
) -> Tuple[List[str], List[str], List[str]]:
    df = df.convert_dtypes()
    n = len(df)

    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "string", "category", "bool", "boolean"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    # Reclassify numeric columns with low cardinality as categorical
    num_low_card = [
        col for col in numeric
        if df[col].nunique(dropna=True) <= max_unique_count
    ]
    numeric = [c for c in numeric if c not in num_low_card]
    categorical = sorted(set(categorical + num_low_card))

    return numeric, categorical, datetime_cols

def get_data_tabm(data:pd.DataFrame=None,):
    numeric_cols, categorical_cols, datetime_cols = categorize_columns(data)
    categories = ColumnCategories(
        numeric=numeric_cols,
        categorical=[col for col in categorical_cols],
        datetime=datetime_cols,
        subtype=[col for col in data.columns if col[0] == 'subtype' and col[1]][:17],
    )
    NNUM = len(categories.numeric)
    NCAT = len(categories.categorical)
    NDATETIME = len(categories.datetime)
    NLABEL = len(categories.subtype)
    CATCARDINALITIES = [data[col].nunique(dropna=True) for col in categories.categorical]
    return NNUM, NCAT, NDATETIME, NLABEL, CATCARDINALITIES, categories

def categorize_columns(
    df, max_unique_count: int = 2, max_unique_ratio: float = 0.05
) -> Tuple[List[str], List[str], List[str]]:
    df = df.convert_dtypes()
    n = len(df)

    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "string", "category", "bool", "boolean"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    # Reclassify numeric columns with low cardinality as categorical
    num_low_card = [
        col for col in numeric
        if df[col].nunique(dropna=True) <= max_unique_count
    ]
    numeric = [c for c in numeric if c not in num_low_card]
    categorical = sorted(set(categorical + num_low_card))

    return numeric, categorical, datetime_cols

def prepare_tabm_inputs(
    df: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    cat_as_codes: bool = True,
) -> tuple:
    """
    Returns (X_num, X_cat) numpy arrays suitable for TabM forward(num, cat).

    numeric -> float32
    categorical -> int64 codes (or original values if cat_as_codes=False)
    """
    X_num = df[numeric_cols].to_numpy(dtype=np.float32) if numeric_cols else np.empty((len(df), 0), np.float32)
    if categorical_cols:
        if cat_as_codes:
            df_cat = df[categorical_cols].apply(lambda s: s.astype('category').cat.codes)
            X_cat = df_cat.to_numpy(dtype=np.int64)
        else:
            # assume already int
            X_cat = df[categorical_cols].to_numpy()
    else:
        X_cat = np.empty((len(df), 0), np.int64)
    return X_num, X_cat