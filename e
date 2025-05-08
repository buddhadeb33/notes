import pandas as pd
import json
import pickle
import os

def read_intermediate_output_local(base_dir, run_id, ds_name, suffix="parquet", as_pandas=True, path_override=None):
    """
    Reads intermediate output from a local folder in SageMaker Studio.
    
    :param base_dir: Base local directory (e.g., "notebook")
    :param run_id: ID for the current run
    :param ds_name: Dataset name (file name without extension)
    :param suffix: File format (parquet, json, pkl, csv)
    :param as_pandas: Whether to return a pandas DataFrame (if applicable)
    :param path_override: Full path override
    :return: Loaded object (DataFrame or Python object)
    """
    if path_override:
        file_path = path_override
    else:
        file_path = os.path.join(base_dir, run_id, f"{ds_name}.{suffix}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if suffix == "parquet" and as_pandas:
        return pd.read_parquet(file_path)
    elif suffix == "csv" and as_pandas:
        return pd.read_csv(file_path)
    elif suffix == "json":
        with open(file_path, 'r') as f:
            return json.load(f)
    elif suffix == "pkl":
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file suffix: {suffix}")
