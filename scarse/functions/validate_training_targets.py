import pandas as pd

def validate_training_targets(data_path, score_col, classification):

    try:
        if data_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path, sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file at '{data_path}': {e}")
    
    target_cols = [score_col] if isinstance(score_col, str) else score_col
    
    # Check required columns exist
    missing_cols = []

    for col in target_cols:
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")

    
    for col in target_cols:
        if df[col].isnull().any():
            raise ValueError(f"Target column '{col}' contains missing (NaN) values.")
        
    # Regression validation
    if not classification:

        for col in target_cols:

            if not pd.api.types.is_numeric_dtype(df[col]):

                raise ValueError(f"Regression target '{col}' must contain numeric values.")

            if df[col].nunique() < 2:

                raise ValueError(f"Regression target '{col}' contains fewer than 2 unique values.")

    # Classification validation
    elif classification:

        for col in target_cols:

            n_unique = df[col].nunique()

            if n_unique < 2:

                raise ValueError(f"Classification target '{col}' contains only one class (requires at least 2 classes)")

            if n_unique == len(df):

                raise ValueError(f"Classification target '{col}' contains all unique values (looks like an ID column).")

            class_counts = df[col].value_counts()

            tiny_classes = class_counts[class_counts < 2]

            if len(tiny_classes) > 0:

                raise ValueError(f"Classification target '{col}' contains classes with fewer than 2 samples: {list(tiny_classes.index)}")
            