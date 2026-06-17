import pandas as pd

def validate_sequences(data_path, seq_col):

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    try:
        if data_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path, sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file at '{data_path}': {e}")
    
    # Check required columns exist
    missing_cols = []
    if seq_col not in df.columns:
        missing_cols.append(seq_col)
    
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")
    
    # Check for empty or null sequences
    if df[seq_col].isnull().any():
        raise ValueError(f"Column '{seq_col}' contains missing (NaN) values.")

    if (df[seq_col].astype(str).str.strip() == "").any():
        raise ValueError(f"Column '{seq_col}' contains empty sequences.")
    
    # Amino acid validation
    invalid_rows = []

    for idx, seq in enumerate(df[seq_col].astype(str)):

        seq = seq.strip().upper()

        # empty sequence
        if seq == "":
            invalid_rows.append(
                f"Row {idx + 1}: empty sequence"
            )
            continue

        invalid_chars = set(seq) - valid_aa

        if invalid_chars:

            invalid_rows.append(
                f"Row {idx + 1}: invalid amino acids "
                f"{sorted(invalid_chars)}"
            )

    if invalid_rows:

        preview = "<br>".join(invalid_rows[:10])

        if len(invalid_rows) > 10:
            preview += (
                f"<br>... and {len(invalid_rows)-10} more rows"
            )

        raise ValueError(f"Invalid amino acid sequences detected: {preview}")