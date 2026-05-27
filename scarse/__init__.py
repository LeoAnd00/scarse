from .functions import functions as scarse
import pandas as pd

scarse_env = None

def train(data_path,
          classification,
          seq_col,
          score_col,
          random_seed=42,
          folds=10,
          n_trials=100, 
          foundation="facebook/esm2_t33_650M_UR50D",
          optuna_print=True):
    """
    Train and optimize models using sequence embeddings.

    This function initializes a global optimization environment and
    executes the full training pipeline, including dataset loading,
    embedding generation, hyperparameter optimization with Optuna,
    and cross-validation evaluation.

    The trained environment is stored globally and later used by
    :func:`pred` to generate predictions for new sequences.

    Parameters
    ----------
    data_path : str
        Path to the training dataset (CSV format).
    classification : bool
        Whether the task is a classification problem.

        - ``True`` → classification models are trained.
        - ``False`` → regression models are trained.
    seq_col : str
        Name of the column containing amino acid sequences.
    score_col : str or list of str
        Name(s) of the column(s) containing target values.
    random_seed : int, default=42
        Random seed used for reproducibility.
    folds : int, default=10
        Number of cross-validation folds used during optimization.
    n_trials : int, default=100
        Number of Optuna trials for hyperparameter optimization.
    foundation : str, default="facebook/esm2_t33_650M_UR50D"
        Specify which foundation model to use.
    optuna_print : bool, default=True
        Whether to display Optuna progress output.

    Returns
    -------
    dict
        Cross-validation performance metrics for each target variable.

    Notes
    -----
    This function creates a global environment (`scarse_env`) that
    stores the trained models and configuration. This environment
    must exist before calling :func:`pred`.
    """
    
    global scarse_env

    try:
        if data_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(data_path)
        df = pd.read_csv(data_path, sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file at '{data_path}': {e}")

    # Check required columns exist
    missing_cols = []
    if seq_col not in df.columns:
        missing_cols.append(seq_col)

    if isinstance(score_col, list):
        for col in score_col:
            if col not in df.columns:
                missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")
    
    # Check for empty or null sequences
    if df[seq_col].isnull().any():
        raise ValueError(f"Column '{seq_col}' contains missing (NaN) values.")

    if (df[seq_col].astype(str).str.strip() == "").any():
        raise ValueError(f"Column '{seq_col}' contains empty sequences.")
    
    target_cols = score_col if isinstance(score_col, list) else [score_col]
    for col in target_cols:
        if df[col].isnull().any():
            raise ValueError(f"Target column '{col}' contains missing (NaN) values.")

    scarse_env = scarse.ModelOptimization(data_path=data_path, 
                                          seq_col=seq_col,
                                          score_col=score_col,
                                          random_seed=random_seed, 
                                          classification=classification,
                                          foundation=foundation)

    cv_performance = scarse_env.train(folds, random_seed, n_trials, optuna_print)

    return cv_performance

def pred(data_path, 
         seq_col):
    """
    Generate predictions for new sequences using trained models.

    This function applies the models trained with :func:`train`
    to a new dataset of sequences and returns predicted values.

    Parameters
    ----------
    data_path : str
        Path to a CSV file containing sequences to predict.
    seq_col : str
        Name of the column containing amino acid sequences.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the input sequences and predicted
        outputs for each trained target variable.

    Raises
    ------
    RuntimeError
        If :func:`train` has not been executed before calling
        this function.

    Notes
    -----
    The function uses the globally stored training environment
    (`scarse_env`) created by :func:`train`.
    """

    global scarse_env
    if scarse_env is None:
        raise RuntimeError("You must call train() before pred().")

    df_pred = scarse_env.pred(data_path, seq_col)

    return df_pred
