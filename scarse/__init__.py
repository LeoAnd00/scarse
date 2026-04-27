from .functions import functions as scarse

scarse_env = None

def train(data_path,
          classification,
          seq_col,
          score_col,
          random_seed=42,
          folds=10,
          n_trials=100, 
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

    scarse_env = scarse.ModelOptimization(data_path=data_path, 
                                          seq_col=seq_col,
                                          score_col=score_col,
                                          random_seed=random_seed, 
                                          classification=classification)

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

    df_pred = scarse_env.pred(data_path, seq_col)

    return df_pred
