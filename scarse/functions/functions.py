import random
import warnings
import time
import numpy as np
import gc
import pandas as pd
import torch
import optuna
import scipy.optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    DotProduct
)
from sklearn.base import clone
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from transformers import AutoTokenizer, EsmModel


# ----------------------------------------------------------------------
# Optuna run-time limits for the GPR hyperparameter search
# ----------------------------------------------------------------------
# A single GPR fit can occasionally get stuck optimising its kernel
# hyperparameters. Two optional wall-clock limits keep that from stalling the
# whole optimisation, without changing any result unless a limit is reached:
#
#   trial_timeout - seconds one Optuna trial (all CV folds) may take. Checked
#                   inside the kernel optimiser, so it also interrupts a single
#                   fit that is stuck. The trial is then discarded (pruned).
#   study_timeout - seconds the whole Optuna study may take. When reached, no
#                   new trials start and the best trial so far is used.
#
# Both default to None (no limit). Every time a limit is reached it is printed,
# so it can be reported.

#: Deadline (time.monotonic()) for the trial currently being evaluated, or None.
#: Module level on purpose: sklearn's clone() deep-copies estimator parameters,
#: so a deadline stored on the optimiser object would not reach the clones.
_FIT_DEADLINE = None


class _TrialTimeLimitExceeded(Exception):
    """Raised inside a GPR fit when the current Optuna trial runs out of time."""


def _time_limited_lbfgs(obj_func, initial_theta, bounds):
    """sklearn's default GPR optimiser (L-BFGS-B), plus the trial deadline.

    Makes exactly the call sklearn's built-in ``fmin_l_bfgs_b`` makes, so the
    fit is identical when no deadline is set.
    """
    def wrapped(theta, *args, **kwargs):
        if _FIT_DEADLINE is not None and time.monotonic() > _FIT_DEADLINE:
            raise _TrialTimeLimitExceeded()
        return obj_func(theta, *args, **kwargs)

    res = scipy.optimize.minimize(wrapped, initial_theta, method="L-BFGS-B",
                                  jac=True, bounds=bounds)
    return res.x, res.fun


class ModelOptimization:
    """
    End-to-end framework for training and optimizing SCARSE on peptide sequence
    datasets using ESM-2 embeddings.

    The framework performs regression with automated hyperparameter
    optimization using Optuna.
    """
    def __init__(
        self,
        data_path,
        seq_col = "sequence",
        score_col = ["score"],
        random_seed=42,
        emb_batch_size=64,
        foundation="facebook/esm2_t33_650M_UR50D"):
        """
        Initialize the model optimization framework.

        This constructor configures the optimization environment, sets random
        seeds for reproducibility, and prepares internal variables required
        for data preprocessing, embedding extraction, and model optimization.

        Parameters
        ----------
        data_path : str
            Path to the input CSV file containing sequences and target values.
        seq_col : str, default="sequence"
            Name of the column containing amino acid sequences.
        score_col : list[str], default=["score"]
            Column name(s) containing target variables.
        random_seed : int, default=42
            Random seed used for Python, NumPy, and PyTorch to ensure
            reproducible experiments.
        emb_batch_size : int, default=64
            Batch size used during sequence embedding generation.
        foundation : str, default="facebook/esm2_t33_650M_UR50D"
            Specify which foundation model to use.
        """

        # Parameters
        self.data_path = data_path
        self.random_seed = random_seed
        self.emb_batch_size = emb_batch_size
        self.seq_col = seq_col
        self.score_col = [score_col] if isinstance(score_col, str) else score_col
        self.model_name = foundation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        self.df = None
        self.seq_to_score = {}
        self.training_sequences = []
        self.test_sequences = []

        # Seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def prep_data(self, seq_col="sequence", score_col=["score"]):
        """
        Load and preprocess a sequence dataset from a CSV file.

        The method validates required columns, converts sequence data to
        string format, and converts target values to numeric form.

        A mapping from sequence to target value(s) is also created for
        efficient lookup during model training.

        Parameters
        ----------
        seq_col : str, default="sequence"
            Name of the column containing amino acid sequences.
        score_col : list[str], default=["score"]
            Column name(s) containing target variables.

        Raises
        ------
        ValueError
            If required columns are missing from the dataset.

        Notes
        -----
        - Supports both single-target and multi-target prediction.
        """
        if self.data_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(self.data_path)
        else:
            df = pd.read_csv(self.data_path, sep=None, engine='python')
        required_cols = set([seq_col] + score_col)
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Input file must contain columns: {required_cols}. Found: {df.columns.tolist()}")

        # Store column names of the scores
        self.target_names = score_col

        # Convert sequence to string
        df["sequence"] = df[seq_col].astype(str)

        # Convert scores to float
        for col in score_col:
            df[col] = df[col].astype(float)

        # Keep only sequence + score columns
        df = df[["sequence"] + score_col].copy()

        self.df = df

        # Create a mapping from sequence -> list of scores if multiple columns
        if len(score_col) == 1:
            self.seq_to_score = dict(zip(df["sequence"], df[score_col[0]]))
        else:
            self.seq_to_score = dict(zip(df["sequence"], df[score_col].values.tolist()))

    def load_model(self):
        """
        Load the pretrained protein language model used for embeddings.

        The model and tokenizer are loaded and moved to the configured device
        (GPU if available) and set to evaluation mode.

        Notes
        -----
        - Uses the ESM-2 650M protein language model by default.
        - The model is only used for feature extraction (no gradient updates).
        """
        model_source = self.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=False)
        self.model = EsmModel.from_pretrained(model_source)

        self.model = self.model.to(self.device)
        self.model.eval()


    def compute_embeddings(self,
                           sequences,
                           batch_size=None):
        """
        Convert amino acid sequences into numerical embeddings using
        a pretrained protein language model, ESM-2 650M.

        Each sequence is tokenized and passed through the transformer
        model. Residue-level embeddings from the final hidden layer
        are mean-pooled to produce a fixed-length vector representation.

        Parameters
        ----------
        sequences : list[tuple[str, str]] or list[str]
            Sequences to embed. Typically provided as ``(id, sequence)``
            tuples.
        batch_size : int or None, optional
            Batch size used during embedding computation. If None,
            ``self.emb_batch_size`` is used.

        Returns
        -------
        np.ndarray
            Array of embeddings with shape ``(n_sequences, embedding_dim)``.

        Notes
        -----
        - Embeddings are computed without gradient tracking.
        """
        if batch_size is None:
            batch_size = self.emb_batch_size

        foundation_embeddings = []
        n = len(sequences)

        for i in range(0, n, batch_size):
            batch = sequences[i:i+batch_size]
            labels, seqs = zip(*batch)

            encoded = self.tokenizer(
                list(seqs),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            last_layer = hidden_states[-1:]
            stacked = torch.stack(last_layer, dim=0)
            mean_layers = stacked.mean(dim=0)

            for j, seq in enumerate(seqs):
                mask = attention_mask[j].bool().to(mean_layers.device)
                seq_emb = mean_layers[j, mask].mean(dim=0)
                foundation_embeddings.append(seq_emb.cpu().numpy())

        all_embeddings = np.vstack(foundation_embeddings)

        return all_embeddings

    def initialize_training_set(self):
        """
        Initialize the training sequence list.

        This method extracts all sequences from the preprocessed dataset
        and stores them as the default training set used during
        embedding generation and model optimization.
        """
        self.training_sequences = self.df["sequence"].tolist()

    def train(self,
              folds=10,
              random_seed=42,
              n_trials=100,
              optuna_print=True,
              trial_timeout=None,
              study_timeout=None):
        """
        Run the full model optimization and training pipeline.

        This method performs the following steps:

        1. Load and preprocess the dataset
        2. Generate embeddings for all training sequences using the
        configured protein language model
        3. Create cross-validation folds
        4. Optimize hyperparameters of the downstream Gaussian process
        regression model using Optuna
        5. Evaluate the optimized model using cross-validation metrics

        Hyperparameters are optimized using Optuna with a
        Tree-structured Parzen Estimator (TPE) sampler.

        Parameters
        ----------
        folds : int, default=10
            Number of cross-validation folds used during optimization.
        random_seed : int, default=42
            Random seed used for reproducibility across Python,
            NumPy, and PyTorch.
        n_trials : int, default=100
            Number of Optuna trials used to search for optimal
            hyperparameters.
        optuna_print : bool, default=True
            Whether to display Optuna progress bars during optimization.
        trial_timeout : float or None, default=None
            Wall-clock seconds one Optuna trial (all CV folds) may take before
            it is discarded. Checked inside the GPR kernel optimiser, so it also
            interrupts a single fit that is stuck. None = no limit.
        study_timeout : float or None, default=None
            Wall-clock seconds the whole Optuna study (per target) may take;
            when reached, the best trial so far is used. None = no limit.

        Returns
        -------
        dict
            Dictionary containing final performance metrics for each
            target label.

        Attributes Created
        ------------------
        all_best_models : dict
            Mapping from target label name to the best trained model.
        final_metrics : dict
            Final evaluation metrics for each target label.
        y_train : np.ndarray
            Training target array used for fitting models.
        n_targets : int
            Number of target variables in the dataset.

        Notes
        -----
        Regression metrics computed:
            - Mean Squared Error (MSE)
            - Root Mean Squared Error (RMSE)
            - Mean Absolute Error (MAE)
            - R² score
            - Spearman correlation

        Cross-validation is used to estimate performance while
        Optuna searches the hyperparameter space. The Optuna limits only
        change a result when they are actually reached (see the note at the
        top of this module).
        """
        global _FIT_DEADLINE

        self.prep_data(seq_col=self.seq_col, score_col=self.score_col)
        self.initialize_training_set()

        # Suppress convergence and feature name warnings
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        ## Seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Load foundation model
        self.load_model()

        label_seq = [(f"train_{i}", seq) for i, seq in enumerate(self.training_sequences)]
        X_train_temp = self.compute_embeddings(sequences=label_seq)

        y_train = []
        for s in self.training_sequences:
            y_train.append(self.seq_to_score[s])
        y_train = np.array(y_train)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        n_targets = y_train.shape[1]
        self.n_targets = n_targets
        n_samples = X_train_temp.shape[0]
        folds = min(folds, n_samples)

        cv = KFold(n_splits=folds, shuffle=True, random_state=random_seed)

        seq_to_emb = {}
        for idx, seq in enumerate(self.training_sequences):
            seq_to_emb[seq] = X_train_temp[idx]

        seq_array = np.array(self.training_sequences)

        folds_per_label = {}

        for label_idx in range(n_targets):
            folds_list = []

            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(seq_array)):

                X_train_fold = np.array([seq_to_emb[self.training_sequences[idx]] for idx in train_idx])
                X_val_fold = np.array([seq_to_emb[self.training_sequences[idx]] for idx in valid_idx])

                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_val_fold = scaler.transform(X_val_fold)

                train_df = {
                    'sequence': X_train_fold,
                    'score': y_train[train_idx, label_idx]
                }

                val_df = {
                    'sequence': X_val_fold,
                    'score': y_train[valid_idx, label_idx]
                }

                dataset_dict = {
                    'train': train_df,
                    'validation': val_df
                }

                # Store DatasetDict for each fold
                folds_list.append(dataset_dict)

            folds_per_label[f"label_{label_idx}"] = folds_list

        top_model = "GaussianProcessRegressor"
        model_configs = {
            "GaussianProcessRegressor": {"class": GaussianProcessRegressor, "params": {
                "alpha": ("float", 1e-10, 1e-6),
                "normalize_y": ("categorical", [True, False]),
                "kernel": ("categorical", [
                    RBF(),
                    Matern(),
                    RationalQuadratic(),
                    DotProduct()])
            }}
        }

        self.all_best_models = {}
        self.final_metrics = {}

        # Iterate over targets and models
        for label_idx in range(n_targets):

            cfg = model_configs[top_model]
            self.current_name = top_model

            ModelClass = cfg["class"]
            param_bounds = cfg["params"]

            self.mse_tracker = np.inf

            n_timed_out = [0]

            def objective(trial):
                global _FIT_DEADLINE

                _FIT_DEADLINE = (time.monotonic() + trial_timeout
                                 if trial_timeout else None)
                try:

                    params = {}
                    for name_, cfg in param_bounds.items():
                        ptype = cfg[0]
                        if ptype == "int":
                            params[name_] = trial.suggest_int(name_, cfg[1], cfg[2])
                        elif ptype == "float":
                            params[name_] = trial.suggest_float(name_, cfg[1], cfg[2])
                        elif ptype == "float_log":
                            params[name_] = trial.suggest_float(name_, cfg[1], cfg[2], log=True)
                        elif ptype == "categorical":
                            params[name_] = trial.suggest_categorical(name_, cfg[1])

                    # Same optimiser as sklearn's default, but it honours the
                    # trial deadline when trial_timeout is set.
                    model = ModelClass(**params, optimizer=_time_limited_lbfgs)

                    y_vall_all = []
                    y_pred_all = []

                    for fold_idx, dataset_dict in enumerate(folds_per_label[f"label_{label_idx}"]):
                        if _FIT_DEADLINE is not None and time.monotonic() > _FIT_DEADLINE:
                            raise _TrialTimeLimitExceeded()
                        model_fold = clone(model)
                        train_df = dataset_dict["train"]
                        val_df = dataset_dict["validation"]

                        X_train_fold = np.vstack(train_df["sequence"])
                        X_val_fold = np.vstack(val_df["sequence"])

                        y_train_fold = np.vstack(train_df["score"]).ravel()
                        y_val_fold = np.vstack(val_df["score"]).ravel()

                        model_fold.fit(X_train_fold, y_train_fold)
                        y_pred = model_fold.predict(X_val_fold)

                        y_val_fold = np.asarray(val_df["score"]).ravel()

                        y_vall_all.extend(y_val_fold.tolist())
                        y_pred_all.extend(y_pred.tolist())

                    mse = mean_squared_error(y_vall_all, y_pred_all)

                    if mse < self.mse_tracker:
                        rho, p = spearmanr(y_vall_all, y_pred_all)
                        self.final_metrics[self.target_names[label_idx]] = {
                            "MSE": float(mse),
                            "RMSE": float(np.sqrt(mse)),
                            "MAE": float(mean_absolute_error(y_vall_all, y_pred_all)),
                            "R2": float(r2_score(y_vall_all, y_pred_all)),
                            "Spearman correlation": float(rho)
                        }
                        self.mse_tracker = mse

                    return mse

                except _TrialTimeLimitExceeded:
                    n_timed_out[0] += 1
                    print(f"  Trial {trial.number} stopped after {trial_timeout} s "
                          f"(kernel={params.get('kernel')}, "
                          f"alpha={params.get('alpha', float('nan')):.3g}, "
                          f"normalize_y={params.get('normalize_y')})", flush=True)
                    raise optuna.TrialPruned()
                except Exception as e:
                    print("Trial failed:", e)
                    raise optuna.TrialPruned()
                finally:
                    _FIT_DEADLINE = None

            optuna.logging.set_verbosity(optuna.logging.ERROR)
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_seed))
            study.optimize(objective, n_trials=n_trials, timeout=study_timeout,
                           show_progress_bar=optuna_print)

            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            if study_timeout is not None and len(study.trials) < n_trials:
                print(f"  Study time limit ({study_timeout} s) reached after "
                      f"{len(study.trials)} trials.", flush=True)
            if n_timed_out[0]:
                print(f"  {n_timed_out[0]} trial(s) hit the trial time limit "
                      f"({trial_timeout} s).", flush=True)
            if not completed:
                raise RuntimeError(
                    "No Optuna trial completed within the given limits. Increase "
                    "trial_timeout / study_timeout, or leave them as None.")

            best_params = study.best_params

            best_model = ModelClass(**best_params)

            self.all_best_models[self.target_names[label_idx]] = best_model
            self.y_train = y_train

            del study
            gc.collect()

        return self.final_metrics

    def pred(self, test_seqs_path, seq_col="sequence"):
        """
        Generate predictions for new sequences using the optimized models.

        This method loads a dataset containing sequences, computes
        embeddings using the pretrained protein language model,
        and applies the optimized models obtained during training
        to generate predictions.

        Parameters
        ----------
        test_seqs_path : str
            Path to a CSV file containing sequences to predict.
        seq_col : str, default="sequence"
            Name of the column containing amino acid sequences.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing:

            - Input sequences
            - Predicted numeric score for each target.

        Raises
        ------
        ValueError
            If the required sequence column is not present in the input file.

        Notes
        -----
        Steps performed by this method:

        1. Load the input CSV file
        2. Validate required columns
        3. Compute embeddings for training and test sequences
        4. Standardize feature representations
        5. Fit the optimized model for each target label
        6. Generate predictions for test sequences
        """
        if test_seqs_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(test_seqs_path)
        else:
            df = pd.read_csv(test_seqs_path, sep=None, engine="python")
        required_cols = set([seq_col])
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Input file must contain columns: {required_cols}. Found: {df.columns.tolist()}")

        # Convert sequence to string
        df[seq_col] = df[seq_col].astype(str)

        label_seq = [(f"train_{i}", seq) for i, seq in enumerate(self.training_sequences)]
        X_train = self.compute_embeddings(sequences=label_seq)

        label_seq_test = [(f"test_{i}", seq) for i, seq in enumerate(df[seq_col])]
        X_test = self.compute_embeddings(sequences=label_seq_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        df_pred = pd.DataFrame({seq_col: df[seq_col]})

        for label_idx, label in enumerate(self.target_names):

            target_train = self.y_train[:, label_idx]

            model = clone(self.all_best_models[label])
            model.fit(X_train, target_train)

            pred_scores = model.predict(X_test)
            df_pred[f'pred_{label}'] = pred_scores

        return df_pred
