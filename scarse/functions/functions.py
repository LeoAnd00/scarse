import random
import warnings
import numpy as np
import gc
import pandas as pd
import torch
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import ExtraTreesClassifier
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
    r2_score,
    log_loss,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.exceptions import ConvergenceWarning
from transformers import AutoTokenizer, EsmModel

class ModelOptimization:
    """
    End-to-end framework for training and optimizing SCARSE on peptide sequence datasets using ESM-2 embeddings.

    The framework supports both regression and classification tasks
    with automated hyperparameter optimization using Optuna.
    """
    def __init__(
        self,
        data_path,
        seq_col = "sequence",
        score_col = ["score"],
        random_seed=42,
        emb_batch_size=64,
        classification=False,
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
        classification : bool, default=False
            Whether the task is a classification problem. If False, the
            framework performs regression.
        foundation : str, default="facebook/esm2_t33_650M_UR50D"
            Specify which foundation model to use.
        """

        # Parameters
        self.data_path = data_path
        self.random_seed = random_seed
        self.emb_batch_size = emb_batch_size
        self.seq_col = seq_col
        self.score_col = score_col
        self.model_name = foundation
        self.classification = classification
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
        string format, and converts target values to numeric form. For
        classification tasks, categorical labels are encoded using
        ``sklearn.preprocessing.LabelEncoder``.

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
        - Classification targets are automatically label encoded.
        """
        if self.data_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(self.data_path)
        df = pd.read_csv(self.data_path, sep=None, engine='python')
        required_cols = set([seq_col] + score_col)
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Input file must contain columns: {required_cols}. Found: {df.columns.tolist()}")
        
        # Store column names of the scores
        self.target_names = score_col

        # Convert sequence to string
        df["sequence"] = df[seq_col].astype(str)

        # Convert scores to float
        if not self.classification:
            for col in score_col:
                df[col] = df[col].astype(float)
        else:
            self.label_enc = {}
            self.n_classes = {}
            for col in score_col:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_enc[col] = le
                self.n_classes[col] = len(le.classes_)

        # Keep only sequence + score columns
        df = df[["sequence"] + score_col].copy()

        self.df = df

        # Create a mapping from sequence → list of scores if multiple columns
        if len(score_col) == 1:
            self.seq_to_score = dict(zip(df["sequence"], df[score_col[0]]))
        else:
            self.seq_to_score = dict(zip(df["sequence"], df[score_col].values.tolist()))

    def load_model(self):
        """
        Load the pretrained protein language model used for embeddings.

        The model and tokenizer are loaded from a local directory
        bundled with the application. The model is moved to the
        configured device (GPU if available) and set to evaluation mode.

        Notes
        -----
        - Uses the ESM-2 650M protein language model.
        - Model files must exist in the packaged resource directory
        ``models/esm-model``.
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
    
    def compute_roc_auc(self, y_test, proba, label_idx):
        """
        Compute the ROC-AUC score for binary or multiclass classification.

        The function automatically detects whether the classification
        problem is binary or multiclass based on the number of classes
        present in the trained model. For binary classification, the
        probability of the positive class is used. For multiclass tasks,
        a one-vs-rest (OvR) strategy with weighted averaging is applied.

        Parameters
        ----------
        model : sklearn.base.ClassifierMixin
            Trained classification model supporting the ``predict_proba`` method.
        X_test : np.ndarray
            Feature matrix of shape (n_samples, n_features) used for evaluation.
        y_test : array-like
            True class labels corresponding to ``X_test``.
        label_idx : int
            Index of current label of interest.

        Returns
        -------
        float
            Computed ROC-AUC score.

        Notes
        -----
        - Binary classification uses the probability of the positive class.
        - Multiclass classification uses weighted one-vs-rest ROC-AUC.
        """

        label_name = self.target_names[label_idx]
        le = self.label_enc[label_name]

        # Ensure consistent class order
        classes = le.transform(le.classes_)

        n_classes = self.n_classes[self.target_names[label_idx]]

        if n_classes == 2:
            positive_class_index = 1
            return roc_auc_score(y_test, proba[:, positive_class_index])

        else:
            # Binarize true labels
            y_test_bin = label_binarize(y_test, classes=classes)

            # Multiclass case: One-vs-Rest
            return roc_auc_score(
                y_test_bin,
                proba,
                average="weighted",
                multi_class="ovr"
            )

    def train(self, 
              folds=10, 
              random_seed=42, 
              n_trials=100, 
              optuna_print=True):
        """
        Run the full model optimization and training pipeline.

        This method performs the following steps:

        1. Load and preprocess the dataset
        2. Generate embeddings for all training sequences using the
        configured protein language model
        3. Create cross-validation folds
        4. Optimize hyperparameters of downstream models using Optuna
        5. Evaluate the optimized model using cross-validation metrics

        The downstream model depends on the task type:

        Regression
            GaussianProcessRegressor

        Classification
            ExtraTreesClassifier

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

        Classification metrics computed:
            - Accuracy
            - Balanced Accuracy
            - F1 score (weighted)
            - Matthews Correlation Coefficient
            - ROC-AUC

        Cross-validation is used to estimate performance while
        Optuna searches the hyperparameter space.
        """

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
        
        if self.classification:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_seed)
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=random_seed)

        seq_to_emb = {}
        for idx, seq in enumerate(self.training_sequences):
            seq_to_emb[seq] = X_train_temp[idx]

        seq_array = np.array(self.training_sequences)

        folds_per_label = {}

        for label_idx in range(n_targets):
            folds_list = []
            
            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(seq_array) if not self.classification else cv.split(seq_array, y_train[:, label_idx])):
            
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
                
        if not self.classification:
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
        else:
            top_model = "ExtraTrees"
            model_configs = {
                "ExtraTrees": {
                    "class": ExtraTreesClassifier,
                    "params": {
                        "n_estimators": ("int", 50, 300),
                        "max_depth": ("int", 2, 15),
                        "min_samples_split": ("int", 2, 10),
                        "min_samples_leaf": ("int", 1, 8),
                        "max_features": ("categorical", ["sqrt", "log2", None])
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
            self.final_loss_tracker = np.inf
            
            def objective(trial):

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
                    
                    model = ModelClass(**params)
                    
                    y_vall_all = []
                    y_pred_all = []
                    y_pred_prob_all = []
                    fold_scores = []

                    for fold_idx, dataset_dict in enumerate(folds_per_label[f"label_{label_idx}"]):
                        model_fold = clone(model)
                        train_df = dataset_dict["train"]
                        val_df = dataset_dict["validation"]

                        X_train_fold = np.vstack(train_df["sequence"])
                        X_val_fold = np.vstack(val_df["sequence"])

                        if not self.classification:
                            y_train_fold = np.vstack(train_df["score"]).ravel()
                            y_val_fold = np.vstack(val_df["score"]).ravel()

                            model_fold.fit(X_train_fold, y_train_fold)
                            y_pred = model_fold.predict(X_val_fold)

                            y_val_fold = np.asarray(val_df["score"]).ravel()

                            y_vall_all.extend(y_val_fold.tolist())
                            y_pred_all.extend(y_pred.tolist())

                        else:
                            y_train_fold = train_df["score"].astype(int)
                            y_val_fold = val_df["score"].astype(int)

                            model_fold.fit(X_train_fold, y_train_fold)

                            y_pred = model_fold.predict(X_val_fold)
                            y_proba = model_fold.predict_proba(X_val_fold)
                            label_enc = self.label_enc[self.target_names[label_idx]]
                            loss = log_loss(y_val_fold, y_proba, labels=label_enc.transform(label_enc.classes_))
                            
                            y_vall_all.extend(y_val_fold.tolist())
                            y_pred_all.extend(y_pred.tolist())
                            y_pred_prob_all.append(y_proba)

                            fold_scores.append(loss)

                    if not self.classification:
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
                    else:
                        final_loss = np.mean(fold_scores)
                        if final_loss < self.final_loss_tracker:

                            acc = accuracy_score(y_vall_all, y_pred_all)
                            bacc = balanced_accuracy_score(y_vall_all, y_pred_all)
                            labels = self.label_enc[self.target_names[label_idx]].transform(self.label_enc[self.target_names[label_idx]].classes_)
                            f1 = f1_score(y_vall_all, y_pred_all, average="weighted", labels=labels)
                            mcc = matthews_corrcoef(y_vall_all, y_pred_all)
                            y_pred_prob_all = np.vstack(y_pred_prob_all)
                            roc_auc = self.compute_roc_auc(y_vall_all, y_pred_prob_all, label_idx)
                            
                            self.final_metrics[self.target_names[label_idx]] = {
                                "Accuracy": float(acc),
                                "Balanced_Accuracy": float(bacc),
                                "F1_weighted": float(f1),
                                "MCC": float(mcc),
                                "ROC_AUC": float(roc_auc)
                            }
                            self.final_loss_tracker = final_loss

                        return final_loss
                except Exception as e:
                    print("Trial failed:", e)
                    raise optuna.TrialPruned()

            optuna.logging.set_verbosity(optuna.logging.ERROR) 
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_seed))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=optuna_print)
            
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

        The method supports both regression and classification tasks.

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
            - Predicted values (regression) or predicted labels (classification)
            - Class probabilities (classification only)

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
        5. Train the optimized model for each target label
        6. Generate predictions for test sequences

        Classification outputs include:

            - Predicted class label
            - Probability for each class

        Regression outputs include:

            - Predicted numeric score for each target.
        """

        df = pd.read_csv(test_seqs_path, sep=None, engine='python')
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

            if not self.classification:
                pred_scores = model.predict(X_test)
                df_pred[f'pred_{label}'] = pred_scores
            else:
                pred_labels = model.predict(X_test)

                # probabilities (if available)
                if hasattr(model, "predict_proba"):
                    pred_proba = model.predict_proba(X_test)
                    classes = model.classes_
                else:
                    # fallback if classifier has no predict_proba
                    pred_proba = None
                    classes = None

                # inverse transform labels
                le = self.label_enc[self.target_names[label_idx]]
                pred_labels_tra = le.inverse_transform(pred_labels)

                # Save predicted label
                df_pred[f'pred_{label}'] = pred_labels_tra
                df_pred[f'pred_encoded_{label}'] = pred_labels

                # Save probabilities per class
                if pred_proba is not None:
                    for i, cls in enumerate(classes):
                        df_pred[f'prob_{label}_{cls}'] = pred_proba[:, i]

        return df_pred
    