# SCARSE: Small-sample Classification And Regression Solution for low-resource peptide Engineering
<p align="center">
  <img src="figures/scarse_banner.png" alt="Workflow Diagram" width="800"/>
</p>

## Abstract
Reliable estimation of downstream performance in low-data peptide machine learning is critical for guiding early-stage AI-driven peptide engineering, yet it is often unclear how to assess whether a model will be effective in iterative discovery settings. Here, we show that cross validation R² score can serve as a simple and robust proxy for predicting active learning workflow performance, enabling early-stage evaluation of model suitability for sequential peptide optimization. To support this, we introduce SCARSE, a machine learning framework combining ESM-2 protein language model embeddings with Gaussian process regression and extremely randomized trees classification, designed for low-resource peptide property prediction (20–500 training samples). We benchmark SCARSE across 23 peptide and small-protein datasets covering substitution and indel variants, antimicrobial peptides, cell-penetrating peptides, and toxic/non-toxic peptides. The protein language model approach significantly outperforms a hand-engineered descriptor baseline on substitution and indel tasks, while the two approaches achieve comparable performance on shorter peptide non-mutant datasets where simpler descriptors capture enough of the signal. In simulated active learning workflows, SCARSE consistently outperforms baseline and random sampling strategies, and notably we demonstrate that CV R² computed from as few as 50 labeled peptides can be sufficient to estimate final active learning endpoint performance, providing a practical, data-efficient criterion for deciding whether a given dataset combined with SCARSE is suitable for iterative peptide discovery. SCARSE is released as a pip package and is available via HuggingFace Spaces to facilitate integration into peptide engineering workflows.

## How SCARSE works

SCARSE is designed for peptide property prediction in low-data regimes by combining protein language model embeddings with classical machine learning methods.

The workflow consists of the following steps:

1. **Input data**
   - A CSV file containing peptide sequences and one or more target variables.
   - The sequence column (`seq_col`) should contain amino acid sequences.
   - The target column(s) (`score_col`) contain regression values or class labels.

2. **Sequence embedding**
   - Sequences are converted into numerical representations using the ESM-2 protein language model.

3. **Model selection**
   - Depending on the task:
     - **Regression** → Gaussian Process Regression  
     - **Classification** → Extremely Randomized Trees  
   - These models are chosen for robustness in small-sample settings.

4. **Hyperparameter optimization**
   - Models are tuned using cross-validation and Optuna-based optimization.
   - The number of folds and optimization trials can be controlled by the user.

5. **Training output**
   - Cross-validation performance metrics are returned.
   - A trained model environment is stored internally and reused for prediction.

6. **Prediction**
   - New sequences are embedded using the same pipeline.
   - The trained model generates predictions for each target variable.

---

## Notes and best practices

- **Call order matters**  
  You must run `scarse.train()` before calling `scarse.pred()`, as the trained model is stored internally.

- **Data quality is important**  
  - Ensure no missing or empty sequences  
  - Use consistent formatting (standard amino acid codes(ACDEFGHIKLMNPQRSTVWY))

- **Small datasets are supported**  
  SCARSE have been evaluated for datasets as small as ~20 samples, but performance generally improves with more data.

## Tested for Python version
- Python version == 3.12.10

## Setup
```
pip install scarse
```

## Usage

### For training on regression problem:
```
import scarse

scarse.train(data_path="../app/train.csv", 
             classification=False, 
             seq_col="sequence",
             score_col=["score"])
```
### For training on classification problem:
```
import scarse

scarse.train(data_path="../app/train.csv", 
             classification=True, 
             seq_col="sequence",
             score_col=["classes"])
```
### For predicting after model have been trained:
```
df_pred = scarse.pred(data_path="../app/test.csv", seq_col="sequence")
```

## Tutorials
See the following tutorial, structured as a Python notebook:
* [tutorial.ipynb](tutorial.ipynb)

## Correlate to active learning end-point performance 
Below we illustrate the relation between CV R² score and end-point active learning performance. <br>
The y-axis display how many times better performance SCARSE guided active learning delivers compared to random sampling when looking at the accumulation of top 10% of peptides. <br>
By comparing the CV R² score of your data to the corresponding figure below for your dataset size one can get and indication of how suitable your data is combined with SCARSE to perform active learning peptide engineering. <br>
Note that this can only be used as a guide to evaluate regression problem performance. <br>

<p align="center">
  <img src="figures/active_learning_performance.png" alt="Workflow Diagram" width="500"/>
</p>

## Citation
Coming soon!
