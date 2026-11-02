# Tutorial

The regression training and testing data are random subsets of the POLG_PESV dataset.<br>
The "sequence" column contains the protein sequence of each variant.<br>
The "mutant" column describes the mutation(s) of each variant.<br>
The "score" column is the original endpoint of POLG_PESV.<br>
The "score_2" column is a copy of the "score" column with Gaussian noise added (std = 0.1).<br>
The "score_shuffled" column (training data only) is the "score" column randomly permuted across sequences (seed = 42). It serves as a negative control: a model trained on it should show no real predictive performance (R² ≈ 0) on the test set.