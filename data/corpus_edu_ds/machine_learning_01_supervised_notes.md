Supervised learning trains a model on examples that include both input features and a known target label. The fitted model is then used to predict labels for new unlabeled inputs.

Classification predicts a discrete class label, such as spam versus not spam. Regression in the machine-learning sense predicts a continuous numeric target.

A training set is used to fit parameters. A validation set is used to compare candidate models or hyperparameters. A test set is used once for a final unbiased performance estimate.

Overfitting occurs when a model captures noise or idiosyncrasies of the training sample and then generalizes poorly to new data. Underfitting occurs when the model is too simple to capture the systematic pattern.

Cross-validation partitions the training data into folds so that each fold serves once as a temporary validation set. k-fold cross-validation is a standard educational procedure for estimating generalization under data scarcity.

Feature scaling methods such as standardization rescale numeric inputs so that algorithms sensitive to feature magnitude, including many distance-based methods, are not dominated by large-range variables.
