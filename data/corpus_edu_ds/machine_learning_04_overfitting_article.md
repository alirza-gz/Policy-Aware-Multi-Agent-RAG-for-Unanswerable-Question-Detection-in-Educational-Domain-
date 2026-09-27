Regularization adds a penalty on model complexity to the training objective. L2 regularization penalizes the sum of squared weights; L1 regularization penalizes the sum of absolute weights and can drive some weights to zero.

Early stopping ends training when validation performance stops improving, reducing the chance that continued optimization fits training noise. It is widely used with iterative learners such as neural networks and gradient boosting.

A learning curve plots training and validation scores against training set size or training iteration. A large persistent gap between training and validation performance suggests overfitting.

Data leakage occurs when information from the test distribution improperly influences training, for example by scaling features using the full dataset before splitting. Leakage produces optimistic offline metrics that fail in deployment.

Hyperparameter search should be confined to training and validation folds. Selecting hyperparameters on the test set converts the test set into a silent validation set and breaks the final evaluation contract.

Ensemble methods combine multiple base models to improve stability or accuracy. Bagging reduces variance by averaging bootstrap-trained models; boosting builds models sequentially to correct previous errors.
