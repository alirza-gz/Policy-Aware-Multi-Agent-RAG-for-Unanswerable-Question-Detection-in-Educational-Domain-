Classification accuracy is the fraction of predictions that match the true labels. Accuracy can be misleading on imbalanced datasets where a majority-class classifier scores well.

Precision is the fraction of predicted positive cases that are truly positive. Recall is the fraction of truly positive cases that the model correctly identifies. There is typically a trade-off between precision and recall.

The F1 score is the harmonic mean of precision and recall. It is often used when both false positives and false negatives matter and class imbalance makes accuracy uninformative.

A confusion matrix tabulates true positives, false positives, true negatives, and false negatives. Many scalar metrics can be derived from the confusion matrix alone.

A receiver operating characteristic (ROC) curve plots true positive rate against false positive rate across decision thresholds. The area under the ROC curve (AUC) summarizes ranking quality.

For regression tasks, mean squared error and mean absolute error are common loss summaries. Mean absolute error is less sensitive to large outliers than squared error.
