Unsupervised learning finds structure in unlabeled data. Common tasks include clustering, dimensionality reduction, and density estimation.

Clustering assigns observations to groups so that items in the same group are more similar to each other than to items in other groups, under a chosen similarity measure.

K-means clustering partitions data into a fixed number k of clusters by iteratively assigning points to the nearest centroid and updating centroids as group means. The algorithm requires a numeric feature space and a chosen k.

Dimensionality reduction projects high-dimensional data into fewer coordinates while preserving selected structure. Principal component analysis (PCA) finds orthogonal directions of maximal variance.

A distance metric such as Euclidean distance defines geometric closeness in feature space. Changing the metric or the feature scaling can change unsupervised results substantially.

Evaluation of unsupervised models often relies on internal indices, stability across restarts, or downstream task utility, because there is no single ground-truth label for every problem.
