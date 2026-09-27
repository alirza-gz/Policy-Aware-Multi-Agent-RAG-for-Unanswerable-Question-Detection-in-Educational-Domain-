In data mining courses, clustering is used to segment customers, documents, or events without predefined class labels. Cluster quality depends on features, distance, and the algorithm's inductive bias.

Hierarchical clustering builds a tree of merges or splits. Agglomerative hierarchical clustering starts with each point alone and merges closest clusters until a stopping rule is met.

A dendrogram visualizes hierarchical merges. Cutting the dendrogram at a height yields a flat partition with a chosen number of clusters.

Density-based methods such as DBSCAN group points in dense regions and can label sparse points as noise. Unlike k-means, DBSCAN does not require specifying the number of clusters in advance.

The silhouette score measures how similar a point is to its own cluster compared with the nearest other cluster. Average silhouette width is often used to compare candidate values of k.

Cluster labels are arbitrary identifiers. Two runs can discover the same partition with permuted label numbers; evaluation should compare partitions, not raw label integers.
