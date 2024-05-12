# Algorithm Name: K-Means Clustering

**Type:** Unsupervised Learning  
**Category:** Clustering  

## Overview

K-Means is a popular clustering algorithm used to partition data into K distinct, non-overlapping clusters based on similarity. It was developed from the notion of mean vectors and centroids in statistical data analysis.

## Key Concepts

- **Centroid:** The center point of each cluster.
- **Cluster Assignment:** Process of assigning data points to the nearest cluster based on distance.
- **Iteration:** Repeated updating of centroids and cluster assignments to minimize variations within clusters.

## How It Works

1. Initialize K centroids randomly.
2. Assign each data point to the nearest centroid.
3. Recompute centroids as the mean of data points in each cluster.
4. Repeat steps 2 and 3 until convergence or a set number of iterations is reached.

## Mathematical Model

Sum of squared distances between data points and their respective cluster centroids is minimized.

## Pseudocode

```plaintext
initialize centroids  
repeat  
    assign points to nearest centroid  
    recompute centroids  
until convergence
```

## Implementation (Python)

```python
def k_means(data, k):
    centroids = initialize_centroids(data, k)
    while not converged:
        clusters = assign_clusters(data, centroids)
        centroids = recalculate_centroids(clusters)
    return clusters
```

## Applications

- Market segmentation
- Document clustering
- Image segmentation
- Anomaly detection

## Strengths

- Simplicity and speed in processing large datasets.
- Easy to understand and implement.

## Limitations

- Sensitive to outliers and initial centroid placement.
- Assumes clusters are spherical and of similar size.

## References and Further Reading

- MacQueen, J. B. (1967). Some Methods for classification and Analysis of Multivariate Observations.
- "K-Means Clustering" on Scholarpedia.
