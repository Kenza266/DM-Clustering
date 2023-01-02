import numpy as np

def dbscan(X, eps, min_samples):
    # X is a matrix of size n x m, where n is the number of samples and m is the number of features
    # eps is the maximum distance between two samples for them to be considered in the same neighborhood
    # min_samples is the minimum number of samples in a neighborhood for a sample to be considered a core sample
    
    # Initialize the cluster labels for each sample to -1 (meaning that the sample has not been assigned to a cluster yet)
    cluster_labels = np.full(X.shape[0], -1)
    
    # Initialize the cluster label to 0
    cluster_label = 0
    
    # Iterate over all samples
    for i in range(X.shape[0]):
        # If the sample has not been assigned to a cluster, it is a new noise sample
        if cluster_labels[i] == -1:
            cluster_labels[i] = -2
            continue
        
        # Find the neighboring samples of the current sample
        neighbors = get_neighbors(X, i, eps)
        
        # If the current sample is a core sample, start a new cluster
        if len(neighbors) >= min_samples:
            cluster_labels[i] = cluster_label
            cluster_label += 1
            
            # Expand the cluster to include all the neighboring samples that are reachable from the current core sample
            cluster_labels = expand_cluster(X, cluster_labels, i, neighbors, cluster_label, eps, min_samples)
    
    return cluster_labels

def get_neighbors(X, i, eps):
    # Find the neighboring samples of sample i
    neighbors = []
    for j in range(X.shape[0]):
        if i != j and np.linalg.norm(X[i] - X[j]) <= eps:
            neighbors.append(j)
    return neighbors

def expand_cluster(X, cluster_labels, i, neighbors, cluster_label, eps, min_samples):
    # Expand the cluster to include all the neighboring samples that are reachable from the current core sample
    cluster_labels[i] = cluster_label
    for j in neighbors:
        if cluster_labels[j] == -1:
            cluster_labels[j] = cluster_label
            new_neighbors = get_neighbors(X, j, eps)
            if len(new_neighbors) >= min_samples:
                neighbors += new_neighbors
    return cluster_labels
