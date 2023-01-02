import numpy as np

def agnes(X, n_clusters):
    # X is a matrix of size n x m, where n is the number of samples and m is the number of features
    # n_clusters is the number of clusters to form
    
    # Initialize the distance matrix
    dist_matrix = np.zeros((X.shape[0], X.shape[0]))
    
    # Calculate the distance between all pairs of samples
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
    
    # Initialize the cluster labels for each sample to its own cluster
    cluster_labels = np.arange(X.shape[0])
    
    # Initialize the number of clusters formed to be the number of samples
    n_clusters_formed = X.shape[0]
    
    # Initialize the minimum distance to be infinity
    min_dist = np.inf
    
    # Iterate until the desired number of clusters is reached
    while n_clusters_formed > n_clusters:
        # Find the pair of clusters with the minimum distance between them
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i != j and dist_matrix[i, j] < min_dist:
                    min_dist = dist_matrix[i, j]
                    cluster_i = i
                    cluster_j = j
        
        # Merge the two clusters with the minimum distance
        cluster_labels[cluster_labels == cluster_labels[cluster_j]] = cluster_labels[cluster_i]
        
        # Update the distance matrix
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if cluster_labels[i] == cluster_labels[j]:
                    dist_matrix[i, j] = np.inf
                else:
                    dist_matrix[i, j] = min(dist_matrix[i, j], np.linalg.norm(X[i] - X[j]))
        
        # Decrement the number of clusters formed
        n_clusters_formed -= 1
        
        # Reset the minimum distance
        min_dist = np.inf
        
    return cluster_labels