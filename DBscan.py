import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import manhattan_distances

class DBscan():
    def __init__(self, eps, min_samples, similarity='hamming'):
        self.eps = eps
        self.min_samples = min_samples
        
        if similarity == 'manhattan':
            self.distance  = manhattan_distances
            #self.distance = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        elif similarity == 'hamming':
            self.distance  = hamming
            #self.distance = lambda x, y: sum(a != b for a, b in zip(x, y))
        

    def cluster(self, X, stop=False, dist_matrix=None):
        self.X = X
        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
        else:
            print('Calculating distance matrix...')
            self.calculate_matrix()

        if stop:
            return

        self.cluster_labels = np.full(self.X.shape[0], -1)    
        cluster_label = 0
        for i in range(self.X.shape[0]):
            if self.cluster_labels[i] == -1:
                self.cluster_labels[i] = -2
            neighbors = self.get_neighbors(i)
            if len(neighbors) >= self.min_samples:
                self.cluster_labels[i] = cluster_label
                cluster_label += 1
                self.cluster_labels = self.expand_cluster(i, neighbors, cluster_label)
        return self.cluster_labels 

    def calculate_matrix(self):
        self.dist_matrix = np.zeros((self.X.shape[0], self.X.shape[0]))
        for i in tqdm(range(self.X.shape[0])):
            for j in range(self.X.shape[0]):
                self.dist_matrix[i, j] = self.distance(self.X[i], self.X[j])

    def get_neighbors(self, i):
        neighbors = []
        for j in range(self.X.shape[0]):
            if i != j and self.dist_matrix[i, j] <= self.eps:
                neighbors.append(j)
        return neighbors

    def expand_cluster(self, i, neighbors, cluster_label):
        self.cluster_labels[i] = cluster_label
        for j in neighbors:
            if self.cluster_labels[j] == -1:
                self.cluster_labels[j] = cluster_label
                new_neighbors = self.get_neighbors(j)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors 
        return self.cluster_labels