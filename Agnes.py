import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import manhattan_distances

class Agnes():
    def __init__(self, similarity='hamming', linkage='centroid'):
        if similarity == 'manhattan':
            self.distance = manhattan_distances
            #self.distance = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        elif similarity == 'hamming':
            self.distance = hamming
            #self.distance = lambda x, y: sum(a != b for a, b in zip(x, y))

        if linkage == 'centroid':
            self.linkage = self.centroid_linkage
        elif linkage == 'single':
            self.linkage = self.single_linkage
        elif linkage == 'complete':
            self.linkage = self.complete_linkage
        elif linkage == 'average':
            self.linkage = self.average_linkage
        elif linkage == 'ward':
            self.linkage = self.ward_linkage

    def cluster_slow(self, X, stop=False, dist_matrix=None):
        self.X = X
        clusters = {i: [i] for i in range(self.X.shape[0])}
        distances = [] 
        nb_clusters = []

        if dist_matrix is not None:
                self.dist_matrix = dist_matrix
        else:
            print('Calculating distance matrix...')
            self.calculate_matrix_slow()

        if stop:
            return

        while len(clusters) > 2:
            min_distance, min_clusters = self.get_min_clusters_slow()
            for min_cluster in min_clusters:
                i, j = min_cluster 
                self.merge_slow(clusters, i, j)
            distances.append(min_distance)
            nb_clusters.append(len(clusters))

        return clusters, distances, nb_clusters

    def calculate_matrix_slow(self):
        self.dist_matrix = {} 
        for i in tqdm(range(self.X.shape[0])):
            l = {}
            for j in range(self.X.shape[0]):
                if i != j: 
                    l[j] = self.distance(self.X[i], self.X[j])
            self.dist_matrix[i] = l

    def merge_slow(self, clusters, i, j):
        clusters[i] += clusters[j]
        del clusters[j]                   
        self.dist_matrix[i] = {c: self.linkage([self.X[k] for k in clusters[i]], [self.X[k] for k in clusters[c]]) for c in self.dist_matrix.keys() if c != i and c != j}
        del self.dist_matrix[j]
        for _, v in self.dist_matrix.items():
            if j in v.keys():
                del v[j]

    def get_min_clusters_slow(self):
        min_clusters = [] 
        min_distance = float('inf') 
        for i in self.dist_matrix.keys():
            l = [x for x in self.dist_matrix[i].keys() if x > i]
            for j in l:
                d = self.dist_matrix[i][j] 
                if d == min_distance:
                    if i not in ([x[0] for x in min_clusters] + [x[1] for x in min_clusters]) and j not in ([x[0] for x in min_clusters] + [x[1] for x in min_clusters]):
                        min_clusters.append((i, j))
                if d < min_distance:
                    min_distance = d
                    min_clusters = [(i, j)] 
        return min_distance, min_clusters 

    def cluster(self, X, stop=False, dist_matrix=None):
        self.X = X
        clusters = [[X[i]] for i in range(self.X.shape[0])] 
        cluster_id = [[i] for i in range(self.X.shape[0])] 
        distances = [] 
        nb_clusters = []

        if dist_matrix is not None:
                self.dist_matrix = dist_matrix
        else:
            print('Calculating distance matrix...')
            self.calculate_matrix()
        if stop:
            return

        self.init_matrix = np.copy(self.dist_matrix)

        while len(clusters) > 2:
            min_distance, min_cluster = self.get_min_cluster() 
            i, j = min_cluster
            self.merge(cluster_id, clusters, i, j)
            distances.append(min_distance)
            nb_clusters.append(len(clusters))
            
        return cluster_id, clusters, distances, nb_clusters    

    def calculate_matrix(self):
        self.dist_matrix = np.full((self.X.shape[0], self.X.shape[0]), float('inf'))
        for i in tqdm(range(self.X.shape[0])):
            for j in range(i+1, self.X.shape[0]):
                self.dist_matrix[i, j] = self.distance(self.X[i].reshape(1, -1), self.X[j].reshape(1, -1))

    def merge(self, cluster_id, clusters, i, j):
        clusters[i] += clusters[j]
        cluster_id[i] += cluster_id[j]
        del clusters[j] 

        self.dist_matrix = np.delete(self.dist_matrix, j, axis=0)
        self.dist_matrix = np.delete(self.dist_matrix, j, axis=1)

        if self.linkage in [self.complete_linkage, self.single_linkage, self.average_linkage]:
            for k in range(0, i):
                self.dist_matrix[k, i] = self.linkage(cluster_id[i], cluster_id[k]) 

            for k in range(i+1, self.dist_matrix.shape[0]-1):
                self.dist_matrix[i, k] = self.linkage(cluster_id[i], cluster_id[k]) 
        else:
            for k in range(0, i):
                self.dist_matrix[k, i] = self.linkage(clusters[i], clusters[k]) 

            for k in range(i+1, self.dist_matrix.shape[0]-1):
                self.dist_matrix[i, k] = self.linkage(clusters[i], clusters[k]) 
            
    def get_min_cluster(self):
        min_distance = np.amin(self.dist_matrix)
        min_index = np.argmin(self.dist_matrix)
        min_coordinates = np.unravel_index(min_index, self.dist_matrix.shape)
        return min_distance, min_coordinates

    def single_linkage(self, c1, c2):
        return np.min(self.init_matrix[c1, :][:, c2])

    def complete_linkage(self, c1, c2):
        return np.max(self.init_matrix[c1, :][:, c2])

    def average_linkage(self, c1, c2):
        return np.mean(self.init_matrix[c1, :][:, c2]) 

    def centroid_linkage(self, c1, c2):
        return self.distance(np.mean(c1, axis=0).reshape(1, -1), np.mean(c2, axis=0).reshape(1, -1))

    def ward_linkage(self, c1, c2):
        n1 = np.array(c1).shape[0]
        n2 = np.array(c2).shape[0]
        return (n1 * n2) / (n1 + n2) * self.distance(np.mean(c1, axis=0).reshape(1, -1), np.mean(c2, axis=0).reshape(1, -1))