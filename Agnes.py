import numpy as np
from tqdm import tqdm

class Agnes():
    def __init__(self, similarity='default', linkage='centroid'):
        if similarity == 'default':
            self.distance = lambda x, y: np.linalg.norm(x - y) 
        elif similarity == 'manhattan':
            self.distance = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        elif similarity == 'hamming':
            self.distance = lambda x, y: sum(a != b for a, b in zip(x, y))

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

    def cluster(self, X, stop=False, dist_matrix=None):
        self.X = X
        clusters = {i: [i] for i in range(self.X.shape[0])}
        distances = [] 

        if dist_matrix is not None:
                self.dist_matrix = dist_matrix
        else:
            print('Calculating distance matrix...')
            self.calculate_matrix()

        if stop:
            return

        nb_clusters = []

        while len(clusters) > 2:
            min_distance = float('inf') 
            min_clusters = [] 

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

            for min_cluster in min_clusters:
                i, j = min_cluster 

                clusters[i] += clusters[j]
                del clusters[j]                   

                self.dist_matrix[i] = {c: self.linkage([X[k] for k in clusters[i]], [X[k] for k in clusters[c]]) for c in self.dist_matrix.keys() if c != i and c != j}
                del self.dist_matrix[j]
                for _, v in self.dist_matrix.items():
                    if j in v.keys():
                        del v[j]

            distances.append(min_distance)
            nb_clusters.append(len(clusters))
        return clusters, distances, nb_clusters

    def calculate_matrix(self):
        self.dist_matrix = {} 
        for i in tqdm(range(self.X.shape[0])):
            l = {}
            for j in range(self.X.shape[0]):
                if i != j: 
                    l[j] = self.distance(self.X[i], self.X[j])
            self.dist_matrix[i] = l

    def single_linkage(self, c1, c2):
        return min(self.distance(x, y) for x in c1 for y in c2)

    def complete_linkage(self, c1, c2):
        return max([self.distance(x, y) for x in c1 for y in c2])

    def average_linkage(self, c1, c2):
        return np.mean([self.distance(x, y) for x in c1 for y in c2])

    def centroid_linkage(self, c1, c2):
        return self.distance(np.mean(c1, axis=0), np.mean(c2, axis=0))

    def ward_linkage(self, c1, c2):
        n1 = c1.shape[0]
        n2 = c2.shape[0]
        return (n1 * n2) / (n1 + n2) * self.distance(np.mean(c1, axis=0), np.mean(c2, axis=0))