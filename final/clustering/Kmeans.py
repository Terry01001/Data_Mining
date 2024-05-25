# K-means++ ref: https://www.cnblogs.com/yixuan-xu/p/6272208.html
# t-sne ref: https://www.mropengate.com/2019/06/t-sne.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class KMEANS:
    def __init__(self, n_clusters, max_iter, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = []

    def initialize(self, k, data):
        centroids = []
        centroids.append(data[np.random.randint(data.shape[0])])

        # loop k-1 times
        for i in range (1,k):
            distances = []
            for dp in data:
                min_dist = np.min([np.linalg.norm(dp - c)**2 for c in centroids])
                distances.append(min_dist)
            distances = np.array(distances)
            prob = distances / np.sum(distances)
            cumulative_prob = prob.cumsum()
            rand_prob = np.random.rand()

            for j, p in enumerate(cumulative_prob):
                if rand_prob < p:
                    centroids.append(data[j])
                    break

        return centroids

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values

        self.centroids = self.initialize(self.n_clusters, data)

        for i in range(self.max_iter):
            self.clusters = {}
            for j in range(self.n_clusters):
                self.clusters[j] = []

            # distribute datapoint to the nearest cluster
            for dp in data:
                distances = [np.linalg.norm(dp - c)**2 for c in self.centroids] 
                cluster = np.argmin(distances)
                self.clusters[cluster].append(dp)

            prev_centroids = np.array(self.centroids)
            # update centroids
            for cluster in self.clusters:
                self.centroids[cluster] = np.mean(self.clusters[cluster], axis=0)
            
            optimal = True
            for centroid in range(self.n_clusters):
                diff = self.centroids[centroid] - prev_centroids[centroid]
                avoid_division_zero = np.where(prev_centroids[centroid] == 0, 1e-10, prev_centroids[centroid])
                change = np.sum(diff / avoid_division_zero)
                if abs(change) > self.tol:
                    optimal = False
                    break

            if optimal:
                break

    def predict(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values

        cluster = []
        for dp in data:
            distances = [np.linalg.norm(dp - c)**2 for c in self.centroids]
            classification = np.argmin(distances)
            cluster.append(classification)

        return cluster

    def plot_clusters(self, opts):
        all_data = []
        labels = []

        for cluster_id, data in self.clusters.items():
            for datapoint in data:
                all_data.append(datapoint)
                labels.append(cluster_id)
        
        all_data = np.array(all_data)

        # use tsne to reduce dimensionality and visualize
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data)-1))
        transformed_data = tsne.fit_transform(all_data)

        plt.figure(figsize=(15, 10))
        scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='inferno',alpha=0.6) # viridis
        # plt.colorbar(scatter)
        plt.title('Clusters visualized with t-SNE')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        # save fig
        datasetname = opts.data_path.split("/")[-1]
        plt.savefig(f'{opts.save_fig_dir}/{datasetname}_clusters_{opts.classify_algo}_{opts.k_value}_{opts.weight}.png')

