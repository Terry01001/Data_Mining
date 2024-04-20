import numpy as np

class KNNClassifier:
    def __init__(self, k, distance_metric='euclidean', weights='uniform', normalize=False, verbose=False):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.normalize = normalize
        self.verbose = verbose
    
    def fit(self, X, y):
        if self.normalize:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            X = (X - self.mean) / self.std
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        else:  # Default is euclidean
            return self.euclidean_distance(x1, x2)
    
    def predict(self, X):
        if self.normalize:
            X = (X - self.mean) / self.std
        predictions = []
        for x in X:
            distances = [self.calculate_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            if self.weights == 'distance':
                # Weighted voting
                weights = 1 / np.array([distances[i] for i in k_indices])
                most_common = np.bincount(k_nearest_labels, weights=weights).argmax()
            else:
                # Uniform weights
                most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
            if self.verbose:
                print(f"Processed {x}, predicted {most_common}")
        return np.array(predictions)
