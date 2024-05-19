import os
import numpy as np
import pandas as pd
from clustering import Kmeans
import argparser
import preprocessing
import utils
import logging
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

def main(args):
    
    X_train, y_train, X_test, y_test = preprocessing.load_data(args)
    X_train_normalize, X_test_normalize = preprocessing.normalize_data(X_train, X_test)

    knn = KNeighborsClassifier(n_neighbors=args.k_value, weights=args.weight)

    knn.fit(X_train_normalize, y_train)
    
    # use a threshold to filter the confidece level, which is the probability of the class. Then use clustering to classify the data which is below the confidence level
    threshold = args.threshold

    probabilities = knn.predict_proba(X_test_normalize)

    low_confidence_indices = np.max(probabilities, axis=1) < threshold
    low_confidence_samples = X_test_normalize[low_confidence_indices]

    # clustering
    kmeans = Kmeans.KMEANS(n_clusters=args.n_clusters, max_iter=args.max_iter)

    kmeans.fit(low_confidence_samples)

    


    


if __name__ == "__main__":
    parser = argparser.get_parser()

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir,exist_ok=True)

    filemode = 'a'
    if args.k_value == args.k_start:  
        filemode = 'w'
    

    logging.basicConfig(
        filename=os.path.join(args.save_dir, f'experiment{args.data_path[-1]}_{args.distance_metric}_{args.weight}.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filemode=filemode
    )

    main(args)