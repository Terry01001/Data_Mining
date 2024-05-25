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
    X_train, X_test = preprocessing.normalize_data(X_train, X_test)
    y_train = y_train.values.ravel()

    trials = 50
    accuracies_log = []
    
    for trial in range(1,trials+1):

        knn = KNeighborsClassifier(n_neighbors=args.k_value, weights=args.weight)

        knn.fit(X_train, y_train)
        classification_predictions = knn.predict(X_test)
        
        # use a threshold to filter the confidece level, which is the probability of the class. Then use clustering to classify the data which is below the confidence level
        threshold = args.threshold

        probabilities = knn.predict_proba(X_test)

        low_confidence_indices = np.max(probabilities, axis=1) < threshold
        low_confidence_samples = X_test[low_confidence_indices]

        # clustering
        kmeans = Kmeans.KMEANS(n_clusters=args.n_clusters, max_iter=args.max_iter)

        kmeans.fit(low_confidence_samples)
        kmeans.plot_clusters(args)

        cluster_labels = utils.assign_labels_to_cluster(args, kmeans.clusters, y_test, low_confidence_indices)
        final_predictions = utils.merge_predictions(args, X_test, classification_predictions, cluster_labels, low_confidence_indices)

        # y_test = y_test.values.ravel()
        acc = accuracy_score(y_test, final_predictions)
        accuracies_log.append(acc)

        logging.info(f"Trial {trial}, Accuracy={acc}")

    avg_acc = np.mean(accuracies_log)
    std_dev = np.std(accuracies_log)

    logging.info(f"Average accuracy: {avg_acc}, std: {std_dev}")

    


    


if __name__ == "__main__":
    parser = argparser.get_parser()

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir,exist_ok=True)
    if not os.path.exists(args.save_fig_dir):
        os.makedirs(args.save_fig_dir,exist_ok=True)

    filemode = 'w'
    logfilename = args.data_path.split("/")[-1]
    
    

    logging.basicConfig(
        filename=os.path.join(args.save_dir, f'{logfilename}_{args.classify_algo}_{args.k_value}_{args.weight}.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filemode=filemode
    )

    main(args)