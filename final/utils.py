import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd 

def assign_labels_to_cluster(opts, clusters, y_test, low_confidence_indices):
    
    actual_labels = y_test[low_confidence_indices].values.ravel()

    unknown_startnum = opts.train_num_classes + 1
    unknown_endnum = unknown_startnum + opts.unknown_num_classes - 1
    unknown_label_counts = {label: 0 for label in range(unknown_startnum, unknown_endnum + 1)}

    for label in actual_labels:
        if unknown_startnum <= label <= unknown_endnum:
            unknown_label_counts[label] += 1
    
    sorted_labels = sorted(unknown_label_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    cluster_labels={}

    for cluster, (label, count) in zip(sorted_clusters, sorted_labels):
        cluster_id = cluster[0]
        cluster_labels[label] = clusters[cluster_id]

    return cluster_labels

def merge_predictions(opts, X, knn_predictions, cluster_labels, low_confidence_indices):

    final_predictions = knn_predictions.copy()
    
    for index, data in X[low_confidence_indices].iterrows():
        data_array = data.values 
        for label, datapoints in cluster_labels.items():
            match_found = any((data_array == dp).all() for dp in datapoints)
            if match_found:
                final_predictions[index] = label
    
    return final_predictions
