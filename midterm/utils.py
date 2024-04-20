import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd 

def load_data(traindf, testdf):
    X_traindf = traindf.drop(columns=['Outcome']).values
    X_testdf = testdf.drop(columns=['Outcome']).values
    y_traindf = traindf['Outcome'].values
    y_testdf = testdf['Outcome'].values

    return X_traindf, y_traindf, X_testdf, y_testdf

def plot_confusion_matrix(y_true, y_pred):
    # confusion matrix
    c_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 7))  
    sns.heatmap(c_matrix, annot=True, cmap='Blues', fmt='d', linewidths=.5)
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # plt.show()

    plt.close()


def plot_results(args):

    log_file = os.path.join(args.save_dir, 'experiment.log')
    k_values, accuracies, precisions, recalls, f1_scores = [], [], [], [], []

    with open(log_file, 'r') as file:
        for line in file:
            if "Experiment" in line:
                parts = line.split(',')
                k_values.append(int(parts[2].split('=')[1]))
                accuracies.append(float(parts[3].split('=')[1]))
                precisions.append(float(parts[4].split('=')[1]))
                recalls.append(float(parts[5].split('=')[1]))
                f1_scores.append(float(parts[6].split('=')[1]))

    plt.figure(figsize=(20, 10))
    plt.plot(k_values, accuracies, 'o-', label='Accuracy')
    plt.plot(k_values, precisions, 's-', label='Precision')
    plt.plot(k_values, recalls, '*-', label='Recall')
    plt.plot(k_values, f1_scores, 'x-', label='F1 Score')
    plt.xlabel('Value of K')
    plt.ylabel('Scores')
    plt.title('Performance Metrics Across Different k Values')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f'performance_metrics_experiment{args.data_path[-1]}.png'))
    # plt.show()
    plt.close()