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

    log_file = os.path.join(args.save_dir, f'experiment{args.data_path[-1]}_{args.distance_metric}_{args.weight}.log')
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
    plt.legend(fontsize='large', title='Metrics', title_fontsize='large')
    plt.savefig(os.path.join(args.save_dir, f'experiment{args.data_path[-1]}_{args.distance_metric}_{args.weight}_performance_metrics.png'))
    # plt.show()
    plt.close()

    average_accuracy = sum(accuracies) / len(accuracies)
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    average_f1_score = sum(f1_scores) / len(f1_scores)

    max_accuracy_index = accuracies.index(max(accuracies))

    max_accuracy = accuracies[max_accuracy_index]
    max_precision = precisions[max_accuracy_index]
    max_recall = recalls[max_accuracy_index]
    max_f1_score = f1_scores[max_accuracy_index]
    
    return {
        "k_value": k_values[max_accuracy_index],
        "max_accuracy": max_accuracy,
        "max_precision": max_precision,
        "max_recall": max_recall,
        "max_f1_score": max_f1_score,
        "avg_accuracy": average_accuracy,
        "avg_precision": average_precision,
        "avg_recall": average_recall,
        "avg_f1_score": average_f1_score
    }

def plot_RF_results(args):

    log_file = os.path.join(args.save_dir, 'RF_experiment.log')
    n_estimators, accuracies, precisions, recalls, f1_scores = [], [], [], [], []

    with open(log_file, 'r') as file:
        for line in file:
            if "Experiment" in line:
                parts = line.split(',')
                n_estimators.append(int(parts[2].split('=')[1]))
                accuracies.append(float(parts[3].split('=')[1]))
                precisions.append(float(parts[4].split('=')[1]))
                recalls.append(float(parts[5].split('=')[1]))
                f1_scores.append(float(parts[6].split('=')[1]))

    plt.figure(figsize=(20, 10))
    plt.plot(n_estimators, accuracies, 'o-', label='Accuracy')
    plt.plot(n_estimators, precisions, 's-', label='Precision')
    plt.plot(n_estimators, recalls, '*-', label='Recall')
    plt.plot(n_estimators, f1_scores, 'x-', label='F1 Score')
    plt.xlabel('Value of n_estimators')
    plt.ylabel('Scores')
    plt.title('Performance Metrics Across Different n_estimators Values')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f'{args.model}_experiment{args.data_path[-1]}_performance_metrics.png'))
    # plt.show()
    plt.close()