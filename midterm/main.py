import os
import numpy as np
import pandas as pd
from KNN import KNN
import argparser
import utils
import logging
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main(args):
    train_filepath = os.path.join(args.data_path,'train_data.csv')
    test_filepath = os.path.join(args.data_path, 'test_data.csv')    
    
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    X_train, y_train, X_test, y_test = utils.load_data(train_df, test_df)

    trials = 1
    accuracy_log, precision_log, recall_log, f1_score_log = [],[],[],[]
    for trial in range(trials):
        knn = KNN.KNNClassifier(k=args.k_value, distance_metric=args.distance_metric, weights=args.weight, normalize=True, verbose=False)

        knn.fit(X_train, y_train)

        predictions = knn.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        accuracy_log.append(accuracy)
        precision_log.append(precision)
        recall_log.append(recall)
        f1_score_log.append(f1)

    accuracy = sum(accuracy_log) / len(accuracy_log)
    precision = sum(precision_log) / len(precision_log)
    recall = sum(recall_log) / len(recall_log)
    f1 = sum(f1_score_log) / len(f1_score_log)


    logging.info(f"Experiment{args.data_path[-1]}, k={args.k_value}, Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1_score={f1}")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    if args.k_value == args.k_end:
        eval_dict = utils.plot_results(args)
        logging.info(f"Average Accuracy: {eval_dict['avg_accuracy']}, Average Precision: {eval_dict['avg_precision']}, Average Recall: {eval_dict['avg_recall']}, Average F1: {eval_dict['avg_f1_score']}")
        logging.info(f"Max accuracy: {eval_dict['max_accuracy']}, Max precision: {eval_dict['max_precision']}, Max recall: {eval_dict['max_recall']}, Max f1: {eval_dict['max_f1_score']}, when k={eval_dict['k_value']}")


if __name__ == "__main__":
    parser = argparser.get_parser()

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir,exist_ok=True)

    filemode = 'a'
    if args.k_value == args.k_start:  # and args.data_path[-1] == 'A'
        filemode = 'w'
    

    logging.basicConfig(
        filename=os.path.join(args.save_dir, f'experiment{args.data_path[-1]}_{args.distance_metric}_{args.weight}.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filemode=filemode
    )

    main(args)