import os
import numpy as np
import pandas as pd
from KNN import KNNClassifier 
import argparser
import utils
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main(args):
    train_filepath = os.path.join(args.data_path,'train_data.csv')
    test_filepath = os.path.join(args.data_path, 'test_data.csv')    
    
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    X_train, y_train, X_test, y_test = utils.load_data(train_df, test_df)

    knn = KNNClassifier(k=args.k_value, distance_metric=args.distance_metric, weights=args.weight, normalize=True, verbose=False)

    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"Precision: {precision_score(y_test, predictions)}")
    print(f"Recall: {recall_score(y_test, predictions)}")
    print(f"F1 Score: {f1_score(y_test, predictions)}")



if __name__ == "__main__":
    parser = argparser.get_parser()

    args = parser.parse_args()

    main(args)