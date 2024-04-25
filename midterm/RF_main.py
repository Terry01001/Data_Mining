import os
import numpy as np
import pandas as pd
import argparser
import utils
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main(args):
    train_filepath = os.path.join(args.data_path,'train_data.csv')
    test_filepath = os.path.join(args.data_path, 'test_data.csv')    
    
    train_df = pd.read_csv(train_filepath)
    test_df = pd.read_csv(test_filepath)

    X_train, y_train, X_test, y_test = utils.load_data(train_df, test_df)

    Random_forest = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)

    Random_forest.fit(X_train, y_train)

    predictions = Random_forest.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    logging.info(f"Experiment{args.data_path[-1]}, n_estimators={args.n_estimators}, Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1_score={f1}")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    if args.n_estimators == args.n_end:
        utils.plot_RF_results(args)



if __name__ == "__main__":
    parser = argparser.get_parser()

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir,exist_ok=True)

    filemode = 'a'
    if args.n_estimators == args.n_start:  # and args.data_path[-1] == 'A'
        filemode = 'w'
    

    logging.basicConfig(
        filename=os.path.join(args.save_dir, 'RF_experiment.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filemode=filemode
    )

    main(args)