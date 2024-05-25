import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd 

def load_data(opts):

    if opts.data_path == 'dataset/Arrhythmia_Data_Set':
        train_path = os.path.join(opts.data_path, 'train_data.csv')
        train_label = os.path.join(opts.data_path, 'train_label.csv')
        test_path = os.path.join(opts.data_path, 'test_data.csv') 
        test_label = os.path.join(opts.data_path, 'test_label.csv')  

        X_train = pd.read_csv(train_path, header=None)
        X_test = pd.read_csv(test_path, header=None)
        y_train = pd.read_csv(train_label, header=None)
        y_test = pd.read_csv(test_label, header=None)


    elif opts.data_path == 'dataset/gene_expression_cancer_RNA-Seq_Data_Set':
        train_path = os.path.join(opts.data_path, 'train_data.csv')
        train_label = os.path.join(opts.data_path, 'train_label.csv')
        test_path = os.path.join(opts.data_path, 'test_data.csv')
        test_label = os.path.join(opts.data_path, 'test_label.csv')

        X_train = pd.read_csv(train_path)
        X_test = pd.read_csv(test_path)
        y_train = pd.read_csv(train_label)
        y_test = pd.read_csv(test_label)

    return X_train, y_train, X_test, y_test



def normalize_data(X_train, X_test):

    # drop the column with the same value for all samples
    # X_train.drop(columns=X_train.std()[X_train.std() == 0].index, inplace=True)
    # X_test.drop(columns=X_train.std()[X_train.std() == 0].index, inplace=True)

    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    X_train = (X_train - X_train.mean()) / X_train.std().replace(0, 1e-5)
    X_test = (X_test - X_test.mean()) / X_test.std().replace(0, 1e-5)

    return X_train, X_test


