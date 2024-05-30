import argparse

def get_parser():
    """
    Returns an ArgumentParser object with the default arguments for the KNN classifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_value', type=int,default=5, help='The value of k for the KNN classifier')
    parser.add_argument('--data_path', type=str, help='The path containing the  data')
    parser.add_argument('--save_dir', type=str, help='The path to save the output')


    parser.add_argument('--n_clusters', type=int, help='The number of clusters to create')
    parser.add_argument('--max_iter', type=int, help='The maximum number of iterations')
    parser.add_argument('--tol', type=float, help='The tolerance for the stopping criterion')
    parser.add_argument('--threshold',type=float, help='The threshold for classify unknown type')

    parser.add_argument('--train_num_classes', type=int, help='The number of classes in the training dataset')
    parser.add_argument('--unknown_num_classes', type=int, help='The unknown number of classes in the test dataset')

    parser.add_argument('--save_fig_dir', type=str, help='The path to save the figures')
    parser.add_argument('--classify_algo', type=str, help='The algorithm to use for classification')

    # SVM
    parser.add_argument('--kernel', type=str, help='The kernel to use for the SVM')

    # not used in this project
    parser.add_argument('--distance_metric',type=str, default='euclidean', help='The distance metric to use for the KNN classifier')
    parser.add_argument('--weight', type=str, default='uniform', help='The weighting scheme to use for the KNN classifier')
    parser.add_argument('--k_start', type=int, default=3, help='The starting value of k for the KNN classifier')
    parser.add_argument('--k_end', type=int, default=11, help='The ending value of k for the KNN classifier')

    # not used in this project
    parser.add_argument('--n_estimators', type=int, default=10, help='The number of tree of random forest')
    parser.add_argument('--n_start', type=int, default=10, help='The starting value of n for the random forest')
    parser.add_argument('--n_end', type=int, default=100, help='The ending value of n for the random forest')
    parser.add_argument('--model', type=str, default='RF', help='The model to use for the random forest')

    return parser

