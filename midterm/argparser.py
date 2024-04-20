import argparse

def get_parser():
    """
    Returns an ArgumentParser object with the default arguments for the KNN classifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_value', type=int,default=5, help='The value of k for the KNN classifier')
    parser.add_argument('--data_path', type=str, help='The path containing the  data')
    parser.add_argument('--distance_metric',type=str, default='euclidean', help='The distance metric to use for the KNN classifier')
    parser.add_argument('--weight', type=str, default='uniform', help='The weighting scheme to use for the KNN classifier')
    parser.add_argument('--save_dir', type=str, help='The path to save the output')
    parser.add_argument('--k_start', type=int, default=3, help='The starting value of k for the KNN classifier')
    parser.add_argument('--k_end', type=int, default=11, help='The ending value of k for the KNN classifier')

    return parser

