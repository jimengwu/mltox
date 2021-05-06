from helper_model import *
from scipy.spatial.distance import cdist, pdist, squareform
from collections import Counter
import h2o
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description='Running KNN_model for datasets.')
    parser.add_argument("-i1", "--input", dest="inputFile", required=True)
    parser.add_argument("-ah",
                        "--alpha_h",
                        dest="alpha_h",
                        required=True,
                        nargs='?',
                        type=float)
    parser.add_argument("-ap",
                        "--alpha_p",
                        dest="alpha_p",
                        required=True,
                        nargs='?',
                        type=float)
    parser.add_argument("-o",
                        "--output",
                        dest="outputFile",
                        default="binary.txt")
    return parser.parse_args()


args = getArguments()
X, Y = load_data(args.inputFile,
                 'binary',
                 categorical,
                 non_categorical,
                 encoding_value=1,
                 seed=42)

print('calcultaing distance matrix..', ctime())
distance_matrix = dist_matrix(X, X, non_categorical, categorical, args.alpha_h,
                              args.alpha_p)

best_results = RASAR_simple(distance_matrix,
                            Y,
                            X,
                            "noinvitro",
                            n_neighbors=1,
                            db_invitro="noinvitro",
                            model="LR")

info = []
info.append('''Accuracy: \t {}, se: {}
            RMSE: \t\t {}, se: {}
            Sensitivity: \t {}, se: {}
            Precision: \t {}, se: {}
            F1: \t {}, se:{}'''.format(
    best_results["avg_accs"], best_results["se_accs"],
    best_results["avg_rmse"], best_results["se_rmse"],
    best_results["avg_sens"], best_results["se_sens"],
    best_results["avg_precs"], best_results["se_precs"],
    best_results["avg_f1"], best_results["se_f1"]))

filename = args.outputFile
dirname = os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)

with open(filename, 'w') as file_handler:
    for item in info:
        file_handler.write("{}\n".format(item))

# python RASAR_simple.py -i1 data/LOEC/loec_processed.csv -ah 0.615848211066026 -ap 16.23776739188721 -o results/rasar/loec_rasar.txt -o2 results/rasar/loec_rasar_result.csv
# python RASAR_simple.py -i1 data/NOEC/noec_processed.csv -ah 0.06951927961775606  -ap 0.2069138081114788 -o results/rasar/noec_rasar.txt -o2 results/rasar/noec_rasar_result.csv
# python RASAR_simple.py -i1 data/LC50/lc50_processed.csv -ah 0.2069138081114788 -ap 0.615848211066026 -o results/rasar/lc50_rasar.txt -o2 results/rasar/lc50_rasar_result.csv
# python RASAR_simple.py -i1 data/LC50/lc50_processed_rainbow.csv -ah 5.455594781168514 -ap 143.8449888287663 -o results/rainbow/lc50_rasar_rainbow_binary.txt -o2 results/rainbow/lc50_rasar_rainbow_binary_result.csv
#
