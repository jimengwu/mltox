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
    parser.add_argument("-label",
                        "--train_label",
                        dest='train_label',
                        required=True)
    parser.add_argument("-fixed", "--fixed_threshold", dest="fixed_threshold")
    parser.add_argument("-effect",
                        "--train_effect",
                        dest='train_effect',
                        required=True)
    parser.add_argument("-o",
                        "--output",
                        dest="outputFile",
                        default="binary.txt")
    return parser.parse_args()


args = getArguments()
X, Y = load_data(args.inputFile,
                 categorical_columns=categorical,
                 non_categorical_columns=non_categorical,
                 encoding='multiclass',
                 encoding_value=[0.1, 1, 10, 100],
                 seed=42)
print('calcultaing distance matrix..', ctime())
distance_matrix = dist_matrix(X, X, non_categorical, categorical, args.alpha_h,
                              args.alpha_p)

best_results = RASAR_simple(distance_matrix,
                            Y,
                            X,
                            "NO",
                            n_neighbors=2,
                            invitro="NO",
                            invitro_form="NO",
                            db_invitro="NO",
                            encoding="multiclass",
                            model="RF")

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

# python RASAR_simple_multi.py -i1 data/LOEC/loec_processed.csv  -label 'LOEC' -effect 'MOR' -ah 0.8858667904100823 -ap 6.158482110660261 -o results/rasar/loec_rasar_multi.txt -o2 results/rasar/loec_rasar_multi_result.csv
# python RASAR_simple_multi.py -i1 data/NOEC/noec_processed.csv  -label 'NOEC' -effect 'MOR' -ah 0.018329807108324356  -ap 0.12742749857031335 -o results/rasar/noec_rasar_multi.txt -o2 results/rasar/noec_rasar_multi_result.csv
# python RASAR_simple_multi.py -i1 data/LC50/lc50_processed.csv  -label ['LC50','EC50'] -effect 'MOR' -ah 0.12742749857031335 -ap 0.3359818286283781 -o results/rasar/lc50_rasar_multi.txt -o2 results/rasar/lc50_rasar_multi_result.csv
# python RASAR_simple_multi.py -i1 data/LC50/lc50_processed_rainbow.csv  -label ['LC50','EC50'] -effect 'MOR' -ah 0.3359818286283781 -ap 0.8858667904100823 -o results/rainbow/lc50_rasar_rainbow_multi.txt -o2 results/rainbow/lc50_rasar_rainbow_multi_result.csv
