from helper_model import recall_score, precision_score, accuracy_score, mean_squared_error, f1_score, categorical, non_categorical, train_test_split, KNeighborsClassifier, sqrt, load_data, select_alpha, dist_matrix, MinMaxScaler, confusion_matrix
import numpy as np
from time import ctime
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description='Running KNN_model for datasets.')
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-l",
                        "--leaf_ls",
                        dest="leaf_list",
                        required=True,
                        nargs='+',
                        type=int)
    parser.add_argument("-n",
                        "--neighbors",
                        dest="neighbors",
                        required=True,
                        nargs='+',
                        type=int)
    parser.add_argument("-o",
                        "--output",
                        dest="outputFile",
                        default="binary.txt")
    return parser.parse_args()


args = getArguments()

# loading data & splitting into train and test dataset
X, Y = load_data(args.inputFile,
                 encoding='binary',
                 categorical_columns=categorical,
                 non_categorical_columns=non_categorical,
                 encoding_value=1,
                 seed=42)

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=42)

# using 5-fold cross validation to choose the alphas with best accuracy
sequence_alpha = np.logspace(-4, 5, 20)
print(ctime())
best_alpha_h, best_alpha_p, best_leaf, best_neighbor, best_results = select_alpha(
    X_train, sequence_alpha, Y_train, categorical, non_categorical,
    args.leaf_list, args.neighbors)
print(ctime())

# validate on the test dataset
minmax = MinMaxScaler().fit(X_train[non_categorical])
X_train[:][non_categorical] = minmax.transform(X_train.loc[:, non_categorical])
X_test[:][non_categorical] = minmax.transform(X_test.loc[:, non_categorical])

matrix = dist_matrix(X_test, X_train, non_categorical, categorical,
                     best_alpha_h, best_alpha_p)
matrix_train = dist_matrix(X_train, X_train, non_categorical, categorical,
                           best_alpha_h, best_alpha_p)
neigh = KNeighborsClassifier(n_neighbors=best_neighbor,
                             metric='precomputed',
                             leaf_size=best_leaf)
neigh.fit(matrix_train, Y_train.astype('int').ravel())
y_pred = neigh.predict(matrix)

accs = accuracy_score(Y_test, y_pred)
sens = precision_score(Y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
specs = tn / (tn + fp)
precs = sqrt(mean_squared_error(Y_test, y_pred))
f1 = f1_score(Y_test, y_pred)

print("Accuracy: ", accs, 'se:', best_results["St.err Acc"].values[0],
      "\n Sensitivity:", sens, 'se:', best_results["St.err Sens"].values[0],
      "\n Specificity", specs, 'se:', best_results["St.err Spec"].values[0],
      "\n Precision", precs, 'se:', best_results["St.err Prec"].values[0],
      "\n F1 score:", f1, 'se:', best_results["St.err F1"].values[0])

# saving the information into a file
info = []
info.append(
    '''The best params were alpha_h:{}, alpha_p:{} ,leaf:{},neighbor:{}'''.
    format(best_alpha_h, best_alpha_p, best_leaf, best_neighbor))
info.append('''Accuracy:  {}, Se.Accuracy:  {} 
		\nSensitivity:  {}, Se.Sensitivity: {}
        \nSpecificity:  {}, Se.Specificity:{}
		\nPrecision:  {}, Se.Precision: {}
		\nf1_score:{}, Se.f1_score:{}'''.format(
    accs, best_results["St.err Acc"].values[0], sens,
    best_results["St.err Sens"].values[0], specs,
    best_results["St.err Spec"].values[0], precs,
    best_results["St.err Prec"].values[0], f1,
    best_results["St.err F1"].values[0]))

info.append(
    'The parameters was selected from {}'.format('np.logspace(-4, 5, 20)'))
filename = args.outputFile
dirname = os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)
with open(filename, 'w') as file_handler:
    for item in info:
        file_handler.write("{}\n".format(item))

# python KNN.py -i data/NOEC/noec_processed.csv  -l 10 20 30 40 50 60 70 80 90 -n 1 3 5 -o  results/noec_binary.txt
# python KNN.py -i data/LOEC/loec_processed.csv  -l 10 20 30 40 50 60 70 80 90 -n 1 3 5 -o  results/knn/loec_binary.txt

# python KNN_multiclass.py -i data/NOEC/noec_processed.csv  -l 10 20 30 40 50 60 70 80 90 -n 1 3 5  -o  results/noec_multiclass.txt
# python KNN_multiclass.py -i data/LOEC/loec_processed.csv  -l 10 20 30 40 50 60 70 80 90 -n 1 3 5  -o  results/loec_multiclass.txt

# python KNN.py -i data/LC50/lc50_processed.csv  -l 10 30 50 70 90 -n  5 -o  results/lc50_binary.txt
# python KNN_multiclass.py -i data/LC50/lc50_processed.csv  -l 10 30 50 70 90 -n 5  -o  results/lc50_multiclass.txt
#

# python KNN.py -i data/LC50/lc50_processed_rainbow.csv  -l 10 30 50 70 90 -n  5 -o  results/rainbow/lc50_binary_rainbow.txt
# python KNN_multiclass.py -i data/LC50/lc50_processed_rainbow.csv  -l 10 30 50 70 90 -n 5  -o  results/rainbow/lc50_multiclass_rainbow.txt

# python KNN.py -i data/invitro/invivo_norepeated.csv  -l 10 30 50 70 90 -n  5 -o  results/invitro/invivo_norepeated.txt
# python KNN_multiclass.py -i data/invitro/invivo_norepeated.csv  -l 10 30 50 70 90 -n 5  -o  results/invitro/invivo_norepeated_multi.txt

# python KNN.py -i lc_db_processed.csv  -l 10 20 30 40 50 60 70 80 90 -n 1 -o  knn/1nn_bi.txt
# python KNN.py -i lc_db_processed.csv  -l 10 20 30 40 50 60 70 80 90 -n 3 -o  knn/3nn_bi.txt
# python KNN.py -i lc_db_processed.csv  -l 10 20 30 40 50 60 70 80 90 -n 5 -o  knn/5nn_bi.txt
