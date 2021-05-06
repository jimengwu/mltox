

import pandas as pd
from scipy.spatial.distance import pdist, squareform,cdist
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from time import ctime
from tqdm import tqdm
import argparse
from helper_model import *
import multiprocessing as mp
import argparse
import sys
import os

import pickle
def getArguments():
    parser = argparse.ArgumentParser(description='Running KNN_model for datasets.')
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-n", "--neighbors", dest="neighbors", required=True,nargs='+', type=int)
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", required=True)
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", required=True)
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()

args = getArguments()


if __name__ == '__main__':
    # def setArguments():
    #     parser = argparse.ArgumentParser(description='Select best parameter for simple RASAR.')
    #     parser.add_argument("-l", "--label", dest="label", required=True,
    #                         help="select which model")
    #     return parser.parse_args()

    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
                'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']
    # non_categorical was numerical features, whcih will be standarized. \
    # Mol,bonds_number, atom_number was previously log transformed due to the maginitude of their values.

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count', 
                'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP','water_solubility',
                'melting_point']
    print('load data...',ctime())

    # args = setArguments()
    X, Y = load_data(args.inputFile,'multiclass',categorical,non_categorical,encoding_value =[0.1,1,10,100], seed = 42)
    # X=X[:500]
    # Y=Y[:500]
    print('calcultaing distance matrix..',ctime())

    basic_mat, matrix_h, matrix_p = cal_matrixs(X,X,categorical,non_categorical)

    results = []
    splitter = KFold(n_splits=4, shuffle=True,random_state = 10)
    folds = list(splitter.split(X, Y))

    i = 1
    if args.hamming_alpha == 'logspace':
        sequence_ap = np.logspace(-4,5,20)
        sequence_ah = sequence_ap
    else:
        sequence_ap = [float(args.pubchem2d_alpha)]
        sequence_ah = [float(args.hamming_alpha)]   
    # sequence_ah = np.logspace(-4,5,20)
    # sequence_ap = np.logspace(-4,5,20)
    # sequence_ah = [0.2069138081114788]
    # sequence_ap = [1.8329807108324339]

    # print('same ah & ap')
    print('start running')
    for n in args.neighbors:
        a = {}
        avg_accs=0
        for ah in sequence_ah:
            for ap in sequence_ap:
                results = []
                with mp.Pool(128) as pool:
                    for num, fold in enumerate(folds):
                        print("*"*50, i, end='\r')
                        distance_matrix = matrix_combine(basic_mat,matrix_h,matrix_p,ah,ap)
                        dist_matr_train = distance_matrix.iloc[fold[0],fold[0]]
                        dist_matr_test = distance_matrix.iloc[fold[1],fold[0]]
                        y_train,y_test = Y[fold[0]], Y[fold[1]]
                        X_train,X_test =  X.iloc[fold[0]], X.iloc[fold[1]]


                        res = pool.apply_async(model_mul, args=(dist_matr_train,
                                                                dist_matr_test,
                                                                  X_train,
                                                                  y_train,
                                                                  y_test,
                                                                  n,
                                                                  ah,
                                                                  ap,
                                                                  num) )

                        results.append(res)
                        i = i+ 1
                    results = [res.get() for res in results]
                
                if np.mean([results[i]['accs'] for i in range(len(results))])> avg_accs:
                    a["accs"] = np.mean([results[i]['accs'] for i in range(len(results))])
                    a["f1"] = np.mean([results[i]['f1'] for i in range(len(results))])
                    a['se_accs'] = sem([results[i]['accs'] for i in range(len(results))])
                    a['se_f1'] = sem([results[i]['f1'] for i in range(len(results))])                    
                    a['neighbors'] = n
                    a['ah'] = ah
                    a['ap'] = ap
                    avg_accs = np.mean([results[i]['accs'] for i in range(len(results))])
                    print("success!")
        with open(args.outputFile+f'{n}.pkl', 'wb') as f:
            pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
        
        # with open(f'{n}.pkl', 'rb') as f:
        #     check =  pickle.load(f)
        # print(check)


# python RASAR_mulneigh_mul.py -i 'data/LC50/lc50_processed.csv' -n 1 2 3 4 5 6 7 8 9 10 -ah 0.2069138081114788 -ap 1.8329807108324339 -o results/multipleneighbor/woalpha/multiclass/KNN_mul_
# python RASAR_mulneigh_mul.py -i 'data/LC50/lc50_processed.csv' -n 1 2 3 4 5 6 7 8 9 10 -ah 0.2069138081114788 -ap 5.455594781168514 -o results/multipleneighbor/woalpha/multiclass/1NN_mul_
# python RASAR_mulneigh_mul.py -i 'data/LC50/lc50_processed.csv' -n 1 2 3 4 5 6 7 8 9 10 -ah 1 -ap 1 -o results/multipleneighbor/woalpha/ahap1/multiclass/mul_