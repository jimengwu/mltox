from helper_model import *
import h2o
from tqdm import tqdm
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description='Running KNN_model for datasets.')
    parser.add_argument("-i1", "--input", dest="inputFile", required=True)
    parser.add_argument("-idf", "--input2", dest="inputFile2", required=True)
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
    parser.add_argument("-effect",
                        "--train_effect",
                        dest='train_effect',
                        required=True)
    parser.add_argument("-fixed",
                        "--fixed_threshold",
                        dest="fixed_threshold",
                        default='no')
    parser.add_argument("-o",
                        "--output",
                        dest="outputFile",
                        default="binary.txt")
    return parser.parse_args()


args = getArguments()
db_mortality, db_datafusion = load_datafusion_datasets(
    args.inputFile,
    args.inputFile2,
    categorical_columns=categorical,
    non_categorical_columns=non_categorical,
    fixed=args.fixed_threshold,
    encoding='multiclass',
    encoding_value=[0.1, 1, 10, 100])

X = db_mortality.drop(columns='conc1_mean').copy()
y = db_mortality.conc1_mean.values

distance_matrix = dist_matrix(X, X, non_categorical, categorical, args.alpha_h,
                              args.alpha_p)

db_datafusion_matrix = dist_matrix(
    X,
    db_datafusion.drop(columns="conc1_mean").copy(), non_categorical,
    categorical, args.alpha_h, args.alpha_p)

del db_mortality

best_results = cv_datafusion_rasar_multiclass(db_datafusion_matrix,
                                              distance_matrix,
                                              "no",
                                              X,
                                              y,
                                              db_datafusion,
                                              "no",
                                              train_label=args.train_label,
                                              train_effect=args.train_effect,
                                              final_model=False,
                                              n_neighbors=2,
                                              invitro="False",
                                              invitro_form="no",
                                              encoding="multiclass")

info = []

info.append("""Accuracy: \t {}, se: {}
RMSE: \t\t {}, se: {}
Sensitivity: \t {}, se: {}
Precision: \t {}, se: {}
F1: \t {},se:{}
""".format(best_results["avg_accs"], best_results["se_accs"],
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

# python RASAR_multi.py -i1 data/LOEC/loec_processed.csv  -i2 data/LOEC/loec_processed_df_itself.csv -label 'LOEC' -effect 'MOR' -ah 0.8858667904100823 -ap 6.158482110660261 -o results/mortality/loec_df_itself_multi.txt -o2 results/mortality/loec_df_itself_multi_result.txt
# python RASAR_multi.py -i1 data/NOEC/noec_processed.csv  -i2 data/NOEC/noec_processed_df_itself.csv -label 'NOEC' -effect 'MOR' -ah 0.018329807108324356  -ap 0.12742749857031335 -o results/mortality/noec_df_itself_multi.txt -o2 results/mortality/noec_df_itself_multi_result.csv
# python RASAR_multi.py -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_itself.csv  -label ['LC50','EC50'] -effect 'MOR' -ah 0.12742749857031335 -ap 0.3359818286283781 -o results/mortality/lc50_df_itself_multi.txt -o2 results/mortality/lc50_df_itself_multi_result.csv
# python RASAR_multi.py -i1 data/LC50/lc50_processed_rainbow.csv  -i2 data/LC50/lc50_processed_df_rainbow.csv -label ['LC50','EC50'] -effect 'MOR' -ah 0.3359818286283781 -ap 0.8858667904100823 -o results/rainbow/lc50_df_rainbow_multiclass.txt

# python RASAR_multi.py -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_acc.csv  -label ['LC50','EC50'] -effect 'MOR' -ah 0.12742749857031335 -ap 0.3359818286283781 -o results/effect/lc50_df_acc_nomor_multi.txt -o2 results/effect/lc50_df_acc_nomor_multi_result.csv
# python RASAR_multi.py -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_beh.csv  -label ['LC50','EC50'] -effect 'MOR' -ah 0.12742749857031335 -ap 0.3359818286283781 -o results/effect/lc50_df_beh_nomor_multi.txt -o2 results/effect/lc50_df_beh_nomor_multi_result.csv
# python RASAR_multi.py -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_enz.csv  -label ['LC50','EC50'] -effect 'MOR' -ah 0.12742749857031335 -ap 0.3359818286283781 -o results/effect/lc50_df_enz_nomor_multi.txt -o2 results/effect/lc50_df_enz_nomor_multi_result.csv
# python RASAR_multi.py -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_gen.csv  -label ['LC50','EC50'] -effect 'MOR' -ah 0.12742749857031335 -ap 0.3359818286283781 -o results/effect/lc50_df_gen_multi.txt -o2 results/effect/lc50_df_gen_multi_result.csv
# python RASAR_multi.py -i1 data/LC50/lc50_processed.csv  -i2 data/LC50/lc50_processed_df_bcm.csv  -label ['LC50','EC50'] -effect 'MOR' -ah 0.12742749857031335 -ap 0.3359818286283781 -o results/effect/lc50_df_bcm_nomor_multi.txt -o2 results/effect/lc50_df_bcm_nomor_multi_result.csv
