from helper_model import *
import argparse
import sys
import h2o
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description='Running KNN_model for datasets.')
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-idf",
                        "--inputdatafusion",
                        dest="input_datafusion_File",
                        required=True)
    parser.add_argument("-n", "--neighbors", dest='neighbors', required=True)
    parser.add_argument("-invitro",
                        "--invitro",
                        dest="wo_invitro",
                        required=True,
                        nargs='+')
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-ah",
                        "--alpha_h",
                        dest="hamming_alpha",
                        required=True)
    parser.add_argument("-ap",
                        "--alpha_p",
                        dest="pubchem2d_alpha",
                        required=True)
    parser.add_argument("-invitro_form",
                        "--invitro_form",
                        dest="invitro_form",
                        required=True)
    parser.add_argument("-o",
                        "--output",
                        dest="outputFile",
                        default="binary.txt")
    return parser.parse_args()


args = getArguments()

non_categorical = [
    'ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
    'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP',
    'water_solubility', 'melting_point'
]
categorical = [
    'class', 'tax_order', 'family', 'genus', "species", 'control_type',
    'media_type', 'application_freq_unit', "exposure_type", "conc1_type",
    'obs_duration_mean'
]
# chem_columns = ['test_cas','pubchem2d','smiles','organism','tissue']
conc = ['conc1_mean']
drop_columns = ['Unnamed: 0']

if args.encoding == "binary":
    db_mortality, db_datafusion = load_datafusion_datasets(
        args.inputFile,
        args.input_datafusion_File,
        categorical,
        non_categorical,
        drop_columns,
        encoding='binary',
        encoding_value=1,
        seed=42)

    X = db_mortality.drop(columns='conc1_mean').copy()
    X['invitro_label'] = np.where(X["ec50"].values > 0.6, 0, 1)
    Y = db_mortality.conc1_mean.values
elif args.encoding == "multiclass":
    db_mortality, db_datafusion = load_datafusion_datasets(
        args.inputFile,
        args.input_datafusion_File,
        categorical,
        non_categorical,
        drop_columns,
        encoding='multiclass',
        encoding_value=[0.1, 1, 10, 100],
        seed=42)
    X = db_mortality.drop(columns='conc1_mean').copy()
    X['invitro_label'] = multiclass_encoding(X['ec50'], [0.006, 0.3, 63, 398])
    Y = db_mortality.conc1_mean.values

print("calcultaing distance matrix..", ctime())
basic_mat, matrix_h, matrix_p = cal_matrixs(X, X, categorical, non_categorical)
basic_mat_x_df, matrix_h_x_df, matrix_p_x_df = cal_matrixs(
    X,
    db_datafusion.drop(columns="conc1_mean").copy(), categorical,
    non_categorical)

print("successfully calculated distance matrix..", ctime())

if args.hamming_alpha == 'logspace':
    sequence_ap = np.logspace(-4, 5, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [args.pubchem2d_alpha]
    sequence_ah = [args.hamming_alpha]
i = 0
db_invitro_matrix = "No"
db_invitro = "overlap"
for invitro in (args.wo_invitro):
    best_accs = 0
    for ah in (sequence_ah):
        for ap in sequence_ap:
            print(
                "*" * 50,
                round(
                    i / (len(args.neighbors) * len(sequence_ap) *
                         len(sequence_ah) * len(args.wo_invitro)), 5), ctime())
            distance_matrix = matrix_combine(basic_mat, matrix_h, matrix_p,
                                             float(ah), float(ap))
            db_datafusion_matrix = matrix_combine(basic_mat_x_df,
                                                  matrix_h_x_df, matrix_p_x_df,
                                                  float(ah), float(ap))
            if args.encoding == "multiclass":
                results = cv_datafusion_rasar_multiclass(
                    db_datafusion_matrix,
                    distance_matrix,
                    db_invitro_matrix,
                    X,
                    Y,
                    db_datafusion,
                    n_neighbors=int(args.neighbors),
                    train_label=["LC50", "EC50"],
                    train_effect="MOR",
                    db_invitro=db_invitro,
                    final_model=False,
                    invitro=invitro,
                    invitro_form=args.invitro_form,
                    encoding=args.encoding)
            elif args.encoding == "binary":
                results = cv_datafusion_rasar(db_datafusion_matrix,
                                              distance_matrix,
                                              db_invitro_matrix,
                                              X,
                                              Y,
                                              db_datafusion,
                                              db_invitro,
                                              n_neighbors=int(args.neighbors),
                                              train_label=["LC50", "EC50"],
                                              train_effect="MOR",
                                              invitro=invitro,
                                              invitro_form=args.invitro_form,
                                              encoding=args.encoding)

            if results["avg_accs"] > best_accs:
                best_results = results
                best_accs = results["avg_accs"]
                best_alpha_h = ah
                best_alpha_p = ap

            i = i + 1
    info = []
    info.append("""The best params were alpha_h:{}, alpha_p:{}""".format(
        best_alpha_h, best_alpha_p))
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
    info.append(best_results["fold"])
    info.append(best_results["model"])
    filename = args.outputFile + str(invitro) + "_" + str(
        args.neighbors) + "_" + str(args.invitro_form) + '.txt'
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, 'w') as file_handler:
        for item in info:
            file_handler.write("{}\n".format(item))

# python RASAR_datafusion_model_addinginvitro.py  -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv -invitro "True" "False" "own" -invitro_form "both" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasardf/binary/n2/01/new_invivo_eawag_repeated_5nn_"
# python RASAR_datafusion_model_addinginvitro.py  -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv -invitro "True" "False"  -invitro_form "both" -n 2 -ah 0.2069138081114788 -ap 0.615848211066026 -o "results/invivotro/rasardf/binary/n2/01/new_invivo_eawag_repeated_LC50_5nn_"
# python RASAR_datafusion_model_addinginvitro.py  -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv -invitro "True" "own" -invitro_form "label" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasardf/binary/n2/01/new_invivo_eawag_repeated_5nn_"
# python RASAR_datafusion_model_addinginvitro.py  -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv -invitro "True"  -invitro_form "label" -n 2 -ah 0.2069138081114788 -ap 0.615848211066026 -o "results/invivotro/rasardf/binary/n2/01/new_invivo_eawag_repeated_LC50_5nn_"
# python RASAR_datafusion_model_addinginvitro.py  -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv -invitro "True" "own" -invitro_form "number" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasardf/binary/n2/01/new_invivo_eawag_repeated_5nn_"
# python RASAR_datafusion_model_addinginvitro.py  -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv -invitro "True"  -invitro_form "number" -n 2 -ah 0.2069138081114788 -ap 0.615848211066026 -o "results/invivotro/rasardf/binary/n2/01/new_invivo_eawag_repeated_LC50_5nn_"
# python RASAR_datafusion_model_addinginvitro.py  -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv -invitro "True" "False"  -invitro_form "both" -n 2 -ah "logspace" -ap "logspace" -o "results/invivotro/rasardf/binary/n2/01/new_invivo_eawag_repeated_rasar_"
# python RASAR_datafusion_model_addinginvitro.py  -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv -invitro "True"  -invitro_form "label" -n 2 -ah "logspace" -ap "logspace" -o "results/invivotro/rasardf/binary/n2/01/new_invivo_eawag_repeated_rasar_"
# python RASAR_datafusion_model_addinginvitro.py  -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv -invitro "True"  -invitro_form "number" -n 2 -ah "logspace" -ap "logspace" -o "results/invivotro/rasardf/binary/n2/01/new_invivo_eawag_repeated_rasar_"

# python RASAR_datafusion_model_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv  -e "multiclass" -invitro "True" "own" "False" -invitro_form "both" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasardf/multiclass/invivo_eawag_repeated_5nn_"
# python RASAR_datafusion_model_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv  -e "multiclass" -invitro "True" "own" "False" -invitro_form "both" -n 2 -ah 143.8449888287663 -ap 143.8449888287663 -o "results/invivotro/rasardf/multiclass/invivo_eawag_repeated_1nn_"
# python RASAR_datafusion_model_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv  -e "multiclass" -invitro "True" "own"  -invitro_form "label" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasardf/multiclass/invivo_eawag_repeated_5nn_"
# python RASAR_datafusion_model_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv  -e "multiclass" -invitro "True" "own"  -invitro_form "label" -n 2 -ah 143.8449888287663 -ap 143.8449888287663 -o "results/invivotro/rasardf/multiclass/invivo_eawag_repeated_1nn_"
# python RASAR_datafusion_model_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv  -e "multiclass" -invitro "True" "own"  -invitro_form "number" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasardf/multiclass/invivo_eawag_repeated_5nn_"
# python RASAR_datafusion_model_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv  -e "multiclass" -invitro "True" "own"  -invitro_form "number" -n 2 -ah 143.8449888287663 -ap 143.8449888287663 -o "results/invivotro/rasardf/multiclass/invivo_eawag_repeated_1nn_"
# python RASAR_datafusion_model_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv  -e "multiclass" -invitro "True"  -invitro_form "number" -n 2 -ah "logspace" -ap "logspace" -o "results/invivotro/rasardf/multiclass/invivo_eawag_repeated_rasar_"
# python RASAR_datafusion_model_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv  -e "multiclass" -invitro "True"  -invitro_form "label" -n 2 -ah "logspace" -ap "logspace" -o "results/invivotro/rasardf/multiclass/invivo_eawag_repeated_rasar_"
# python RASAR_datafusion_model_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv -idf data/LC50/lc50_processed_df_itself.csv  -e "multiclass" -invitro "True" "False" -invitro_form "both" -n 2 -ah "logspace" -ap "logspace" -o "results/invivotro/rasardf/multiclass/invivo_eawag_repeated_rasar_"
