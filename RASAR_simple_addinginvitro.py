from helper_model import *
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(
        description='Running KNN_model for datasets.')
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-n", "--neighbors", dest='neighbors', required=True)
    parser.add_argument("-ah",
                        "--alpha_h",
                        dest="hamming_alpha",
                        required=True)
    parser.add_argument("-ap",
                        "--alpha_p",
                        dest="pubchem2d_alpha",
                        required=True)
    parser.add_argument("-new", "--new", dest="new", default=False)
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-invitro",
                        "--invitro",
                        dest="invitro",
                        required=True,
                        nargs='+')
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

conc_column = 'conc1_mean'
if args.encoding == "binary":
    X, Y = load_data(args.inputFile,
                     'binary',
                     categorical,
                     non_categorical,
                     encoding_value=1,
                     seed=42)
    X['invitro_label'] = np.where(X["ec50"].values > 1, 0, 1)
elif args.encoding == "multiclass":
    X, Y = load_data(args.inputFile,
                     'multiclass',
                     categorical,
                     non_categorical,
                     encoding_value=[0.1, 1, 10, 100],
                     seed=42)
    X['invitro_label'] = multiclass_encoding(X['ec50'], [0.006, 0.3, 63, 398])

print('calcultaing distance matrix..', ctime())

basic_mat, matrix_h, matrix_p = cal_matrixs(X, X, categorical, non_categorical)

if args.hamming_alpha == 'logspace':
    sequence_ap = np.logspace(-4, 5, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [args.pubchem2d_alpha]
    sequence_ah = [args.hamming_alpha]
db_invitro_matrix = "No"
db_invitro = "overlap"
i = 1
for invitro in args.invitro:
    for n in args.neighbors:
        best_accs = 0
        for ah in (sequence_ah):
            for ap in sequence_ap:
                print("*" * 50,
                      round(
                          i / (len(sequence_ap) * len(sequence_ah) *
                               len(args.invitro)), 3),
                      ctime(),
                      end='\r')
                distance_matrix = matrix_combine(basic_mat, matrix_h, matrix_p,
                                                 float(ah), float(ap))

                results = RASAR_simple(distance_matrix,
                                       Y,
                                       X,
                                       db_invitro_matrix,
                                       n_neighbors=int(n),
                                       invitro=invitro,
                                       db_invitro=db_invitro,
                                       invitro_form=args.invitro_form,
                                       encoding=args.encoding)
                i = i + 1
                print(ah, ap, results["avg_accs"])
                if results["avg_accs"] > best_accs:
                    print(ah, ap, results["avg_accs"])
                    best_results = results
                    best_accs = results["avg_accs"]
                    best_alpha_h = ah
                    best_alpha_p = ap

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
        info.append(best_results["model"])

        import os
        filename = args.outputFile + str(invitro) + "_" + str(n) + "_" + str(
            args.invitro_form) + '.txt'
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(filename, 'w') as file_handler:
            for item in info:
                file_handler.write("{}\n".format(item))

# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv     -invitro "True" "own" -invitro_form "number" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasar/binary/n2/rf/new_invivo_eawag_repeated_5nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv     -invitro "True" -invitro_form "number" -n 2 -ah 0.2069138081114788 -ap 0.615848211066026 -o "results/invivotro/rasar/binary/n2/rf/new_invivo_eawag_repeated_LC50_5nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv     -invitro "True" "own" -invitro_form "label" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasar/binary/n2/rf/new_invivo_eawag_repeated_5nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv     -invitro "True" -invitro_form "label" -n 2 -ah 0.2069138081114788 -ap 0.615848211066026 -o "results/invivotro/rasar/binary/n2/rf/new_invivo_eawag_repeated_LC50_5nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv     -invitro "True" "False" "own" -invitro_form "both" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasar/binary/n2/rf/new_invivo_eawag_repeated_5nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv     -invitro "True" "False" -invitro_form "both" -n 2 -ah 0.2069138081114788 -ap 0.615848211066026 -o "results/invivotro/rasar/binary/n2/rf/new_invivo_eawag_repeated_LC50_5nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv     -invitro "True" "False"  -invitro_form "both" -n 2 -ah 'logspace' -ap 'logspace' -o "results/invivotro/rasar/binary/n2/rf/new_invivo_eawag_repeated_rasar_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv     -invitro "True" -invitro_form "label" -n 2 -ah 'logspace' -ap 'logspace' -o "results/invivotro/rasar/binary/n2/rf/new_invivo_eawag_repeated_rasar_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv     -invitro "True" -invitro_form "number" -n 2 -ah 'logspace' -ap 'logspace' -o "results/invivotro/rasar/binary/n2/rf/new_invivo_eawag_repeated_rasar_"

# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv  -e "multiclass" -invitro "True" "own" "False" -invitro_form "both" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasar/multiclass/new_invivo_eawag_repeated_5nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv  -e "multiclass" -invitro "True" "own" "False" -invitro_form "both" -n 2 -ah 143.8449888287663 -ap 143.8449888287663 -o "results/invivotro/rasar/multiclass/new_invivo_eawag_repeated_1nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv  -e "multiclass" -invitro "True" "own"  -invitro_form "label" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasar/multiclass/new_invivo_eawag_repeated_5nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv  -e "multiclass" -invitro "True" "own"  -invitro_form "label" -n 2 -ah 143.8449888287663 -ap 143.8449888287663 -o "results/invivotro/rasar/multiclass/new_invivo_eawag_repeated_1nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv  -e "multiclass" -invitro "True" "own"  -invitro_form "number" -n 2 -ah 143.8449888287663 -ap 1274.2749857031322 -o "results/invivotro/rasar/multiclass/new_invivo_eawag_repeated_5nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv  -e "multiclass" -invitro "True" "own"  -invitro_form "number" -n 2 -ah 143.8449888287663 -ap 143.8449888287663 -o "results/invivotro/rasar/multiclass/new_invivo_eawag_repeated_1nn_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv  -e "multiclass" -invitro "True" "False"  -invitro_form "both" -n 2 -ah 'logspace' -ap 'logspace' -o "results/invivotro/rasar/multiclass/rf/new_invivo_eawag_repeated_rasar_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv  -e "multiclass" -invitro "True" -invitro_form "label" -n 2 -ah 'logspace' -ap 'logspace' -o "results/invivotro/rasar/binary/multiclass/new_invivo_eawag_repeated_rasar_"
# python RASAR_simple_addinginvitro.py -i data/invitro/invivo_eawag_repeated_label.csv  -e "multiclass" -invitro "True" -invitro_form "number" -n 2 -ah 'logspace' -ap 'logspace' -o "results/invivotro/rasar/binary/multiclass/new_invivo_eawag_repeated_rasar_"