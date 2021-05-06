from helper_model import *
import argparse
import sys
import h2o
import os
import multiprocessing as mp
import pickle


def getArguments():
    parser = argparse.ArgumentParser(
        description="Datafusion rasar model with adding invitro.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-i2", "--input2", dest="inputFile2", required=True)
    parser.add_argument("-idf",
                        "--inputdatafusion",
                        dest="input_datafusion_File",
                        required=True)
    parser.add_argument("-n",
                        "--neighbors",
                        dest="neighbors",
                        required=True,
                        nargs='+',
                        type=int)
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument("-invitro",
                        "--invitro",
                        dest="wo_invitro",
                        required=True,
                        nargs="+")
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

if __name__ == "__main__":

    non_categorical = [
        "ring_number",
        "tripleBond",
        "doubleBond",
        "alone_atom_number",
        "oh_count",
        "atom_number",
        "bonds_number",
        "Mol",
        "MorganDensity",
        "LogP",
        "water_solubility",
        "melting_point",
    ]
    categorical = [
        "class",
        "tax_order",
        "family",
        "genus",
        "species",
        "control_type",
        "media_type",
        "application_freq_unit",
        "exposure_type",
        "conc1_type",
        "obs_duration_mean",
    ]
    conc = ["conc1_mean"]
    drop_columns = ["Unnamed: 0"]

    if args.encoding == "binary":
        db_mortality, db_datafusion, db_invitro = load_datafusion_datasets_invitro(
            args.inputFile,
            args.input_datafusion_File,
            args.inputFile2,
            categorical,
            non_categorical,
            drop_columns,
            encoding="binary",
            encoding_value=1,
            seed=42,
        )
        db_invitro["invitro_label"] = np.where(db_invitro["ec50"].values > 0.6,
                                               0, 1)
    elif args.encoding == "multiclass":
        db_mortality, db_datafusion, db_invitro = load_datafusion_datasets_invitro(
            args.inputFile,
            args.input_datafusion_File,
            args.inputFile2,
            categorical,
            non_categorical,
            drop_columns,
            encoding="multiclass",
            encoding_value=[0.1, 1, 10, 100],
            seed=42,
        )
        db_invitro["invitro_label"] = multiclass_encoding(
            db_invitro["ec50"], [0.006, 0.3, 63, 398])

    X = db_mortality.drop(columns="conc1_mean").copy()
    Y = db_mortality.conc1_mean.values

    print("calcultaing distance matrix..", ctime())
    basic_mat, matrix_h, matrix_p = cal_matrixs(X, X, categorical,
                                                non_categorical)
    basic_mat_x_df, matrix_h_x_df, matrix_p_x_df = cal_matrixs(
        X,
        db_datafusion.drop(columns="conc1_mean").copy(), categorical,
        non_categorical)
    basic_mat_x_invitro, matrix_h_x_invitro, matrix_p_x_invitro = cal_matrixs(
        X, db_invitro.copy(), categorical_both, non_categorical)
    print("successfully calculated distance matrix..", ctime())
    if args.hamming_alpha == "logspace":
        sequence_ap = np.logspace(-4, 5, 20)
        sequence_ah = sequence_ap
    else:
        sequence_ap = [args.pubchem2d_alpha]
        sequence_ah = [args.hamming_alpha]

    i = 0

    for invitro in args.wo_invitro:
        best_accs = 0
        for n in (args.neighbors):
            for ah in sequence_ah:
                for ap in sequence_ap:
                    print(
                        "*" * 50,
                        round(
                            i / (len(args.neighbors) * len(sequence_ap) *
                                 len(sequence_ah) * len(args.wo_invitro)), 5),
                        ctime(),
                    )
                    distance_matrix = matrix_combine(basic_mat, matrix_h,
                                                     matrix_p, float(ah),
                                                     float(ap))
                    db_datafusion_matrix = matrix_combine(
                        basic_mat_x_df, matrix_h_x_df, matrix_p_x_df,
                        float(ah), float(ap))
                    db_invitro_matrix = matrix_combine(basic_mat_x_invitro,
                                                       matrix_h_x_invitro,
                                                       matrix_p_x_invitro,
                                                       float(ah), float(ap))

                    if args.encoding == "binary":
                        results = cv_datafusion_rasar(
                            db_datafusion_matrix,
                            distance_matrix,
                            db_invitro_matrix,
                            X,
                            Y,
                            db_datafusion,
                            db_invitro,
                            n_neighbors=int(n),
                            train_label=["LC50", "EC50"],
                            train_effect="MOR",
                            invitro=invitro,
                            invitro_form=args.invitro_form,
                            encoding=args.encoding)
                    elif args.encoding == "multiclass":
                        results = cv_datafusion_rasar_multiclass(
                            db_datafusion_matrix,
                            distance_matrix,
                            db_invitro_matrix,
                            X,
                            Y,
                            db_datafusion,
                            n_neighbors=int(n),
                            train_label=["LC50", "EC50"],
                            train_effect="MOR",
                            db_invitro=db_invitro,
                            final_model=False,
                            invitro=invitro,
                            invitro_form=args.invitro_form,
                            encoding=args.encoding)
                    # del distance_matrix, db_datafusion_matrix, db_invitro_matrix
                    if results["avg_accs"] > best_accs:
                        best_results = results
                        best_accs = results["avg_accs"]
                        best_alpha_h = ah
                        best_alpha_p = ap
                    i = i + 1
                    # h2o.shutdown()

            info = []
            info.append(
                """The best params were alpha_h:{}, alpha_p:{}""".format(
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

            filename = (args.outputFile + str(invitro) + "_" + str(n) + "_" +
                        str(args.invitro_form) + "_" + str(args.encoding) +
                        ".txt")
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(filename, "w") as file_handler:
                for item in info:
                    file_handler.write("{}\n".format(item))

# python RASAR_datafusion_model_addinginvitro_general.py  -i  data/LC50/lc50_processed.csv -i2 "data/invitro/invivo_eawag_repeated_label.csv" -idf 'data/LC50/lc50_processed_df_itself.csv' -invitro "True" "False" "own" -invitro_form "both" -n 2 -ah 0.2069138081114788 -ap 0.615848211066026 -o "results/invivotro/rasardf/general/new_lc50_eawag_LC50_5nn_"
# python RASAR_datafusion_model_addinginvitro_general.py  -e "multiclass" -i  data/LC50/lc50_processed.csv -i2 "data/invitro/invivo_eawag_repeated_label.csv" -idf data/LC50/lc50_processed_df_itself.csv -invitro "True" "False" "own" -invitro_form "both" -n 2 3 -ah 0.12742749857031335 -ap 0.3359818286283781 -o "new_lc50_eawag_LC50_5nn_"
# python RASAR_datafusion_model_addinginvitro_general.py  -i  data/LC50/lc50_processed.csv -i2 "data/invitro/invivo_eawag_repeated_label.csv" -idf 'data/LC50/lc50_processed_df_itself.csv' -invitro "True" "False" "own" -invitro_form "both" -n 2 -ah 'logspace' -ap 'logspace' -o "results/invivotro/rasardf/general/new_lc50_eawag_LC50_5nn_"
# python RASAR_datafusion_model_addinginvitro_general.py  -e "multiclass" -i  data/LC50/lc50_processed.csv -i2 "data/invitro/invivo_eawag_repeated_label.csv" -idf data/LC50/lc50_processed_df_itself.csv -invitro "True" "False" "own" -invitro_form "both" -n 2 3 -ah logspace -ap logspace -o "lc50_eawag_rasar_"
