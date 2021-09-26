from helper_model import *
from sklearn.model_selection import train_test_split, ParameterSampler
import h2o
from tqdm import tqdm
import argparse

import os


def getArguments():
    parser = argparse.ArgumentParser(
        description="Running DataFusion RASAR model for invivo datasets or merged invivo & invitro dataset."
    )
    parser.add_argument("-i", "--input", help="inputFile position", required=True)
    parser.add_argument(
        "-idf", "--input_df", help="input datafusion File position", required=True
    )
    parser.add_argument("-e", "--encoding", help="encoding", default="binary")
    parser.add_argument(
        "-il",
        "--invitro_label",
        help=" input invitro form: number, label, both, representing using the concentration value\
             of invitro experiment, labeled class value of the invitro experiment, or both",
        default="number",
    )

    parser.add_argument(
        "-wi",
        "--w_invitro",
        help="using the invitro as input or not: True, False, own;\
         representing using invivo plus invitro information as input, using only invivo information as input\
             using only invitro information as input",
        default="False",
    )
    parser.add_argument(
        "-ah", "--alpha_h", help="alpha_hamming", required=True, nargs="?"
    )
    parser.add_argument(
        "-ap", "--alpha_p", help="alpha_pubchem", required=True, nargs="?"
    )
    parser.add_argument(
        "-n",
        "--n_neighbors",
        help="number of neighbors in the RASAR model",
        nargs="?",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-endpoint", "--train_endpoint", help="train_endpoint", required=True
    )
    parser.add_argument("-effect", "--train_effect", help="train_effect", required=True)
    parser.add_argument("-o", "--output", help="outputFile", default="binary.txt")
    return parser.parse_args()


# example:
# python .../RASAR_df.py -i .../lc50_processed.csv  -idf  .../datafusion.csv  -endpoint ['LC50','EC50'] -effect 'MOR'  -ah 0.1 -ap 0.1 -o df_rasar.txt
# python .../RASAR_df.py -i1 .../lc50_processed_w_invitro.csv -idf  .../datafusion.csv -il label -wi True -endpoint ['LC50','EC50'] -effect 'MOR' -ah 0.1 -ap 0.1 -o .../df_rasar_invitro.txt


args = getArguments()
if args.encoding == "binary":
    encoding = "binary"
    encoding_value = 1
elif args.encoding == "multiclass":
    encoding = "multiclass"
    encoding_value = [0.1, 1, 10, 100]

# -------------------loading data & preprocessing--------------------
db_mortality, db_datafusion = load_datafusion_datasets(
    args.input,
    args.input_df,
    categorical_columns=categorical,
    encoding=encoding,
    encoding_value=encoding_value,
)

X = db_mortality.drop(columns="conc1_mean").copy()
Y = db_mortality.conc1_mean.values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("Data loaded.", ctime())

print("calcultaing distance matrix..", ctime())
matrix_euc, matrix_h, matrix_p = cal_matrixs(
    X_train, X_train, categorical, non_categorical
)
print("calcultaing datafusion distance matrix..", ctime())
matrix_euc_df, matrix_h_df, matrix_p_df = cal_matrixs(
    X_train,
    db_datafusion.drop(columns="conc1_mean").copy(),
    categorical,
    non_categorical,
)
print("distance matrixes successfully calculated!", ctime())


if args.alpha_h == "logspace":
    sequence_ap = np.logspace(-2, 0, 20)
    sequence_ah = sequence_ap
else:
    sequence_ap = [float(args.alpha_p)]
    sequence_ah = [float(args.alpha_h)]

if args.w_invitro == "True":
    db_invitro = "overlap"
else:
    db_invitro = "noinvitro"


# -------------------training --------------------
if encoding == "binary":
    model = RandomForestClassifier(random_state=10)
    hyper_params_tune = {
        "max_depth": [i for i in range(10, 30, 6)],
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=1000, num=11)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
    }
elif encoding == "multiclass":
    h2o.init()
    h2o.no_progress()
    model = H2ORandomForestEstimator(seed=10)
    hyper_params_tune = {
        "ntrees": [i for i in range(10, 1000, 10)],
        "max_depth": [i for i in range(10, 1000, 10)],
        "min_rows": [1, 10, 100, 1000],
        "sample_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    }

params_comb = list(ParameterSampler(hyper_params_tune, n_iter=20, random_state=2))

best_accs = 0
best_p = dict()

count = 1
for ah in sequence_ah:
    for ap in sequence_ap:
        for i in range(0, len(params_comb)):
            print(
                "*" * 50,
                count / (len(sequence_ap) ** 2 * len(params_comb)),
                ctime(),
                end="\r",
            )
            for k, v in params_comb[i].items():
                setattr(model, k, v)

            results = cv_datafusion_rasar(
                matrix_euc,
                matrix_h,
                matrix_p,
                matrix_euc_df,
                matrix_h_df,
                matrix_p_df,
                db_invitro_matrix="nan",
                ah=ah,
                ap=ap,
                X=X_train,
                Y=Y_train,
                db_datafusion=db_datafusion,
                train_endpoint=args.train_endpoint,
                train_effect=args.train_effect,
                model=model,
                n_neighbors=args.n_neighbors,
                invitro=args.w_invitro,
                invitro_form=args.invitro_label,
                db_invitro=db_invitro,
                encoding=encoding,
            )

            if results["avg_accs"] > best_accs:
                best_p = params_comb[i]
                best_accs = results["avg_accs"]
                best_results = results
                best_ah = ah
                best_ap = ap
                print("success.", best_accs)

            count = count + 1

# -------------------tested on test dataset--------------------
print("start testing...", ctime())
for k, v in best_p.items():
    setattr(model, k, v)

train_index = X_train.index
test_index = X_test.index


matrix_euc, matrix_h, matrix_p = cal_matrixs(X, X, categorical, non_categorical)
matrix_euc_df, matrix_h_df, matrix_p_df = cal_matrixs(
    X, db_datafusion.drop(columns="conc1_mean").copy(), categorical, non_categorical
)

matrix_euc = pd.DataFrame(matrix_euc)
max_euc = matrix_euc.iloc[train_index, train_index].values.max()

matrix = pd.DataFrame(
    best_ah * matrix_h + best_ap * matrix_p + matrix_euc.divide(max_euc).values
)
db_datafusion_matrix = pd.DataFrame(
    best_ah * matrix_h_df
    + best_ap * matrix_p_df
    + pd.DataFrame(matrix_euc_df).divide(max_euc).values
)

del (matrix_euc, matrix_h, matrix_p, matrix_euc_df, matrix_h_df, matrix_p_df)

simple_rasar_train, simple_rasar_test = cal_data_simple_rasar(
    matrix.iloc[train_index.astype("int64"), train_index.astype("int64")],
    matrix.iloc[test_index.astype("int64"), train_index.astype("int64")],
    Y_train,
    args.n_neighbors,
    encoding,
)

datafusion_rasar_train, datafusion_rasar_test = cal_data_datafusion_rasar(
    train_index,
    test_index,
    X_train,
    X_test,
    db_datafusion,
    db_datafusion_matrix,
    args.train_endpoint,
    args.train_effect,
    encoding,
)
del (matrix, db_datafusion_matrix)

train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis=1)
test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)

invitro_form = args.invitro_label
invitro = args.w_invitro

# adding invitro information
if str(db_invitro) == "overlap":
    if (invitro != "False") & (invitro_form == "number"):
        train_rf["invitro_conc"] = X_train.invitro_conc.reset_index(drop=True)
        test_rf["invitro_conc"] = X_test.invitro_conc.reset_index(drop=True)

    elif (invitro != "False") & (invitro_form == "label"):
        train_rf["invitro_label"] = X_train.invitro_label_half.reset_index(drop=True)
        test_rf["invitro_label"] = X_test.invitro_label_half.reset_index(drop=True)

    elif (invitro != "False") & (invitro_form == "both"):
        train_rf["invitro_conc"] = X_train.invitro_conc.reset_index(drop=True)
        test_rf["invitro_conc"] = X_test.invitro_conc.reset_index(drop=True)
        train_rf["invitro_label"] = X_train.invitro_label_half.reset_index(drop=True)
        test_rf["invitro_label"] = X_test.invitro_label_half.reset_index(drop=True)
    elif (invitro != "False") & (invitro_form == "label_half"):
        train_rf["invitro_label_half"] = X.iloc[
            train_index, :
        ].invitro_label.reset_index(drop=True)
        test_rf["invitro_label_half"] = X.iloc[test_index, :].invitro_label.reset_index(
            drop=True
        )

    elif (invitro != "False") & (invitro_form == "both_half"):
        train_rf["invitro_conc"] = X.iloc[train_index, :].invitro_conc.reset_index(
            drop=True
        )
        test_rf["invitro_conc"] = X.iloc[test_index, :].invitro_conc.reset_index(
            drop=True
        )
        train_rf["invitro_label_half"] = X.iloc[
            train_index, :
        ].invitro_label.reset_index(drop=True)
        test_rf["invitro_label_half"] = X.iloc[test_index, :].invitro_label.reset_index(
            drop=True
        )


del (
    datafusion_rasar_test,
    datafusion_rasar_train,
    simple_rasar_test,
    simple_rasar_train,
)

print(train_rf.columns)

# calculateing the scores
if encoding == "binary":

    model.fit(train_rf, Y_train)
    y_pred = model.predict(test_rf)

    accs = accuracy_score(Y_test, y_pred)
    sens = recall_score(Y_test, y_pred, average="macro")
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()
    specs = tn / (tn + fp)
    precs = precision_score(Y_test, y_pred, average="macro")
    f1 = f1_score(Y_test, y_pred, average="macro")
elif encoding == "multiclass":

    train_rf.loc[:, "target"] = Y_train
    test_rf.loc[:, "target"] = Y_test

    train_rf_h2o = h2o.H2OFrame(train_rf)
    test_rf_h2o = h2o.H2OFrame(test_rf)

    for col in train_rf.columns:
        if "label" in col:
            train_rf_h2o[col] = train_rf_h2o[col].asfactor()
            test_rf_h2o[col] = test_rf_h2o[col].asfactor()

    train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
    test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

    model.train(y="target", training_frame=train_rf_h2o)
    y_pred = model.predict(test_rf_h2o).as_data_frame()["predict"]

    accs = accuracy_score(Y_test, y_pred)
    sens = recall_score(Y_test, y_pred, average="macro")
    specs = np.nan
    precs = precision_score(Y_test, y_pred, average="macro")
    f1 = f1_score(Y_test, y_pred, average="macro")

print(
    """Accuracy:  {}, Se.Accuracy:  {} 
		\nSensitivity:  {}, Se.Sensitivity: {}
        \nSpecificity:  {}, Se.Specificity:{}
		\nPrecision:  {}, Se.Precision: {}
		\nf1_score:{}, Se.f1_score:{}""".format(
        accs,
        best_results["se_accs"],
        sens,
        best_results["se_sens"],
        specs,
        best_results["se_specs"],
        precs,
        best_results["se_precs"],
        f1,
        best_results["se_f1"],
    )
)

# ----------------save the information into a file-------
info = []

info.append(
    """Accuracy:  {}, Se.Accuracy:  {} 
    \nSensitivity:  {}, Se.Sensitivity: {}
        \nSpecificity:  {}, Se.Specificity:{}
    \nPrecision:  {}, Se.Precision: {}
    \nf1_score:{}, Se.f1_score:{}""".format(
        accs,
        best_results["se_accs"],
        sens,
        best_results["se_sens"],
        specs,
        best_results["se_specs"],
        precs,
        best_results["se_precs"],
        f1,
        best_results["se_f1"],
    )
)

info.append(
    "alpha_h:{}, alpha_p: {}, best hyperpatameters:{}".format(best_ah, best_ap, best_p)
)


str2file(info, args.output)
