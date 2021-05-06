import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    precision_score,
    accuracy_score,
    mean_squared_error,
    f1_score,
)
from time import ctime
from math import sqrt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
import warnings
import heapq
from numpy.lib.stride_tricks import as_strided
from collections import Counter

warnings.filterwarnings("ignore")

# categorical features will ordinal encoded

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
# non_categorical was numerical features, whcih will be standarized. \
# Mol,bonds_number, atom_number was previously log transformed due to the maginitude of their values.

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
# comparing was used to identify similar experiments
comparing = ["test_cas"] + categorical

non_categorical_invitro = [
    "logkow",
    "water_solubility",
    "henry_constant",
    "atom_number",
    "alone_atom_number",
    "bonds_number",
    "doubleBond",
    "tripleBond",
    "ring_number",
    "Mol",
    "MorganDensity",
    "LogP",
    "oh_count",
    "melting_point",
    "ntc",
]

categorical_invitro = [
    "cell_line",
    "endpoint",
    "timepoint_hour",
    "insert",
    "passive_dosing",
    "plate_size",
    "solvent",
    "conc_determination_nominal_or_measured",
    "medium",
    "fbs",
]
categorical_both = ["class", "tax_order", "family", "genus", "species"]

tqdm.pandas(desc="Bar")


def load_data_mlp(DATA_PATH, encoding, encoding_value=1, seed=42):
    db = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"]).drop_duplicates()
    db_raw = db.copy()

    # MinMax trasform for numerical variables
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])

    # One hot Encoding for categorical variables
    for i in categorical:
        jobs_encoder = LabelBinarizer()
        jobs_encoder.fit(db[i])
        transformed = jobs_encoder.transform(db[i])
        if jobs_encoder.classes_.shape[0] > 2:
            ohe_df = pd.DataFrame(
                transformed,
                columns=[
                    str(i) + "_" + str(j)
                    for j in range(jobs_encoder.classes_.shape[0])
                ],
            )
        else:
            ohe_df = pd.DataFrame(transformed, columns=[str(i) + "_0"])
        db = pd.concat([db, ohe_df], axis=1).drop([i], axis=1)

    # # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        db["conc1_mean"] = np.where(db["conc1_mean"].values > encoding_value,
                                    0, 1)
        print(encoding_value)

    elif encoding == "multiclass":
        label = db["conc1_mean"].copy()
        db["conc1_mean"] = multiclass_encoding(label, encoding_value)
        print(encoding_value)

    X = db.drop(columns="conc1_mean")
    Y = db["conc1_mean"].values
    return X, Y


def load_data_mlp_repeat(DATA_PATH1,
                         DATA_PATH2,
                         encoding,
                         encoding_value=1,
                         seed=42):
    db1 = pd.read_csv(DATA_PATH1).drop(
        columns=["Unnamed: 0"]).drop_duplicates()
    db2 = pd.read_csv(DATA_PATH2).drop(
        columns=["Unnamed: 0"]).drop_duplicates()
    db = pd.concat([db1, db2], axis=0).reset_index(drop=True)
    # MinMax trasform for numerical variables
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])

    # One hot Encoding for categorical variables
    for i in categorical:
        jobs_encoder = LabelBinarizer()
        jobs_encoder.fit(db[i])
        transformed = jobs_encoder.transform(db[i])
        if jobs_encoder.classes_.shape[0] > 2:
            ohe_df = pd.DataFrame(
                transformed,
                columns=[
                    str(i) + "_" + str(j)
                    for j in range(jobs_encoder.classes_.shape[0])
                ],
            )
        else:
            ohe_df = pd.DataFrame(transformed)
        db = pd.concat([db, ohe_df], axis=1).drop([i], axis=1)

    # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        db["conc1_mean"] = np.where(db["conc1_mean"].values > encoding_value,
                                    0, 1)
        print(encoding_value)

    elif encoding == "multiclass":
        label = db["conc1_mean"].copy()
        db["conc1_mean"] = multiclass_encoding(label, encoding_value)
        print(encoding_value)

    X_norepeat = db[db1.shape[0]:].reset_index(drop=True).drop(
        columns="conc1_mean")
    Y_norepeat = db[db1.shape[0]:].reset_index(drop=True)["conc1_mean"].values
    X_repeat = db[:db1.shape[0]].reset_index(drop=True).drop(
        columns="conc1_mean")
    Y_repeat = db[:db1.shape[0]].reset_index(drop=True)["conc1_mean"].values
    return X_repeat, Y_repeat, X_norepeat, Y_norepeat


def load_data(DATA_PATH,
              encoding,
              categorical_columns,
              non_categorical_columns,
              drop_columns=["Unnamed: 0"],
              conc_column="conc1_mean",
              encoding_value=1,
              seed=42):
    db = pd.read_csv(DATA_PATH).drop(drop_columns, axis=1)
    for nc in non_categorical_columns:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])

    # Ordinal Encoding for categorical variables
    encoder = OrdinalEncoder(dtype=int)
    encoder.fit(db[categorical_columns])
    db[categorical_columns] = encoder.transform(db[categorical_columns]) + 1

    # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        if encoding_value == "own":
            print(np.median(db[conc_column]))
            db[conc_column] = np.where(
                db[conc_column].values > np.median(db[conc_column]), 0, 1)
        else:
            print(encoding_value)
            db[conc_column] = np.where(db[conc_column].values > encoding_value,
                                       0, 1)

    elif encoding == "multiclass":
        if encoding_value == "own":
            print(db[conc_column].quantile([0.2, 0.4, 0.6, 0.8]).values)
            db[conc_column] = multiclass_encoding(
                db[conc_column].copy(),
                db[conc_column].quantile([0.2, 0.4, 0.6, 0.8]).values,
            )
        else:
            print(encoding_value)
            db[conc_column] = multiclass_encoding(db[conc_column].copy(),
                                                  encoding_value)

    X = db.drop(columns=conc_column)
    Y = db[conc_column].values
    return X, Y


def load_data_fish(DATA_PATH,
                   DATA_PATH_fish,
                   encoding,
                   encoding_value=1,
                   seed=42):
    db = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"]).drop_duplicates()
    db_fish = pd.read_csv(DATA_PATH_fish).drop(
        columns=["Unnamed: 0"]).drop_duplicates()
    db_raw = db_fish.copy()

    # MinMax trasform for numerical variables
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])
        db_fish[[nc]] = minmax.transform(db_fish[[nc]])

    # Ordinal Encoding for categorical variables
    db, db_fish = encoding_categorical(categorical, db, db_fish)

    # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        db["conc1_mean"] = np.where(db["conc1_mean"].values > encoding_value,
                                    0, 1)
        db_fish["conc1_mean"] = np.where(
            db_fish["conc1_mean"].values > encoding_value, 0, 1)
        print(encoding_value)

    elif encoding == "multiclass":
        db["conc1_mean"] = multiclass_encoding(db["conc1_mean"].copy(),
                                               encoding_value)
        db_fish["conc1_mean"] = multiclass_encoding(
            db_fish["conc1_mean"].copy(), encoding_value)
        print(encoding_value)

    X = db.drop(columns="conc1_mean")
    Y = db["conc1_mean"].values
    X_fish = db_fish.drop(columns="conc1_mean")
    Y_fish = db_fish["conc1_mean"].values
    return X, Y, X_fish, Y_fish


def load_invivo_invitro(DATA_PATH,
                        DATA_PATH_invitro,
                        encoding,
                        encoding_value=1,
                        seed=42):
    # load invivo & invitro datasets
    db = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"]).drop_duplicates()
    db_invitro = (pd.read_csv(DATA_PATH_invitro).drop(
        columns=["Unnamed: 0"]).drop_duplicates())
    db_invitro = db_invitro[non_categorical +
                            ["test_cas", "ec50", "pubchem2d"]]

    db_invitro["species"] = "mykiss"
    db_invitro["class"] = "Actinopterygii"
    db_invitro["tax_order"] = "Salmoniformes"
    db_invitro["family"] = "Salmonidae"
    db_invitro["genus"] = "Oncorhynchus"

    # MinMax trasform for numerical variables
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])
        db_invitro[[nc]] = minmax.transform(db_invitro[[nc]])

    # Ordinal Encoding for categorical variables
    db, db_invitro = encoding_categorical(categorical_both, db, db_invitro)

    cate = [i for i in categorical if i not in categorical_both]
    encoder = OrdinalEncoder(dtype=int)
    encoder.fit(db[cate])
    db[cate] = encoder.transform(db[cate]) + 1

    # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        db["conc1_mean"] = np.where(db["conc1_mean"].values > encoding_value,
                                    0, 1)
        print(encoding_value)

    elif encoding == "multiclass":
        db["conc1_mean"] = multiclass_encoding(db["conc1_mean"].copy(),
                                               encoding_value)
        print(encoding_value)

    X = db.drop(columns="conc1_mean")
    Y = db["conc1_mean"].values

    return X, Y, db_invitro


def hamming_matrix(X1, X2, cat_features):
    return squareform(pdist(X1[cat_features], metric="hamming"))


def hamming_matrix_df(X1, X2, cat_features):
    return cdist(X1[cat_features], X2[cat_features], metric="hamming")


def euclidean_matrix_df(X1, X2, num_features):
    return cdist(X1[num_features], X2[num_features], metric="euclidean")


def euclidean_matrix(X1, X2, num_features):
    return squareform(pdist(X1[num_features], metric="euclidean"))


def change_label(x):
    return x.str.replace("", " ").str.strip().str.split(" ")


def pubchem2d_matrix(X1, X2):
    return squareform(
        pdist(pd.DataFrame([
            x.replace("", " ").strip().split(' ')
            for x in list(X1["pubchem2d"])
        ]),
              metric="hamming"))


def pubchem2d_matrix_df(X1, X2):
    df_1 = pd.DataFrame(
        [x.replace("", " ").strip().split(' ') for x in list(X1["pubchem2d"])])

    df_2 = pd.DataFrame(
        [x.replace("", " ").strip().split(' ') for x in list(X2["pubchem2d"])])

    return cdist(df_1, df_2, metric="hamming")


def cal_matrixs(X1, X2, categorical, non_categorical):
    if X1.shape == X2.shape:
        if np.all(X1 == X2):
            basic_mat = euclidean_matrix(X1, X2, non_categorical)
            matrix_h = hamming_matrix(X1, X2, categorical)
            matrix_p = pubchem2d_matrix(X1, X2)
    else:
        basic_mat = euclidean_matrix_df(X1, X2, non_categorical)
        matrix_h = hamming_matrix_df(X1, X2, categorical)
        matrix_p = pubchem2d_matrix_df(X1, X2)
    return basic_mat, matrix_h, matrix_p


# euclidean matrix will always has 1 as parameter.
def matrix_combine(basic_mat, matrix_h, matrix_p, ah, ap):
    dist_matr = ah * matrix_h
    dist_matr += basic_mat
    dist_matr += ap * matrix_p
    dist_matr = pd.DataFrame(dist_matr)
    return dist_matr


def dist_matrix(X1, X2, non_categorical, categorical, ah, ap):
    if X1.shape == X2.shape:
        if np.all(X1 == X2):
            matrix_h = hamming_matrix(X1, X2, categorical)
            dist_matr = ah * matrix_h
            del matrix_h
            basic_mat = euclidean_matrix(X1, X2, non_categorical)
            dist_matr += basic_mat
            del basic_mat
            matrix_p = pubchem2d_matrix(X1, X2)
            dist_matr += ap * matrix_p
            del matrix_p
            dist_matr = pd.DataFrame(dist_matr)
    else:
        matrix_h = hamming_matrix_df(X1, X2, categorical)
        dist_matr = ah * matrix_h
        del matrix_h
        basic_mat = euclidean_matrix_df(X1, X2, non_categorical)
        dist_matr += basic_mat
        del basic_mat
        matrix_p = pubchem2d_matrix_df(X1, X2)
        dist_matr += ap * matrix_p
        del matrix_p
        dist_matr = pd.DataFrame(dist_matr)
    return dist_matr


def multiclass_encoding(var, threshold=[10**-1, 10**0, 10**1, 10**2]):
    var_ls = list(var)
    for i in range(len(var_ls)):
        if var_ls[i] <= threshold[0]:
            var_ls[i] = 5

        elif threshold[0] < var_ls[i] <= threshold[1]:
            var_ls[i] = 4

        elif threshold[1] < var_ls[i] <= threshold[2]:
            var_ls[i] = 3

        elif threshold[2] < var_ls[i] <= threshold[3]:
            var_ls[i] = 2

        else:
            var_ls[i] = 1
    return pd.to_numeric(var_ls, downcast="integer")


def encoding_categorical(cat_cols, database, database_df):
    # fit to the categories in datafusion dataset and training dataset
    categories = []
    for column in database[cat_cols]:
        # add non-existing new category in df dataset but not in training dataset.
        cat_final = np.append(
            database[column].unique(),
            tuple(i for i in database_df[column].unique()
                  if i not in database[column].unique()),
        )
        cat_final = np.array(sorted(cat_final))
        categories.append(cat_final)

    encoder = OrdinalEncoder(categories=categories, dtype=int)

    encoder.fit(database[cat_cols])
    database[cat_cols] = encoder.transform(database[cat_cols]) + 1

    encoder.fit(database_df[cat_cols])
    database_df[cat_cols] = encoder.transform(database_df[cat_cols]) + 1

    return database, database_df


def load_datafusion_datasets(
    DATA_MORTALITY_PATH,
    DATA_OTHER_ENDPOINT_PATH,
    categorical_columns,
    non_categorical_columns,
    drop_columns=["Unnamed: 0"],
    conc_column="conc1_mean",
    fixed="no",
    encoding="binary",
    encoding_value=1,
    seed=42,
):

    class_threshold = pd.read_csv("data/threshold.csv")
    db_datafusion = pd.read_csv(DATA_OTHER_ENDPOINT_PATH).drop(
        columns=drop_columns)
    db_mortality = pd.read_csv(DATA_MORTALITY_PATH).drop(columns=drop_columns)
    db_raw = db_mortality.copy()

    # MinMax trasform for numerical variables
    for nc in non_categorical_columns:
        minmax = MinMaxScaler()
        minmax.fit(db_mortality[[nc]])
        db_mortality[[nc]] = minmax.transform(db_mortality[[nc]])
        db_datafusion[[nc]] = minmax.transform(db_datafusion[[nc]])

    # Ordinal encoding for categorical variables
    db_mortality, db_datafusion = encoding_categorical(categorical_columns,
                                                       db_mortality,
                                                       db_datafusion)

    # concentration class labeling
    if encoding == "binary":
        db_mortality["conc1_mean"] = np.where(
            db_mortality["conc1_mean"].values > encoding_value, 0, 1)

        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, "conc1_mean"]
            if fixed == "yes":
                print("fixed threshold")
                threshold = class_threshold[class_threshold.effect == ef]
                db_datafusion.loc[db_datafusion.effect == ef,
                                  "conc1_mean"] = np.where(
                                      conc > float(threshold.ths_binary), 0, 1)
            else:
                # print('unfixed threshold')
                db_datafusion.loc[db_datafusion.effect == ef,
                                  "conc1_mean"] = np.where(
                                      conc > np.median(conc), 0, 1)
        # mortality always classified using 1
        # db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'] = np.where(conc > encoding_value, 0, 1)
    elif encoding == "multiclass":

        t = db_mortality["conc1_mean"].copy()
        db_mortality["conc1_mean"] = multiclass_encoding(t, encoding_value)
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef,
                                     "conc1_mean"].copy()
            if fixed == "yes":
                print("fixed threshold")
                threshold = class_threshold[class_threshold.effect == ef]
                db_datafusion.loc[db_datafusion.effect == ef,
                                  "conc1_mean"] = multiclass_encoding(
                                      conc.values,
                                      [
                                          float(threshold.ths_1),
                                          float(threshold.ths_2),
                                          float(threshold.ths_3),
                                          float(threshold.ths_4),
                                      ],
                                  )
            else:
                print("unfixed threshold")
                db_datafusion.loc[db_datafusion.effect == ef,
                                  "conc1_mean"] = multiclass_encoding(
                                      conc.values,
                                      conc.quantile([0.2, 0.4, 0.6,
                                                     0.8]).values)
        # mortality always classified using 0.1，1，10，100
        # conc = db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'].copy()
        # db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'] = multiclass_encoding(conc.values, encoding_value)
    return db_mortality, db_datafusion


def load_datafusion_datasets_invitro(
    DATA_MORTALITY_PATH,
    DATA_OTHER_ENDPOINT_PATH,
    DATA_INVITRO_PATH,
    categorical_columns,
    non_categorical_columns,
    drop_columns=["Unnamed: 0"],
    conc_column="conc1_mean",
    fixed="no",
    encoding="binary",
    encoding_value=1,
    seed=42,
):

    class_threshold = pd.read_csv("data/threshold.csv")
    db_datafusion = pd.read_csv(DATA_OTHER_ENDPOINT_PATH).drop(
        columns=drop_columns)
    db_mortality = pd.read_csv(DATA_MORTALITY_PATH).drop(columns=drop_columns)
    db_raw = db_mortality.copy()
    db_invitro = (pd.read_csv(DATA_INVITRO_PATH).drop(
        columns=["Unnamed: 0"]).drop_duplicates())
    db_invitro = db_invitro[non_categorical +
                            ["test_cas", "ec50", "pubchem2d"]]

    db_invitro["species"] = "mykiss"
    db_invitro["class"] = "Actinopterygii"
    db_invitro["tax_order"] = "Salmoniformes"
    db_invitro["family"] = "Salmonidae"
    db_invitro["genus"] = "Oncorhynchus"

    # MinMax trasform for numerical variables
    for nc in non_categorical_columns:
        minmax = MinMaxScaler()
        minmax.fit(db_mortality[[nc]])
        db_mortality[[nc]] = minmax.transform(db_mortality[[nc]])
        db_datafusion[[nc]] = minmax.transform(db_datafusion[[nc]])
        db_invitro[[nc]] = minmax.transform(db_invitro[[nc]])

    # Ordinal encoding for categorical variables
    db_mortality, db_datafusion = encoding_categorical(categorical_columns,
                                                       db_mortality,
                                                       db_datafusion)
    db_raw, db_invitro = encoding_categorical(categorical_both, db_raw,
                                              db_invitro)

    # concentration class labeling
    if encoding == "binary":
        db_mortality["conc1_mean"] = np.where(
            db_mortality["conc1_mean"].values > encoding_value, 0, 1)
        db_invitro["invitro_label"] = np.where(db_invitro["ec50"].values > 0.6,
                                               0, 1)
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, "conc1_mean"]
            if fixed == "yes":
                print("fixed threshold")
                threshold = class_threshold[class_threshold.effect == ef]
                db_datafusion.loc[db_datafusion.effect == ef,
                                  "conc1_mean"] = np.where(
                                      conc > float(threshold.ths_binary), 0, 1)
            else:
                # print('unfixed threshold')
                db_datafusion.loc[db_datafusion.effect == ef,
                                  "conc1_mean"] = np.where(
                                      conc > np.median(conc), 0, 1)
        # mortality always classified using 1
        # db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'] = np.where(conc > encoding_value, 0, 1)
    elif encoding == "multiclass":

        t = db_mortality["conc1_mean"].copy()
        db_mortality["conc1_mean"] = multiclass_encoding(t, encoding_value)
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef,
                                     "conc1_mean"].copy()
            if fixed == "yes":
                print("fixed threshold")
                threshold = class_threshold[class_threshold.effect == ef]
                db_datafusion.loc[db_datafusion.effect == ef,
                                  "conc1_mean"] = multiclass_encoding(
                                      conc.values,
                                      [
                                          float(threshold.ths_1),
                                          float(threshold.ths_2),
                                          float(threshold.ths_3),
                                          float(threshold.ths_4),
                                      ],
                                  )
            else:
                print("unfixed threshold")
                db_datafusion.loc[db_datafusion.effect == ef,
                                  "conc1_mean"] = multiclass_encoding(
                                      conc.values,
                                      conc.quantile([0.2, 0.4, 0.6,
                                                     0.8]).values)
        # mortality always classified using 0.1，1，10，100
        # conc = db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'].copy()
        # db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'] = multiclass_encoding(conc.values, encoding_value)
    return db_mortality, db_datafusion, db_invitro


# ----------------------------------------------------------------------KNN model------------------------------------------------------------


def select_alpha(X_train, sequence_ham, Y_train, categorical_columns,
                 non_categorical_columns, leaf_ls, neighbors):
    print("calcaulting the matrix...")
    basic_mat, matrix_h, matrix_p = cal_matrixs(X_train, X_train,
                                                categorical_columns,
                                                non_categorical_columns)
    best_alpha_h = 0
    best_alpha_p = 0
    best_accs = 0

    print("Start selecting the best parameters....")
    for ah in tqdm(sequence_ham):
        for ap in sequence_ham:
            for leaf in leaf_ls:
                for neigh in neighbors:
                    dist_matr = matrix_combine(basic_mat, matrix_h, matrix_p,
                                               ah, ap)
                    accs, rmse, sens, precs, f1 = KNN_model(
                        dist_matr, Y_train, leaf, neigh)
                    avg_accs = np.mean(accs)
                    se_accs = sem(accs)
                    avg_rmse = np.mean(rmse)
                    se_rmse = sem(rmse)
                    avg_sens = np.mean(sens)
                    se_sens = sem(sens)
                    avg_precs = np.mean(precs)
                    se_precs = sem(precs)
                    avg_f1 = np.mean(f1)
                    se_f1 = sem(f1)
                    if avg_accs > best_accs:
                        print(
                            """New best params found! alpha_h:{}, alpha_p:{},rmse:{},accs:{}, leaf:{},neighbor:{}"""
                            .format(ah, ap, avg_rmse, avg_accs, leaf, neigh))
                        best_alpha_h = ah
                        best_alpha_p = ap
                        best_rmse = avg_rmse
                        best_accs = avg_accs
                        best_leaf = leaf
                        best_neighbor = neigh

    return best_alpha_h, best_alpha_p, best_leaf, best_neighbor


def KNN_model(X, Y, leaf, neighbor, seed=25):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    rmse = []
    sens = []
    precs = []
    f1 = []
    for train_index, test_index in kf.split(X):
        x_train = X.iloc[train_index, train_index]
        x_test = X.iloc[test_index, train_index]
        y_train = Y[train_index]
        y_test = Y[test_index]

        neigh = KNeighborsClassifier(n_neighbors=neighbor,
                                     metric="precomputed",
                                     leaf_size=leaf)
        neigh.fit(x_train, y_train.astype("int").ravel())
        y_pred = neigh.predict(x_test)

        accs.append(accuracy_score(y_test, y_pred))
        sens.append(recall_score(y_test, y_pred, average="weighted"))
        precs.append(precision_score(y_test, y_pred, average="weighted"))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        f1.append(f1_score(y_test, y_pred, average="weighted"))
    return accs, rmse, sens, precs, f1


# --------------------------------------------------------------------- RASAR---------------------------------------------------------


def find_neighbor(
    square_matr,
    matr_all_label,
    X_train,
    y_train,
    label,
    encoding="binary",
    exp="train",
    n_neighbors=2,
):

    idx_neigh = pd.DataFrame()
    dist = pd.DataFrame()
    knn = KNeighborsClassifier(metric="precomputed",
                               leaf_size=30,
                               n_neighbors=n_neighbors)

    if len(y_train[y_train == label]) == 1:
        idx_neigh = np.array(square_matr.index * matr_all_label.shape[0])
        dist = pd.DataFrame(np.array([0] * matr_all_label.shape[0]))
    else:
        knn.fit(square_matr, y_train[y_train == label])
        neigh = knn.kneighbors(matr_all_label, return_distance=True)

        for i in range(n_neighbors):
            idx_neigh["idx_neigh" + str(label) + str(i)] = pd.DataFrame(
                neigh[1])[i].apply(
                    lambda x: X_train.iloc[y_train == label].iloc[x].name)
            dist["dist_neigh" + str(label) + str(i)] = pd.DataFrame(
                neigh[0])[i]

        # if the nearest point index equals itself, then replacing it with the second nearst point
        # Distance from the Nearest Neighbor that is NOT itself
        if exp == "train":
            ls = X_train.index == idx_neigh["idx_neigh" + str(label) + str(0)]
            for i in range(n_neighbors - 1):
                idx_neigh["idx_neigh" + str(label) +
                          str(i)][ls] = idx_neigh["idx_neigh" + str(label) +
                                                  str(i + 1)][ls].values
                dist["dist_neigh" + str(label) +
                     str(i)][ls] = dist["dist_neigh" + str(label) +
                                        str(i + 1)][ls].values

        idx_neigh.drop(columns="idx_neigh" + str(label) + str(n_neighbors - 1),
                       inplace=True)
        dist.drop(columns="dist_neigh" + str(label) + str(n_neighbors - 1),
                  inplace=True)

    return idx_neigh, dist


def find_neighbor_invitro(rasar_train, rasar_test, db_invitro,
                          db_mortality_train, db_mortality_test, ah, ap):
    invitro_matrix = dist_matrix(db_invitro, db_invitro, non_categorical,
                                 categorical_both, ah, ap)
    knn = KNeighborsClassifier(metric="precomputed", n_neighbors=1)
    knn.fit(pd.DataFrame(invitro_matrix), np.repeat(0,
                                                    invitro_matrix.shape[0]))

    train_invitro_matrix = dist_matrix(db_mortality_train, db_invitro,
                                       non_categorical, categorical_both, ah,
                                       ap)
    neigh = knn.kneighbors(train_invitro_matrix, return_distance=True)
    # print(rasar_train.shape)
    # print(len(neigh[0]))
    rasar_train[["invitro_conc"]] = (pd.DataFrame(
        neigh[1]).apply(lambda x: db_invitro["ec50"][x]).reset_index(
            drop=True))
    rasar_train[["invitro_label"]] = (pd.DataFrame(
        neigh[1]).apply(lambda x: db_invitro["invitro_label"][x]).reset_index(
            drop=True))
    rasar_train[["invitro_dist"]] = pd.DataFrame(neigh[0])
    # print(len(pd.DataFrame(neigh[0]).values))

    test_invitro_matrix = dist_matrix(db_mortality_test, db_invitro,
                                      non_categorical, categorical_both, ah,
                                      ap)
    neigh = knn.kneighbors(test_invitro_matrix, return_distance=True)

    rasar_test[["invitro_conc"]] = (pd.DataFrame(
        neigh[1]).apply(lambda x: db_invitro["ec50"][x]).reset_index(
            drop=True))
    rasar_test[["invitro_label"]] = (pd.DataFrame(
        neigh[1]).apply(lambda x: db_invitro["invitro_label"][x]).reset_index(
            drop=True))
    rasar_test[["invitro_dist"]] = pd.DataFrame(neigh[0])
    # print(len(pd.DataFrame(neigh[0]).values))

    return rasar_train, rasar_test


def take_per_row_strided(A, indx, num_elem=2):
    m, n = A.shape
    # A.shape = (-1)
    A = A.reshape(-1)
    s0 = A.strides[0]
    l_indx = indx + n * np.arange(len(indx))
    out = as_strided(A, (len(A) - num_elem + 1, num_elem), (s0, s0))[l_indx]
    A.shape = m, n
    return out


def cal_data_simple_rasar(train_distance_matrix,
                          test_distance_matrix,
                          y_train,
                          n_neighbors=1,
                          encoding="binary"):
    df_rasar_train = pd.DataFrame()
    df_rasar_test = pd.DataFrame()
    if encoding == "binary":
        label = [0, 1]
    elif encoding == "multiclass":
        label = [1, 2, 3, 4, 5]

    for i in label:
        dist_matr_train_train_i = train_distance_matrix.iloc[:, y_train == i]
        values = dist_matr_train_train_i.values
        values.sort(axis=1)
        indx = (dist_matr_train_train_i == 0).astype(int).sum(axis=1).values
        disti = pd.DataFrame(take_per_row_strided(values, indx, n_neighbors))
        df_rasar_train = pd.concat([disti, df_rasar_train], axis=1)

        dist_matr_test_test_i = test_distance_matrix.iloc[:, y_train == i]
        values = dist_matr_test_test_i.values
        values.sort(axis=1)
        indx = (dist_matr_test_test_i == 0).astype(int).sum(axis=1).values
        disti = pd.DataFrame(take_per_row_strided(values, indx, n_neighbors))
        df_rasar_test = pd.concat([disti, df_rasar_test], axis=1)
    df_rasar_train.columns = range(df_rasar_train.shape[1])
    df_rasar_train.columns = [str(x) for x in df_rasar_train.columns]
    df_rasar_test.columns = range(df_rasar_test.shape[1])
    df_rasar_test.columns = [str(x) for x in df_rasar_test.columns]
    return df_rasar_train, df_rasar_test


def RASAR_simple_fish(distance_matrix, Y, X, Y_fish, X_fish, best_alpha_h,
                      best_alpha_p):
    print("Start CV...", ctime())
    accs = []
    rmse = []
    sens = []
    precs = []
    specs = []

    dist_matr_test = dist_matrix(X_fish, X, non_categorical, categorical,
                                 best_alpha_h, best_alpha_p)
    y_train = Y
    y_test = Y_fish

    rasar_train, rasar_test = cal_data_simple_rasar(
        distance_matrix,
        dist_matr_test,
        X,
        X_fish,
        Y,
        Y_fish,
        np.concatenate([Y, Y_fish]),
    )

    lrc = LogisticRegression(random_state=0, solver="saga")
    lrc.fit(rasar_train[["dist_neigh0", "dist_neigh1"]], y_train)
    y_pred = lrc.predict(rasar_test[["dist_neigh0", "dist_neigh1"]])

    accs.append(accuracy_score(Y_fish, y_pred))
    rmse.append(sqrt(mean_squared_error(Y_fish, y_pred)))
    sens.append(recall_score(Y_fish, y_pred, average="weighted"))
    precs.append(precision_score(Y_fish, y_pred, average="weighted"))
    # x_test = db_raw
    # x_test['y_predicted'] = y_pred
    # x_test['y_test'] = y_test
    # result_pred = x_test

    print("...END Simple RASAR", ctime())

    return accs, rmse, sens, precs


def RASAR_simple(distance_matrix,
                 Y,
                 X,
                 db_invitro_matrix,
                 n_neighbors=2,
                 invitro="False",
                 invitro_form="number",
                 db_invitro="overlap",
                 encoding="binary",
                 model="RF"):

    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    kf = KFold(n_splits=5, shuffle=True, random_state=5645)
    accs = []
    rmse = []
    sens = []
    precs = []
    f1 = []
    f1_macro = []
    result_pred = pd.DataFrame()
    for train_index, test_index in (kf.split(distance_matrix)):
        y_train = Y[train_index]
        y_test = Y[test_index]
        train_rf, test_rf = cal_data_simple_rasar(
            distance_matrix.iloc[train_index, train_index],
            distance_matrix.iloc[test_index,
                                 train_index], y_train, n_neighbors, encoding)
        if encoding == "binary":
            if model == "LR":
                lrc = LogisticRegression(
                    random_state=0,
                    fit_intercept=False,
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=1,
                )
                lrc = LogisticRegression(n_jobs=-1)
            elif model == "RF":
                lrc = RandomForestClassifier(random_state=0,
                                             n_estimators=200,
                                             class_weight="balanced")
                # lrc = RandomForestClassifier(random_state=0, n_estimators=200)
                # lrc = RandomForestClassifier(max_features=None,
                #                              random_state=0,
                #                              n_estimators=200,
                #                              min_samples_split=200,
                #                              min_samples_leaf=100,
                #                              max_depth=100)
        elif encoding == "multiclass":
            lrc = RandomForestClassifier(random_state=0, n_estimators=200)

        if invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()
        if str(db_invitro) == "overlap":
            if (invitro != "False") & (invitro_form == "number"):
                train_rf["invitro_conc"] = X.ec50[train_index].reset_index(
                    drop=True)
                test_rf["invitro_conc"] = X.ec50[test_index].reset_index(
                    drop=True)

            elif (invitro != "False") & (invitro_form == "label"):
                train_rf["invitro_label"] = X.invitro_label[
                    train_index].reset_index(drop=True)
                test_rf["invitro_label"] = X.invitro_label[
                    test_index].reset_index(drop=True)

            elif (invitro != "False") & (invitro_form == "both"):
                train_rf["ec50"] = X.ec50[train_index].reset_index(drop=True)
                test_rf["ec50"] = X.ec50[test_index].reset_index(drop=True)
                train_rf["invitro_label"] = X.invitro_label[
                    train_index].reset_index(drop=True)
                test_rf["invitro_label"] = X.invitro_label[
                    test_index].reset_index(drop=True)
        else:
            if (invitro != "False") & (invitro_form == "number"):
                dist = np.array(
                    db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                train_rf["invitro_conc"] = np.array(conc)
                train_rf["invitro_dist"] = dist

                dist = np.array(
                    db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                test_rf["invitro_conc"] = np.array(conc)
                test_rf["invitro_dist"] = dist

            elif (invitro != "False") & (invitro_form == "label"):
                dist = np.array(
                    db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                label = db_invitro["invitro_label"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                train_rf["invitro_label"] = np.array(label)
                train_rf["invitro_dist"] = dist

                dist = np.array(
                    db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                label = db_invitro["invitro_label"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                test_rf["invitro_label"] = np.array(label)
                test_rf["invitro_dist"] = dist

            elif (invitro != "False") & (invitro_form == "both"):

                dist = np.array(
                    db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                label = db_invitro["invitro_label"][ls]
                # dist = db_invitro_matrix.lookup(
                #     pd.Series(ls).index,
                #     pd.Series(ls).values)
                train_rf["invitro_conc"] = np.array(conc)
                train_rf["invitro_label"] = np.array(label)
                train_rf["invitro_dist"] = dist

                dist = np.array(
                    db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                label = db_invitro["invitro_label"][ls]
                # dist2 = db_invitro_matrix.iloc[test_index, :].lookup(
                #     pd.Series(ls).index,
                #     pd.Series(ls).values)
                # print(dist == dist2)
                test_rf["invitro_conc"] = np.array(conc)
                test_rf["invitro_label"] = np.array(label)
                test_rf["invitro_dist"] = dist

        lrc.fit(train_rf, y_train)
        y_pred = lrc.predict(test_rf)

        del y_train
        if encoding == "binary":
            label = [0, 1]
        elif encoding == "multiclass":
            label = [1, 2, 3, 4, 5]
        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred, average="weighted"))
        precs.append(precision_score(y_test, y_pred, average="weighted"))
        f1.append(f1_score(y_test, y_pred, average="weighted", labels=label))
    # print(str(lrc))
    # print('...END Simple RASAR', ctime())
    results = {}
    results["avg_accs"] = np.mean(accs)
    results["se_accs"] = sem(accs)
    results["avg_rmse"] = np.mean(rmse)
    results["se_rmse"] = sem(rmse)
    results["avg_sens"] = np.mean(sens)
    results["se_sens"] = sem(sens)
    results["avg_precs"] = np.mean(precs)
    results["se_precs"] = sem(precs)
    results["avg_f1"] = np.mean(f1)
    results["se_f1"] = sem(f1)
    results["model"] = str(lrc)

    return results


# --------------------------------------------------------------------- datafusion---------------------------------------------------------


def find_similar_exp(db_mortality, db_datafusion_rasar, db_endpoint, effect,
                     encoding):
    try:
        temp = pd.merge(db_mortality.reset_index(drop=True),
                        db_endpoint[db_endpoint.effect == effect],
                        on=comparing,
                        how='left')
        temp = temp[temp.atom_number_y.notnull()]
        temp = pd.DataFrame(
            temp.groupby(comparing)['conc1_mean'].agg(
                pd.Series.mode)).reset_index()
        temp = pd.merge(db_mortality.reset_index(drop=True),
                        temp,
                        on=comparing,
                        how='left')
        if encoding == "binary":
            temp['conc1_mean'] = np.where(
                temp["conc1_mean"] == 0, -1,
                (np.where(temp["conc1_mean"] == 1, 1, 0)))
        elif encoding == "multiclass":
            temp["conc1_mean"] = temp["conc1_mean"].fillna("Unknown")
    except:
        if encoding == "binary":
            temp = pd.DataFrame(0,
                                index=np.arange(len(db_datafusion_rasar)),
                                columns=["conc1_mean"])
        elif encoding == "multiclass":
            temp = pd.DataFrame("Unknown",
                                index=np.arange(len(db_datafusion_rasar)),
                                columns=["conc1_mean"])

    return temp


def find_datafusion_neighbor(db_datafusion_rasar_train,
                             db_datafusion_rasar_test,
                             db_datafusion,
                             db_datafusion_matrix,
                             train_index,
                             test_index,
                             effect,
                             endpoint,
                             encoding="binary"):
    if encoding == "binary":
        label = [0, 1]
    elif encoding == "multiclass":
        label = [1, 2, 3, 4, 5]

    for a in label:
        db_end_eff = db_datafusion.loc[(db_datafusion.effect == effect)
                                       & (db_datafusion.conc1_mean == a)
                                       & (db_datafusion.endpoint == endpoint)]
        if len(db_end_eff) == 0:
            continue
        else:
            train_test_matrix = db_datafusion_matrix.iloc[
                train_index,
                np.array((db_datafusion.effect == effect)
                         & (db_datafusion.conc1_mean == a)
                         & (db_datafusion.endpoint == endpoint)).nonzero()[0]]
            train_test_matrix = train_test_matrix.reset_index(drop=True)
            train_test_matrix.columns = range(train_test_matrix.shape[1])
            col_name = endpoint + "_" + effect + "_" + str(a)

            # values = train_test_matrix.values
            # values.sort(axis=1)
            # indx = (train_test_matrix == 0).astype(int).sum(axis=1).values
            # temp = take_per_row_strided(values, indx, 1)
            # db_datafusion_rasar_train[col_name] = pd.Series(temp.reshape(-1))

            db_datafusion_rasar_train[col_name] = np.array(
                train_test_matrix.min(axis=1))

            test_test_matrix = db_datafusion_matrix.iloc[
                test_index,
                np.array((db_datafusion.effect == effect)
                         & (db_datafusion.conc1_mean == a)
                         & (db_datafusion.endpoint == endpoint)).nonzero()[0]]
            test_test_matrix = test_test_matrix.reset_index(drop=True)
            indx = (test_test_matrix == 0).astype(int).sum(axis=1).values
            test_test_matrix.columns = range(test_test_matrix.shape[1])

            # values = test_test_matrix.values
            # values.sort(axis=1)
            # indx = (test_test_matrix == 0).astype(int).sum(axis=1).values
            # temp = take_per_row_strided(values, indx, 1)
            # db_datafusion_rasar_test[col_name] = pd.Series(temp.reshape(-1))

            db_datafusion_rasar_test[col_name] = np.array(
                test_test_matrix.min(axis=1))

    return db_datafusion_rasar_train, db_datafusion_rasar_test


def cal_data_datafusion_rasar(train_index, test_index, db_mortality_train,
                              db_mortality_test, db_datafusion,
                              db_datafusion_matrix, train_label, train_effect,
                              encoding):
    db_datafusion_rasar_train = pd.DataFrame()
    db_datafusion_rasar_test = pd.DataFrame()

    for endpoint in db_datafusion.endpoint.unique():

        db_endpoint = db_datafusion[db_datafusion.endpoint == endpoint]
        for effect in db_endpoint.effect.unique():
            if (str(effect) == train_effect) & (str(endpoint) in train_label):
                continue
            else:
                db_datafusion_rasar_train, db_datafusion_rasar_test = find_datafusion_neighbor(
                    db_datafusion_rasar_train, db_datafusion_rasar_test,
                    db_datafusion, db_datafusion_matrix, train_index,
                    test_index, effect, endpoint, encoding)

                # FINDING LABELS
                col_name = endpoint + "_" + effect + "_label"
                temp2 = find_similar_exp(db_mortality_train,
                                         db_datafusion_rasar_train,
                                         db_endpoint, effect, encoding)
                try:
                    temp = pd.merge(db_mortality_train.reset_index(drop=True),
                                    db_endpoint[db_endpoint.effect == effect],
                                    on=comparing,
                                    how='left')
                    temp = temp[temp.atom_number_y.notnull()]
                    temp = pd.DataFrame(
                        temp.groupby(comparing)['conc1_mean'].agg(
                            pd.Series.mode)).reset_index()
                    temp = pd.merge(db_mortality_train.reset_index(drop=True),
                                    temp,
                                    on=comparing,
                                    how='left')
                    if encoding == "binary":
                        temp['conc1_mean'] = np.where(
                            temp["conc1_mean"] == 0, -1,
                            (np.where(temp["conc1_mean"] == 1, 1, 0)))
                    elif encoding == "multiclass":
                        temp["conc1_mean"] = temp["conc1_mean"].fillna(
                            "Unknown")
                except:
                    if encoding == "binary":
                        temp = pd.DataFrame(
                            0,
                            index=np.arange(len(db_datafusion_rasar_train)),
                            columns=["conc1_mean"])
                    elif encoding == "multiclass":
                        temp = pd.DataFrame(
                            "Unknown",
                            index=np.arange(len(db_datafusion_rasar_train)),
                            columns=["conc1_mean"])
                db_datafusion_rasar_train[col_name] = temp2["conc1_mean"]
                # print(np.all(temp == temp2))
                # db_datafusion_rasar_test[
                #     endpoint + "_" + effect +
                #     "_label"] = db_mortality_test.apply(
                #         lambda x: find_similar_exp(
                #             x, db_endpoint[db_endpoint.effect == effect],
                #             comparing),
                #         axis=1,
                #     ).reset_index(drop=True)
                temp = find_similar_exp(db_mortality_test,
                                        db_datafusion_rasar_test, db_endpoint,
                                        effect, encoding)

                db_datafusion_rasar_test[col_name] = temp["conc1_mean"]

    return db_datafusion_rasar_train, db_datafusion_rasar_test


def cv_datafusion_rasar(db_datafusion_matrix,
                        distance_matrix,
                        db_invitro_matrix,
                        X,
                        y,
                        db_datafusion,
                        db_invitro,
                        train_label,
                        train_effect,
                        params={},
                        n_neighbors=2,
                        invitro=False,
                        invitro_form="both",
                        encoding="binary"):

    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    accs = []
    rmse = []
    sens = []
    precs = []
    specs = []
    f1 = []
    result_pred = pd.DataFrame()
    for train_index, test_index in kf.split(distance_matrix):

        dist_matr_train = distance_matrix.iloc[train_index, train_index]
        dist_matr_test = distance_matrix.iloc[test_index, train_index]
        y_train = y[train_index]
        y_test = y[test_index]

        simple_rasar_train, simple_rasar_test = cal_data_simple_rasar(
            dist_matr_train, dist_matr_test, y_train, n_neighbors, encoding)
        datafusion_rasar_train, datafusion_rasar_test = cal_data_datafusion_rasar(
            train_index, test_index, X.iloc[train_index], X.iloc[test_index],
            db_datafusion, db_datafusion_matrix, train_label, train_effect,
            encoding)

        train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train],
                             axis=1)
        test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)
        del (
            dist_matr_train,
            dist_matr_test,
            simple_rasar_train,
            simple_rasar_test,
            datafusion_rasar_train,
            datafusion_rasar_test,
        )
        if invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()
        if str(db_invitro) == "overlap":
            if (invitro != "False") & (invitro_form == "number"):
                train_rf["invitro_conc"] = X.ec50[train_index].reset_index(
                    drop=True)
                test_rf["invitro_conc"] = X.ec50[test_index].reset_index(
                    drop=True)

            elif (invitro != "False") & (invitro_form == "label"):
                train_rf["invitro_label"] = X.invitro_label[
                    train_index].reset_index(drop=True)
                test_rf["invitro_label"] = X.invitro_label[
                    test_index].reset_index(drop=True)

            elif (invitro != "False") & (invitro_form == "both"):
                train_rf["ec50"] = X.ec50[train_index].reset_index(drop=True)
                test_rf["ec50"] = X.ec50[test_index].reset_index(drop=True)
                train_rf["invitro_label"] = X.invitro_label[
                    train_index].reset_index(drop=True)
                test_rf["invitro_label"] = X.invitro_label[
                    test_index].reset_index(drop=True)
        else:
            if (invitro != "False") & (invitro_form == "number"):
                dist = np.array(
                    db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                train_rf["invitro_conc"] = np.array(conc)
                train_rf["invitro_dist"] = dist

                dist = np.array(
                    db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                test_rf["invitro_conc"] = np.array(conc)
                test_rf["invitro_dist"] = dist

            elif (invitro != "False") & (invitro_form == "label"):
                dist = np.array(
                    db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                label = db_invitro["invitro_label"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                train_rf["invitro_label"] = np.array(label)
                train_rf["invitro_dist"] = dist

                dist = np.array(
                    db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                label = db_invitro["invitro_label"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                test_rf["invitro_label"] = np.array(label)
                test_rf["invitro_dist"] = dist

            elif (invitro != "False") & (invitro_form == "both"):

                dist = np.array(
                    db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                label = db_invitro["invitro_label"][ls]
                train_rf["invitro_conc"] = np.array(conc)
                train_rf["invitro_label"] = np.array(label)
                train_rf["invitro_dist"] = dist

                dist = np.array(
                    db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                label = db_invitro["invitro_label"][ls]
                test_rf["invitro_conc"] = np.array(conc)
                test_rf["invitro_label"] = np.array(label)
                test_rf["invitro_dist"] = dist

        # print(train_rf.columns)
        # print('done... model...', ctime(), end='')

        if params == {}:
            rfc = RandomForestClassifier(random_state=0,
                                         n_estimators=200,
                                         class_weight="balanced")

            rfc = RandomForestClassifier(random_state=0, n_estimators=200)
            # rfc = RandomForestClassifier(
            #     max_features=None,
            #     random_state=0,
            #     n_estimators=200,
            #     min_samples_split=200,
            #     min_samples_leaf=100,
            #     max_depth=100,
            # )
        else:
            rfc = RandomForestClassifier(n_jobs=-1)
            for k, v in params.items():
                setattr(rfc, k, v)

        rfc.fit(train_rf, y_train)
        y_pred = rfc.predict(test_rf)
        if encoding == "binary":
            label = [0, 1]
        elif encoding == "multiclass":
            label = [1, 2, 3, 4, 5]

        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred, average="weighted"))
        precs.append(precision_score(y_test, y_pred, average="weighted"))
        f1.append(f1_score(y_test, y_pred, average="weighted", labels=label))

        del train_rf, test_rf

    results = {}
    results["avg_accs"] = np.mean(accs)
    results["se_accs"] = sem(accs)
    results["avg_rmse"] = np.mean(rmse)
    results["se_rmse"] = sem(rmse)
    results["avg_sens"] = np.mean(sens)
    results["se_sens"] = sem(sens)
    results["avg_precs"] = np.mean(precs)
    results["se_precs"] = sem(precs)
    results["avg_f1"] = np.mean(f1)
    results["se_f1"] = sem(f1)
    return results


# ---------------------------------------------------------------------datafusion_multiclass--------------------------------------------------


def cal_data_datafusion_rasar_multiclass(
    db_mortality_train,
    db_mortality_test,
    db_datafusion,
    train_label,
    train_effect,
    ah,
    ap,
):

    db_datafusion_rasar_train = pd.DataFrame()
    db_datafusion_rasar_test = pd.DataFrame()

    for endpoint in db_datafusion.endpoint.unique():

        db_endpoint = db_datafusion[db_datafusion.endpoint == endpoint]

        for effect in db_endpoint.effect.unique():
            if (str(effect) == train_effect) & (str(endpoint) in train_label):
                continue
            else:
                for i in range(1, 6):
                    (
                        db_datafusion_rasar_train,
                        db_datafusion_rasar_test,
                    ) = find_datafusion_neighbor(
                        db_datafusion_rasar_train,
                        db_datafusion_rasar_test,
                        db_endpoint,
                        db_mortality_train,
                        db_mortality_test,
                        effect,
                        endpoint,
                        ah,
                        ap,
                        [i],
                    )

                db_datafusion_rasar_train[
                    endpoint + "_" + effect +
                    "_label"] = db_mortality_train.apply(
                        lambda x: find_similar_exp(
                            x,
                            db_endpoint[db_endpoint.effect == effect],
                            comparing,
                            "multiclass",
                        ),
                        axis=1,
                    ).reset_index(drop=True)
                db_datafusion_rasar_test[
                    endpoint + "_" + effect +
                    "_label"] = db_mortality_test.apply(
                        lambda x: find_similar_exp(
                            x,
                            db_endpoint[db_endpoint.effect == effect],
                            comparing,
                            "multiclass",
                        ),
                        axis=1,
                    ).reset_index(drop=True)

    return db_datafusion_rasar_train, db_datafusion_rasar_test


def cv_datafusion_rasar_multiclass(db_datafusion_matrix,
                                   distance_matrix,
                                   db_invitro_matrix,
                                   X,
                                   y,
                                   db_datafusion,
                                   db_invitro,
                                   train_label,
                                   train_effect,
                                   final_model=False,
                                   n_neighbors=2,
                                   invitro=False,
                                   invitro_form="both",
                                   encoding="multiclass"):
    # h2o.init(max_mem_size="20g")
    h2o.init()
    h2o.no_progress()

    # print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    kf = KFold(n_splits=3, shuffle=True, random_state=12)
    accs = []
    rmse = []
    sens = []
    precs = []
    f1 = []
    result_pred = pd.DataFrame()
    for train_index, test_index in kf.split(distance_matrix):

        dist_matr_train = distance_matrix.iloc[train_index, train_index]
        dist_matr_test = distance_matrix.iloc[test_index, train_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # print('Train and test...')
        simple_rasar_train, simple_rasar_test = cal_data_simple_rasar(
            dist_matr_train, dist_matr_test, y_train, n_neighbors, encoding)
        # print('Finish simple dataset')
        # print('Start datafusion dataset.')
        datafusion_rasar_train, datafusion_rasar_test = cal_data_datafusion_rasar(
            train_index, test_index, X.iloc[train_index], X.iloc[test_index],
            db_datafusion, db_datafusion_matrix, train_label, train_effect,
            encoding)
        del dist_matr_train, dist_matr_test

        train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train],
                             axis=1)
        test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)

        if str(db_invitro) == "overlap":
            if (invitro != "False") & (invitro_form == "number"):
                train_rf["invitro_conc"] = X.ec50[train_index].reset_index(
                    drop=True)
                test_rf["invitro_conc"] = X.ec50[test_index].reset_index(
                    drop=True)

            elif (invitro != "False") & (invitro_form == "label"):
                train_rf["invitro_label"] = X.invitro_label[
                    train_index].reset_index(drop=True)
                test_rf["invitro_label"] = X.invitro_label[
                    test_index].reset_index(drop=True)
            elif (invitro != "False") & (invitro_form == "both"):
                train_rf["ec50"] = X.ec50[train_index].reset_index(drop=True)
                test_rf["ec50"] = X.ec50[test_index].reset_index(drop=True)
                train_rf["invitro_label"] = X.invitro_label[
                    train_index].reset_index(drop=True)
                test_rf["invitro_label"] = X.invitro_label[
                    test_index].reset_index(drop=True)
        else:
            if invitro == "own":
                train_rf = pd.DataFrame()
                test_rf = pd.DataFrame()

            if (invitro != "False") & (invitro_form == "number"):
                dist = np.array(
                    db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                train_rf["invitro_conc"] = np.array(conc)
                train_rf["invitro_dist"] = dist

                dist = np.array(
                    db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                test_rf["invitro_conc"] = np.array(conc)
                test_rf["invitro_dist"] = dist

            elif (invitro != "False") & (invitro_form == "label"):
                dist = np.array(
                    db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                label = db_invitro["invitro_label"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                train_rf["invitro_label"] = np.array(label)
                train_rf["invitro_dist"] = dist

                dist = np.array(
                    db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                label = db_invitro["invitro_label"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index,
                    pd.Series(ls).values)
                test_rf["invitro_label"] = np.array(label)
                test_rf["invitro_dist"] = dist

            elif (invitro != "False") & (invitro_form == "both"):

                dist = np.array(
                    db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                label = db_invitro["invitro_label"][ls]
                train_rf["invitro_conc"] = np.array(conc)
                train_rf["invitro_label"] = np.array(label)
                train_rf["invitro_dist"] = dist

                dist = np.array(
                    db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(
                    db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                conc = db_invitro["ec50"][ls]
                label = db_invitro["invitro_label"][ls]
                test_rf["invitro_conc"] = np.array(conc)
                test_rf["invitro_label"] = np.array(label)
                test_rf["invitro_dist"] = dist

        train_rf.loc[:, "target"] = y_train
        test_rf.loc[:, "target"] = y_test

        train_rf_h2o = h2o.H2OFrame(train_rf)
        test_rf_h2o = h2o.H2OFrame(test_rf)

        for col in train_rf.columns:
            if "label" in col:
                train_rf_h2o[col] = train_rf_h2o[col].asfactor()
                test_rf_h2o[col] = test_rf_h2o[col].asfactor()

        train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
        test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

        # print('done...', ctime(), 'model...')

        if not final_model:
            rfc = H2ORandomForestEstimator(
                categorical_encoding="one_hot_explicit", seed=123)
        else:
            ### TO DO TO DO TO DO TO DO ### ### ### ### ### ### ### ### ### ###  TO DO  TO DO  TO DO TO DO TO DO TO DO TO DO
            rfc = (
                H2ORandomForestEstimator()
            )  ### TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO
            ### TO DO TO DO TO DO TO DO ### ### ### ### ### ### ### ### ### ### TO DO TO DO TO DO TO DO TO DO TO DO

        rfc.train(y="target", training_frame=train_rf_h2o)
        y_pred = rfc.predict(test_rf_h2o).as_data_frame()["predict"]

        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred, average="weighted"))
        precs.append(precision_score(y_test, y_pred, average="weighted"))
        f1.append(f1_score(y_test, y_pred, average="weighted"))
        del y_pred, train_rf, test_rf, train_rf_h2o, test_rf_h2o
    # h2o.cluster().shutdown()
    results = {}
    results["avg_accs"] = np.mean(accs)
    results["se_accs"] = sem(accs)
    results["avg_rmse"] = np.mean(rmse)
    results["se_rmse"] = sem(rmse)
    results["avg_sens"] = np.mean(sens)
    results["se_sens"] = sem(sens)
    results["avg_precs"] = np.mean(precs)
    results["se_precs"] = sem(precs)
    results["avg_f1"] = np.mean(f1)
    results["se_f1"] = sem(f1)
    results["fold"] = kf
    results["model"] = rfc
    return results


# ---------------------------------------------------------------------parallel running----------------------------------------------------


# def model(dist_matr_train_0_0,dist_matr_train_train_0,X_train,y_train,
#     dist_matr_train_1_1,dist_matr_train_train_1,dist_matr_test_test_0,
#     dist_matr_test_test_1,y_test,n_neighbors,label,ah,ap,num):
def model(
    dist_matr_train,
    dist_matr_test,
    X_train,
    y_train,
    y_test,
    n_neighbors,
    label,
    ah,
    ap,
    num,
):

    dist_matr_train_0_0 = dist_matr_train.iloc[y_train == 0, y_train == 0]
    dist_matr_train_1_1 = dist_matr_train.iloc[y_train == 1, y_train == 1]

    # To find neighbors for all train experiments --> df_rasar_train
    dist_matr_train_train_0 = dist_matr_train.iloc[:, y_train == 0]
    dist_matr_train_train_1 = dist_matr_train.iloc[:, y_train == 1]

    dist_matr_test_test_0 = dist_matr_test.iloc[:, y_train == 0]
    dist_matr_test_test_1 = dist_matr_test.iloc[:, y_train == 1]

    idx_neigh0, dist0 = find_neighbor(
        dist_matr_train_0_0,
        dist_matr_train_train_0,
        X_train,
        y_train,
        0,
        "binary",
        "train",
        n_neighbors + 1,
    )
    idx_neigh1, dist1 = find_neighbor(
        dist_matr_train_1_1,
        dist_matr_train_train_1,
        X_train,
        y_train,
        1,
        "binary",
        "train",
        n_neighbors + 1,
    )
    df_rasar_train = pd.concat([dist0, dist1], axis=1)
    df_rasar_train = pd.concat([dist1, dist0], axis=1)

    idx_neigh0, dist0 = find_neighbor(
        dist_matr_train_0_0,
        dist_matr_test_test_0,
        X_train,
        y_train,
        0,
        "binary",
        "test",
        n_neighbors + 1,
    )
    idx_neigh1, dist1 = find_neighbor(
        dist_matr_train_1_1,
        dist_matr_test_test_1,
        X_train,
        y_train,
        1,
        "binary",
        "test",
        n_neighbors + 1,
    )
    df_rasar_test = pd.concat([dist1, dist0], axis=1)
    if label == "LR":
        lrc = LogisticRegression(
            random_state=0,
            fit_intercept=False,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=1,
        )
    elif label == "RF":
        # lrc = RandomForestClassifier(random_state=0,n_estimators = 200)
        lrc = RandomForestClassifier(
            max_features=None,
            random_state=0,
            n_estimators=200,
            min_samples_split=200,
            min_samples_leaf=100,
            max_depth=100,
        )

    lrc.fit(df_rasar_train, y_train)
    y_pred = lrc.predict(df_rasar_test)

    results = {}
    results["accs"] = accuracy_score(y_test, y_pred)
    results["f1"] = f1_score(y_test, y_pred, average="weighted")
    results["neighbors"] = n_neighbors
    results["ah"] = ah
    results["ap"] = ap
    results["fold"] = num
    del (
        idx_neigh0,
        dist0,
        idx_neigh1,
        dist1,
        df_rasar_test,
        df_rasar_train,
        dist_matr_train_0_0,
        dist_matr_test_test_0,
        X_train,
        y_train,
        dist_matr_train_train_0,
        dist_matr_train_1_1,
        dist_matr_train_train_1,
        dist_matr_test_test_1,
    )

    return results


def model_mul(dist_matr_train, dist_matr_test, X_train, y_train, y_test,
              n_neighbors, ah, ap, num):

    df_rasar_train = pd.DataFrame()
    df_rasar_test = pd.DataFrame()

    for i in range(1, 6):
        matr_train_i = dist_matr_train.loc[y_train == i, y_train == i]

        # to find neighbors to train data
        matr_train_train_i = dist_matr_train.loc[:, y_train == i]

        # to find neighbors to test data
        matr_test_train_i = dist_matr_test.loc[:, y_train == i]

        # neighbors+1 to aviod the situation that finding itself as nearest neighbor
        idx_neigh_i, dist_neigh_i = find_neighbor(
            matr_train_i,
            matr_train_train_i,
            X_train,
            y_train,
            i,
            "multiclass",
            "train",
            n_neighbors + 1,
        )
        df_rasar_train = pd.concat([df_rasar_train, dist_neigh_i], axis=1)

        idx_neigh_i, dist_neigh_i = find_neighbor(
            matr_train_i,
            matr_test_train_i,
            X_train,
            y_train,
            i,
            "multiclass",
            "test",
            n_neighbors + 1,
        )
        df_rasar_test = pd.concat([df_rasar_test, dist_neigh_i], axis=1)
    lrc = RandomForestClassifier(random_state=0, n_estimators=200)
    lrc.fit(df_rasar_train, y_train)
    y_pred = lrc.predict(df_rasar_test)

    results = {}
    results["accs"] = accuracy_score(y_test, y_pred)
    results["f1"] = f1_score(y_test,
                             y_pred,
                             average="weighted",
                             labels=[1, 2, 3, 4, 5])
    results["neighbors"] = n_neighbors
    results["ah"] = ah
    results["ap"] = ap
    results["fold"] = num
    return results


# -------------------------------old version-------------------
# def load_invitro(DATA_PATH, encoding, encoding_value=1,seed=42):
#     db = pd.read_csv(DATA_PATH).drop(['Unnamed: 0','ec50_ci_lower','ec50_ci_upper','ec10','ec10_ci_lower','ec10_ci_upper'],axis =1)
#     for nc in non_categorical_invitro:
#         minmax = MinMaxScaler()
#         minmax.fit(db[[nc]])
#         db[[nc]] = minmax.transform(db[[nc]])

#     # Ordinal Encoding for categorical variables
#     encoder = OrdinalEncoder(dtype = int)
#     encoder.fit(db[categorical_invitro])
#     db[categorical_invitro] = encoder.transform(db[categorical_invitro])+1

#     # Encoding for target variable: binary and multiclass
#     if encoding == 'binary':
#         if encoding_value == 'own':
#             db['ec50'] = np.where(db['ec50'].values > np.median(db['ec50']), 1, 0)
#         else:
#             db['ec50'] = np.where(db['ec50'].values > 1, 1, 0)

#     elif encoding == 'multiclass':
#         db['ec50'] = multiclass_encoding(db['ec50'].copy(),db['ec50'].quantile([.2,.4,.6,.8]).values)
#     X = db.drop(columns = 'ec50')
#     Y = db['ec50'].values
#     return X,Y

# def right_neighbor(neighbors, X_train, y_train, y_check):
#     # IDX Neighbors
#     if (neighbors[1]).shape[1] == 1:
#         idx_neigh = pd.DataFrame(neighbors[1])[0].apply(lambda x: X_train.iloc[y_train==y_check].iloc[x].name)
#         distance = neighbors[0].ravel()

#     else:
#         idx_neigh_0 = pd.DataFrame(neighbors[1])[0].apply(lambda x: X_train.iloc[y_train==y_check].iloc[x].name)
#         idx_neigh_1 = pd.DataFrame(neighbors[1])[1].apply(lambda x: X_train.iloc[y_train==y_check].iloc[x].name)

#         idx_neigh = idx_neigh_0.copy()

#         # if the nearest point index equals itself, then replacing it with the second nearst point
#         idx_neigh[X_train.index == idx_neigh_0] = idx_neigh_1[X_train.index == idx_neigh_0].values

#         # Distance from the Nearest Neighbor that is NOT itself
#         dist_0 = pd.DataFrame(neighbors[0])[0]
#         dist_1 = pd.DataFrame(neighbors[0])[1]

#         distance = dist_0.copy()
#         distance[X_train.index == idx_neigh_0] = dist_1[X_train.index == idx_neigh_0].values

#     return idx_neigh,distance

# def find_neighbor(square_matr,matr_all_label,X_train,y_train,label, encoding = 'binary',exp= 'train'):
#     n_neighbors = 2
#     leaf_size = 40
#     if exp == 'test':
#         n_neighbors = 1
#     if encoding == 'multiclass':
#         leaf_size = 30

#     knn = KNeighborsClassifier(metric = 'precomputed',leaf_size = leaf_size, n_neighbors = n_neighbors)
#     if len(y_train[y_train == label]) == 1:
#         idx_neigh,dist = np.array(square_matr.index * matr_all_label.shape[0]),np.array([0]* matr_all_label.shape[0])
#     else:
#         knn.fit(square_matr, y_train[y_train == label])
#         neigh = knn.kneighbors(matr_all_label, return_distance = True)
#         idx_neigh, dist = right_neighbor(neigh, X_train, y_train, label)

#         del knn, neigh
#     return idx_neigh, dist

# def RASAR_simple(
#     distance_matrix,
#     Y,
#     X,
#     best_leaf,
#     n_neighbors=2,
#     model="RF",
#     invitro="False",
#     invitro_form="both",
# ):
#     # print('Start CV...', ctime())
#     kf = KFold(n_splits=5, shuffle=True, random_state=10)
#     accs = []
#     rmse = []
#     sens = []
#     precs = []
#     specs = []
#     f1 = []
#     result_pred = pd.DataFrame()
#     for train_index, test_index in kf.split(distance_matrix):

#         dist_matr_train = distance_matrix.iloc[train_index, train_index]
#         dist_matr_test = distance_matrix.iloc[test_index, train_index]
#         y_train = Y[train_index]
#         y_test = Y[test_index]

#         rasar_train, rasar_test = cal_data_simple_rasar(
#             dist_matr_train,
#             dist_matr_test,
#             X.iloc[train_index],
#             X.iloc[test_index],
#             y_train,
#             y_test,
#             Y,
#             best_leaf,
#             n_neighbors,
#         )
#         if model == "LR":
#             lrc = LogisticRegression(
#                 random_state=0,
#                 fit_intercept=False,
#                 solver="saga",
#                 penalty="elasticnet",
#                 l1_ratio=1,
#             )
#         elif model == "RF":
#             lrc = RandomForestClassifier(random_state=0,
#                                          n_estimators=200,
#                                          class_weight="balanced")
#             # lrc = RandomForestClassifier(max_features=None,random_state=0,n_estimators = 200, class_weight='balanced')
#             # ,min_samples_split=200,min_samples_leaf=100,max_depth=100

#         if invitro == "own":
#             rasar_train = pd.DataFrame()
#             rasar_test = pd.DataFrame()

#         if (invitro != "False") & (invitro_form == "number"):
#             rasar_train["ec50"] = X.ec50[train_index].reset_index(drop=True)
#             rasar_test["ec50"] = X.ec50[test_index].reset_index(drop=True)
#         elif (invitro != "False") & (invitro_form == "label"):
#             rasar_train["invitro_label"] = X.invitro_label[
#                 train_index].reset_index(drop=True)
#             rasar_test["invitro_label"] = X.invitro_label[
#                 test_index].reset_index(drop=True)
#         elif (invitro != "False") & (invitro_form == "both"):
#             rasar_train["ec50"] = X.ec50[train_index].reset_index(drop=True)
#             rasar_test["ec50"] = X.ec50[test_index].reset_index(drop=True)
#             rasar_train["invitro_label"] = X.invitro_label[
#                 train_index].reset_index(drop=True)
#             rasar_test["invitro_label"] = X.invitro_label[
#                 test_index].reset_index(drop=True)

#         lrc.fit(rasar_train, y_train)
#         y_pred = lrc.predict(rasar_test)

#         accs.append(accuracy_score(y_test, y_pred))
#         rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
#         sens.append(recall_score(y_test, y_pred, average="weighted"))
#         precs.append(precision_score(y_test, y_pred, average="weighted"))
#         f1.append(f1_score(y_test, y_pred, average="weighted"))
#         # x_test = db_raw.iloc[test_index]
#         # x_test['y_predicted'] = y_pred
#         # x_test['y_test'] = y_test
#         # result_pred = pd.concat([result_pred,x_test])

#     # print('...END Simple RASAR', ctime())
#     # print(rasar_train.columns)
#     del rasar_train, rasar_test
#     return accs, rmse, sens, precs, f1
#     # return rasar_train,rasar_test

# def RASAR_simple_invitro(distance_matrix,
#                          Y,
#                          X,
#                          db_invitro_matrix,
#                          db_invitro,
#                          best_alpha_h,
#                          best_alpha_p,
#                          n_neighbors=2,
#                          model="RF",
#                          invitro="False",
#                          invitro_form="both",
#                          encoding="binary"):

#     kf = KFold(n_splits=5, shuffle=True, random_state=10)
#     accs = []
#     rmse = []
#     sens = []
#     precs = []
#     specs = []
#     f1 = []
#     result_pred = pd.DataFrame()
#     for train_index, test_index in kf.split(distance_matrix):

#         dist_matr_train = distance_matrix.iloc[train_index, train_index]
#         dist_matr_test = distance_matrix.iloc[test_index, train_index]
#         Y_train = Y[train_index]
#         Y_test = Y[test_index]

#         train_rf, test_rf = cal_data_simple_rasar(
#             dist_matr_train, dist_matr_test, Y_train, n_neighbors, encoding)

#         del dist_matr_train, dist_matr_test
#         if encoding == "binary":
#             if model == "LR":
#                 lrc = LogisticRegression(
#                     random_state=0,
#                     fit_intercept=False,
#                     solver="saga",
#                     penalty="elasticnet",
#                     l1_ratio=1,
#                 )
#             elif model == "RF":
#                 # lrc = RandomForestClassifier(random_state=0,
#                 #                              n_estimators=200,
#                 #                              class_weight="balanced")
#                 lrc = RandomForestClassifier(max_features=None,
#                                              random_state=0,
#                                              n_estimators=200,
#                                              min_samples_split=200,
#                                              min_samples_leaf=100,
#                                              max_depth=100)
#         elif encoding == "multiclass":
#             lrc = RandomForestClassifier(random_state=0, n_estimators=200)

#         if invitro == "own":
#             train_rf = pd.DataFrame()
#             test_rf = pd.DataFrame()

#         if (invitro != "False") & (invitro_form == "number"):
#             dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
#             ls = np.array(
#                 db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
#             conc = db_invitro["ec50"][ls]
#             train_rf["invitro_conc"] = np.array(conc)
#             train_rf["invitro_dist"] = np.array(dist)

#             dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
#             ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
#             conc = db_invitro["ec50"][ls]
#             test_rf["invitro_conc"] = np.array(conc)
#             test_rf["invitro_dist"] = np.array(dist)

#         elif (invitro != "False") & (invitro_form == "label"):
#             dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
#             ls = np.array(
#                 db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
#             label = db_invitro["invitro_label"][ls]
#             train_rf["invitro_label"] = np.array(label)
#             train_rf["invitro_dist"] = np.array(dist)

#             dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
#             ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
#             label = db_invitro["invitro_label"][ls]
#             test_rf["invitro_label"] = np.array(label)
#             test_rf["invitro_dist"] = np.array(dist)

#         elif (invitro != "False") & (invitro_form == "both"):

#             dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
#             ls = np.array(
#                 db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
#             conc = db_invitro["ec50"][ls]
#             label = db_invitro["invitro_label"][ls]
#             train_rf["invitro_label"] = np.array(label)
#             train_rf["invitro_conc"] = np.array(conc)
#             train_rf["invitro_dist"] = np.array(dist)

#             dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
#             ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
#             conc = db_invitro["ec50"][ls]
#             label = db_invitro["invitro_label"][ls]
#             test_rf["invitro_label"] = np.array(label)
#             test_rf["invitro_conc"] = np.array(conc)
#             test_rf["invitro_dist"] = np.array(dist)

#         lrc.fit(train_rf, Y_train)
#         y_pred = lrc.predict(test_rf)

#         accs.append(accuracy_score(Y_test, y_pred))
#         rmse.append(sqrt(mean_squared_error(Y_test, y_pred)))
#         sens.append(recall_score(Y_test, y_pred, average="weighted"))
#         precs.append(precision_score(Y_test, y_pred, average="weighted"))
#         f1.append(f1_score(Y_test, y_pred, average="weighted"))
#         del train_rf, test_rf
#     results = {}
#     results["avg_accs"] = np.mean(accs)
#     results["se_accs"] = sem(accs)
#     results["avg_rmse"] = np.mean(rmse)
#     results["se_rmse"] = sem(rmse)
#     results["avg_sens"] = np.mean(sens)
#     results["se_sens"] = sem(sens)
#     results["avg_precs"] = np.mean(precs)
#     results["se_precs"] = sem(precs)
#     results["avg_f1"] = np.mean(f1)
#     results["se_f1"] = sem(f1)
#     return results

# def cal_data_simple_rasar(
#     train_distance_matrix,
#     test_distance_matrix,
#     X_train,
#     X_test,
#     y_train,
#     y_test,
#     Y,
#     best_leaf=10,
#     n_neighbors=1,
# ):
#     # in order to remove situation that finding itself as the nearest points

#     ######## starting DATAFRAME ##########
#     # in order to train 1-NN
#     # find the matrix between points that are all positive or all negative.
#     dist_matr_train_0_0 = train_distance_matrix.iloc[y_train == 0,
#                                                      y_train == 0]
#     dist_matr_train_1_1 = train_distance_matrix.iloc[y_train == 1,
#                                                      y_train == 1]

#     # To find neighbors for all train experiments --> df_rasar_train
#     dist_matr_train_train_0 = train_distance_matrix.iloc[:, y_train == 0]
#     dist_matr_train_train_1 = train_distance_matrix.iloc[:, y_train == 1]

#     #################################################################### DF train RASAR ############################################################
#     # finding the nearest experiments for training experiments that is not itself
#     idx_neigh0, dist0 = find_neighbor(
#         dist_matr_train_0_0,
#         dist_matr_train_train_0,
#         X_train,
#         y_train,
#         0,
#         "binary",
#         "train",
#         n_neighbors + 1,
#     )
#     idx_neigh1, dist1 = find_neighbor(
#         dist_matr_train_1_1,
#         dist_matr_train_train_1,
#         X_train,
#         y_train,
#         1,
#         "binary",
#         "train",
#         n_neighbors + 1,
#     )

#     # neigh = KNeighborsClassifier(n_neighbors=5, metric = 'precomputed', leaf_size = best_leaf)
#     # neigh.fit(train_distance_matrix, y_train.astype('int').ravel())
#     # neighbor = neigh.kneighbors(train_distance_matrix)
#     # df_neighbor = pd.DataFrame(data=neighbor[1],columns=['index_1','index_2','index_3','index_4','index_5'])
#     # # set the label with the max occurence label of five nerighbors
#     # df_neighbor = df_neighbor.apply(lambda x: Y[x])
#     # df_neighbor['number_0'] = (df_neighbor == 0).astype(int).sum(axis=1)
#     # df_neighbor['number_1'] = (df_neighbor == 1).astype(int).sum(axis=1)

#     df_rasar_train = pd.concat([dist0, dist1], axis=1)
#     df_rasar_train = pd.concat([dist1, dist0], axis=1)
#     ####### DF test RASAR ################
#     # To find neighbors for all test experiments --> df_rasar_test
#     dist_matr_test_test_0 = test_distance_matrix.iloc[:, y_train == 0]
#     dist_matr_test_test_1 = test_distance_matrix.iloc[:, y_train == 1]

#     # finding the nearest experiments to test data

#     idx_neigh0, dist0 = find_neighbor(
#         dist_matr_train_0_0,
#         dist_matr_test_test_0,
#         X_train,
#         y_train,
#         0,
#         "binary",
#         "test",
#         n_neighbors + 1,
#     )
#     idx_neigh1, dist1 = find_neighbor(
#         dist_matr_train_1_1,
#         dist_matr_test_test_1,
#         X_train,
#         y_train,
#         1,
#         "binary",
#         "test",
#         n_neighbors + 1,
#     )

#     # finding the number of neighbors with label 0 and label 1
#     # neighbor = neigh.kneighbors(test_distance_matrix)
#     # df_neighbor = pd.DataFrame(data=neighbor[1],columns=['index_1','index_2','index_3','index_4','index_5'])
#     # df_neighbor = df_neighbor.apply(lambda x: Y[x])
#     # df_neighbor['number_0'] = (df_neighbor == 0).astype(int).sum(axis=1)
#     # df_neighbor['number_1'] = (df_neighbor == 1).astype(int).sum(axis=1)

#     df_rasar_test = pd.concat([dist0, dist1], axis=1)
#     df_rasar_test = pd.concat([dist1, dist0], axis=1)

#     return df_rasar_train, df_rasar_test

# def find_neighbor_test(
#     square_matr,
#     matr_all_label,
#     X_train,
#     y_train,
#     label,
#     encoding="binary",
#     exp="train",
#     n_neighbors=2,
# ):

#     idx_neigh = pd.DataFrame()
#     dist = pd.DataFrame()
#     knn = KNeighborsClassifier(metric="precomputed",
#                                leaf_size=30,
#                                n_neighbors=n_neighbors)

#     if len(y_train[y_train == label]) == 1:
#         idx_neigh = np.array(square_matr.index * matr_all_label.shape[0])
#         dist = pd.DataFrame(np.array([0] * matr_all_label.shape[0]))
#     else:
#         knn.fit(square_matr, y_train[y_train == label])
#         neigh = knn.kneighbors(matr_all_label, return_distance=True)

#         for i in range(n_neighbors):
#             idx_neigh["idx_neigh" + str(label) + str(i)] = pd.DataFrame(
#                 neigh[1])[i].apply(
#                     lambda x: X_train.iloc[y_train == label].iloc[x].name)
#             dist["dist_neigh" + str(label) + str(i)] = pd.DataFrame(
#                 neigh[0])[i]

#         # if the nearest point index equals itself, then replacing it with the second nearst point
#         # Distance from the Nearest Neighbor that is NOT itself
#         if exp == "train":
#             ls = X_train.index == idx_neigh["idx_neigh" + str(label) + str(0)]
#             for i in range(n_neighbors - 1):
#                 idx_neigh["idx_neigh" + str(label) +
#                           str(i)][ls] = idx_neigh["idx_neigh" + str(label) +
#                                                   str(i + 1)][ls].values
#                 dist["dist_neigh" + str(label) +
#                      str(i)][ls] = dist["dist_neigh" + str(label) +
#                                         str(i + 1)][ls].values

#         idx_neigh.drop(columns="idx_neigh" + str(label) + str(n_neighbors - 1),
#                        inplace=True)
#         dist.drop(columns="dist_neigh" + str(label) + str(n_neighbors - 1),
#                   inplace=True)

#     return idx_neigh, dist

# def find_similar_exp(exp_mortality,
#                      db_datafusion,
#                      compare_features,
#                      encoding="binary"):
#     out = db_datafusion.conc1_mean[(db_datafusion[compare_features] ==
#                                     exp_mortality[compare_features]).all(
#                                         axis=1)].values
#     if encoding == "multiclass":
#         try:
#             return Counter(out).most_common(1)[0][0]
#         except:
#             return "Unknown"
#     else:
#         try:
#             return -1 if Counter(out).most_common(1)[0][0] == 0 else 1
#         except:
#             return 0

# def find_datafusion_neighbor(
#     db_datafusion_rasar_train,
#     db_datafusion_rasar_test,
#     db_endpoint,
#     db_mortality_train,
#     db_mortality_test,
#     effect,
#     endpoint,
#     ah,
#     ap,
#     label=None,
# ):
#     if label == None:
#         label = [0, 1]

#     for a in label:
#         db_end_eff = db_endpoint[(db_endpoint.effect == effect)
#                                  & (db_endpoint.conc1_mean == a)]
#         if len(db_end_eff) == 0:
#             continue
#         else:
#             train_matrix = dist_matrix(db_end_eff, db_end_eff, non_categorical,
#                                        categorical, ah, ap)
#             knn = KNeighborsClassifier(metric="precomputed", n_neighbors=1)
#             knn.fit(pd.DataFrame(train_matrix),
#                     np.repeat(a, train_matrix.shape[0]))

#             train_test_matrix = dist_matrix(db_mortality_train, db_end_eff,
#                                             non_categorical, categorical, ah,
#                                             ap)
#             neigh = knn.kneighbors(train_test_matrix, return_distance=True)
#             db_datafusion_rasar_train[endpoint + "_" + effect + "_" +
#                                       str(a)] = neigh[0].ravel()

#             test_test_matrix = dist_matrix(db_mortality_test, db_end_eff,
#                                            non_categorical, categorical, ah,
#                                            ap)
#             neigh = knn.kneighbors(test_test_matrix, return_distance=True)
#             db_datafusion_rasar_test[endpoint + "_" + effect + "_" +
#                                      str(a)] = neigh[0].ravel()

#     return db_datafusion_rasar_train, db_datafusion_rasar_test
