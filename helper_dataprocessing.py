import pandas as pd
import numpy as np
# from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
# from rdkit.Chem import rdchem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import Crippen
# from rdkit.Chem import MolFromSmarts
# from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from urllib.request import urlopen
import pubchempy as pcp
import re
from helper_chemproperty import *


def load_raw_data(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES):
    tests = pd.read_csv(DATA_PATH_TESTS, sep='\|', engine='python')
    print('tests loaded')
    species = pd.read_csv(DATA_PATH_SPECIES, sep='\|', engine='python')
    print('species loaded')
    results = pd.read_csv(DATA_PATH_RESULTS, sep='\|', engine='python')
    print('results loaded')

    return tests, species, results


# ---------------------------------------------------------------------
def prefilter(species,
              tests,
              results,
              endpoint=None,
              label=None,
              fish_species=None,
              all_property='itself',
              effect='MOR'):
    results.loc[:, 'effect'] = results.effect.apply(
        lambda x: x.replace('/', '') if '/' in x else x)
    results.loc[:, 'effect'] = results.effect.apply(
        lambda x: x.replace('~', '') if '~' in x else x)
    results.loc[:, 'endpoint'] = results.endpoint.apply(
        lambda x: x.replace('/', '') if '/' in x else x)
    results.loc[:, 'endpoint'] = results.endpoint.apply(
        lambda x: x.replace('*', '') if '*' in x else x)
    if label == 'datafusion':
        resc = results[results.endpoint.str.contains('C')].copy()
        if all_property == 'no':
            rconc = resc.loc[~(results.effect.str.contains(effect)), :]
        elif all_property == 'itself':
            rconc = resc.loc[~((~results.endpoint.str.contains(endpoint)) &
                               (results.effect.str.contains(effect))), :]
        elif all_property == 'all':
            rconc = resc
        elif all_property == 'other':
            rconc = resc.loc[~(results.endpoint.str.contains(endpoint)
                               & results.effect.str.contains(effect)), :]
        print('There are', rconc.shape[0], 'tests.(except ' + endpoint + ')')
        test = tests.copy()
    elif label == 'simple':
        resc = results[(results.endpoint.str.contains(endpoint))]
        print('There are', resc.shape[0], 'tests.(only ' + endpoint + ')')
        rconc = resc[resc.effect.str.contains(effect)]
        print('There are', rconc.shape[0],
              'tests consideing about' + effect + '.')
        test = tests[tests.organism_lifestage != "EM"]
    else:
        rconc = results[results.endpoint.str.contains('C')].copy()
        test = tests[tests.organism_lifestage != "EM"]

    # focus on fishes
    species = species[~species.ecotox_group.isnull()]
    sp_fish = species[species.ecotox_group.str.contains('Fish')]
    # focus on rainbow trout
    if fish_species:
        sp_fish = sp_fish[~sp_fish.common_name.isnull()]
        sp_fish = sp_fish[sp_fish.common_name.str.contains(fish_species)]
    print('There are', sp_fish.shape[0], 'fish species in total.')
    # merging tests and tested fishes
    test_fish = test.merge(sp_fish, on="species_number")
    print('There are', test_fish.shape[0], 'tests on these fishes.')
    # merging experiments and their relative results
    results_prefilter = rconc.merge(test_fish, on='test_id')
    print('All merged into one dataframe. Size was', results_prefilter.shape,
          '.')
    print('The unique chemical number was',
          len(results_prefilter.test_cas.unique()), '.')
    return results_prefilter


# -------------------------------------------------------------------
def impute_conc(results_prefiltered):

    db = results_prefiltered.copy()
    db.loc[:,'conc1_unit'] = db.conc1_unit.apply(lambda x: x.replace("AI ", "") if 'AI' in x else x)
    good_conc_unit = ['ppb', 'ppm', 'ug/L', 'ng/L', 'mg/L', 'ng/ml', 'mg/dm3',\
    'umol/L','mmol/L','ug/ml','g/L','ng/ml','nmol/L','mol/L','g/dm3','ug/dm3']

    db = db[db.conc1_unit.isin(good_conc_unit)].copy()

    to_drop_conc_mean = db[(db.conc1_mean == 'NC') |
                           (db.conc1_mean == 'NR')].index
    db_filtered_mean = db.drop(to_drop_conc_mean).copy()
    db_filtered_mean.loc[:, 'conc1_mean'] = db_filtered_mean.conc1_mean.apply(
        lambda x: x.replace("*", "") if "*" in x else x).copy()

    # remove the concentration that is higher than 100000mg/L
    db_filtered_mean.loc[:, 'conc1_mean'] = db_filtered_mean.conc1_mean.apply(
        lambda x: 100000 if ">100000" in x else x).copy()
    to_drop_invalid_conc = db_filtered_mean[(
        db_filtered_mean.conc1_mean == 0)].index
    db_filtered_mean.drop(to_drop_invalid_conc, inplace=True)

    db_filtered_mean.loc[:, 'conc1_mean'] = db_filtered_mean.conc1_mean.astype(
        float).copy()

    # convert all units into mg/L
    db_filtered_mean.loc[(db_filtered_mean.conc1_unit == 'ppb') | (db_filtered_mean.conc1_unit == 'ug/L') \
     |(db_filtered_mean.conc1_unit == 'ng/ml')|(db_filtered_mean.conc1_unit == 'ug/dm3'), 'conc1_mean'] = \
    db_filtered_mean.conc1_mean[(db_filtered_mean.conc1_unit == 'ppb') | (db_filtered_mean.conc1_unit == 'ug/L')|\
      (db_filtered_mean.conc1_unit == 'ng/ml') |(db_filtered_mean.conc1_unit == 'ug/dm3') ]/(10**(3))

    db_filtered_mean.loc[db_filtered_mean.conc1_unit == 'ng/L', 'conc1_mean'] =\
    db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == 'ng/L']/(10**(6))

    db_filtered_mean.loc[db_filtered_mean.conc1_unit == 'umol/L', 'conc1_mean'] =\
    db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == 'umol/L']/(10**(3))*db_filtered_mean.mol_weight[db_filtered_mean.conc1_unit == 'umol/L']

    db_filtered_mean.loc[db_filtered_mean.conc1_unit == 'mmol/L', 'conc1_mean'] = \
    db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == 'mmol/L']*db_filtered_mean.mol_weight[db_filtered_mean.conc1_unit == 'mmol/L']

    db_filtered_mean.loc[db_filtered_mean.conc1_unit == 'nmol/L', 'conc1_mean'] =\
    db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == 'nmol/L']/(10**(6))*db_filtered_mean.mol_weight[db_filtered_mean.conc1_unit == 'nmol/L']

    db_filtered_mean.loc[db_filtered_mean.conc1_unit == 'mol/L', 'conc1_mean'] =\
    db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == 'mol/L']*(10**(3))*db_filtered_mean.mol_weight[db_filtered_mean.conc1_unit == 'mol/L']

    db_filtered_mean.loc[(db_filtered_mean.conc1_unit == 'g/L') | (db_filtered_mean.conc1_unit == 'g/dm3'), 'conc1_mean'] = \
    db_filtered_mean.conc1_mean[(db_filtered_mean.conc1_unit == 'g/L') | (db_filtered_mean.conc1_unit == 'g/dm3')]*(10**(3))

    db_filtered_mean.drop(columns=['conc1_unit'], inplace=True)

    # remove the experiments with Not Coded or Not Reported concentration type

    to_drop_type = db_filtered_mean.loc[
        (db_filtered_mean.conc1_type == 'NC') |
        (db_filtered_mean.conc1_type == 'NR')].index
    db_filtered_mean.drop(index=to_drop_type, inplace=True)

    return db_filtered_mean


def impute_test_feat(results_prefiltered):

    db = results_prefiltered.copy()

    db.loc[:, 'exposure_type'] = db.exposure_type.apply(
        lambda x: x.replace("/", ""))
    db.loc[:, 'exposure_type'] = db.exposure_type.apply(
        lambda x: x.replace("AQUA - NR", "AQUA") if "AQUA" in x else x)
    db.loc[:, 'exposure_type'] = db.exposure_type.apply(lambda x: 'AQUA'
                                                        if 'NR' in x else x)

    db.drop(columns=['test_location'], inplace=True)

    db.loc[:, 'control_type'] = db.control_type.apply(
        lambda x: x.replace("/", ""))
    db.loc[:, 'control_type'] = db.control_type.apply(lambda x: "Unknown"
                                                      if "NR" in x else x)

    db.loc[:, 'media_type'] = db.media_type.apply(lambda x: x.replace("/", ""))
    to_drop_media = db[db.media_type.isin(['NR', 'CUL', 'NONE', 'NC'])].index
    db.drop(to_drop_media, inplace=True)

    db.loc[:, 'application_freq_unit'] = db.application_freq_unit.apply(
        lambda x: "X" if ('NR' in x) | ('NC' in x) else x)

    return db


def impute_duration(results_prefiltered):
    # convert all units into hour
    db = results_prefiltered.copy()

    good_obs_unit = ["h", "d", "mi", "wk", "mo"]
    db_filtered_unit = db[db.obs_duration_unit.isin(good_obs_unit)].copy()

    to_drop_obs_mean = db_filtered_unit[db_filtered_unit.obs_duration_mean ==
                                        'NR'].index
    db_filtered_unit.drop(to_drop_obs_mean, inplace=True)
    db_filtered_unit.obs_duration_mean = db_filtered_unit.obs_duration_mean.astype(
        float)


    db_filtered_unit.loc[db_filtered_unit.obs_duration_unit == 'd', 'obs_duration_mean'] = \
                                        db_filtered_unit.obs_duration_mean[db_filtered_unit.obs_duration_unit == 'd']\
                                                                    .apply(lambda x: x*24)
    db_filtered_unit.loc[db_filtered_unit.obs_duration_unit == 'mi', 'obs_duration_mean'] = \
                                        db_filtered_unit.obs_duration_mean[db_filtered_unit.obs_duration_unit == 'mi']\
                                                                    .apply(lambda x: x/60)
    db_filtered_unit.loc[db_filtered_unit.obs_duration_unit == 'wk', 'obs_duration_mean'] = \
                                        db_filtered_unit.obs_duration_mean[db_filtered_unit.obs_duration_unit == 'wk']\
                                                                    .apply(lambda x: x*7*24)
    db_filtered_unit.loc[db_filtered_unit.obs_duration_unit == 'mo', 'obs_duration_mean'] = \
                                        db_filtered_unit.obs_duration_mean[db_filtered_unit.obs_duration_unit == 'mo']\
                                                                    .apply(lambda x: x*30*24)

    db_filtered_unit.drop(columns=['obs_duration_unit'], inplace=True)

    db_processed_duration = db_filtered_unit[
        db_filtered_unit.obs_duration_mean.isin([24, 48, 72, 96])].copy()

    return db_processed_duration


def impute_species(results_prefiltered):

    db = results_prefiltered.copy()
    # Dropping missing values relative to species (same values are missing for genus)
    to_drop_spec = db[db.species.isnull()].index
    db.drop(to_drop_spec, inplace=True)

    # Dropping missing values relative to family
    to_drop_fam = db[db.family.isnull()].index
    db.drop(to_drop_fam, inplace=True)

    return db


def select_impute_features(prefiltered_results, keep_columns):

    db = prefiltered_results.copy()
    db = db[keep_columns]

    db = impute_conc(db)

    db = impute_test_feat(db)

    db = impute_duration(db)

    db = impute_species(db)

    return db


# -------------------------------------------------------------------
# for each repeated experiments, set the median concentration as the results
def repeated_experiments(imputed_db):
    db = imputed_db.copy()
    db['fish'] = db['class'] + ' ' + db['tax_order'] + ' ' + db[
        'family'] + ' ' + db['genus'] + ' ' + db['species']

    db_species = db[[
        'class', 'tax_order', 'family', 'genus', "species", 'fish'
    ]]
    db_species = db_species.groupby("fish").first()

    final_db = db.groupby(by=[
        'test_cas', 'obs_duration_mean', 'conc1_type', 'fish', 'exposure_type',
        'control_type', 'media_type', 'application_freq_unit'
    ]).agg('median').reset_index()
    final_db = final_db.merge(db_species, on='fish')
    final_db.drop(columns=['fish', 'mol_weight'], inplace=True)

    return final_db


def to_cas(num):
    s = str(num)
    s = s[:-3] + '-' + s[-3:-1] + '-' + s[-1]
    return s


# -----------------------------------------------------------


def process_chemicals(features):

    chem_feat = adding_smiles_features(features)
    to_drop_nofeat = chem_feat[chem_feat['bonds_number'] == 'NaN'].index
    chem_feat.drop(to_drop_nofeat, inplace=True)

    return chem_feat


def process_smiles_features(chemical_features):

    db = chemical_features.copy()

    db.bonds_number = db.bonds_number.apply(lambda x: np.log1p(x))
    minmax = MinMaxScaler()
    minmax.fit(db[["bonds_number"]])
    db[["bonds_number"]] = minmax.transform(db[["bonds_number"]])

    db.atom_number = db.atom_number.apply(lambda x: np.log1p(x))
    minmax = MinMaxScaler()
    minmax.fit(db[["atom_number"]])
    db[["atom_number"]] = minmax.transform(db[["atom_number"]])

    db.Mol = db.Mol.apply(lambda x: np.log1p(x))
    minmax = MinMaxScaler()
    minmax.fit(db[["Mol"]])
    db[["Mol"]] = minmax.transform(db[["Mol"]])

    return db


# ----------------------------------------------------------
def cas_to_smiles(cas):
    queue = []
    for number in tqdm(cas):
        try:
            data = urlopen('http://cactus.nci.nih.gov/chemical/structure/' +
                           number + '/smiles').read().decode('utf8')
        except:
            data = np.NaN
        if str(data) == 'nan':
            try:
                data = pcp.get_compounds(number, 'name')[0].canonical_smiles
            except:
                data = np.NaN
        queue.append(data)
    return queue


def smiles_to_pubchem(smiles):
    pubchem = []
    for i in tqdm(smiles):
        try:
            pubchem.append(
                pcp.get_compounds(i, 'smiles')[0].cactvs_fingerprint)
        except:
            pubchem.append(np.NaN)
    return pubchem


def smiles_to_molweight(smiles):
    # unit g/mol
    molweight = []
    for i in tqdm(smiles):
        try:
            molweight.append(
                pcp.get_compounds(i, 'smiles')[0].molecular_weight)
        except:
            molweight.append(np.NaN)
    return molweight


# from selenium.webdriver.chrome.options import Options


def add_water_solubility(x):
    queue = []
    options = Options()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(
        'C:/Users/wjmen/Desktop/chromedriver_win32/chromedriver.exe',
        chrome_options=options)
    for url in tqdm(x):
        try:
            driver.get(
                'https://comptox.epa.gov/dashboard/dsstoxdb/results?search=' +
                url + '#properties')
            html = driver.page_source
            tables = pd.read_html(html)
            data = tables[0]
            if not data[data.Property == 'Water Solubility'].empty:
                result = data[data.Property == 'Water Solubility']
                result.replace('-', np.nan, method='bfill', inplace=True)
                solubility = result.fillna(method='bfill',
                                           axis=1).iloc[:, 1].iloc[0]
            else:
                solubility = 'no'
        except:
            solubility = 'no'
        queue.append(solubility)
    driver.close()
    return queue


def null_output_counts(dataframe):

    # Find columns that start with the interesting feature
    features_interested = list(dataframe.columns)

    df_nan = pd.DataFrame(index=features_interested,
                          columns=['null_values_inc_NC_NR%', '#outputs'])

    #Count total NaN + NR + NC
    for i in features_interested:
        df_nan['null_values_inc_NC_NR%'][i] = (
            sum(dataframe[i].isnull()) + len(dataframe[dataframe[i] == "NR"]) +
            len(dataframe[dataframe[i] == "NC"])) / len(dataframe) * 100
        df_nan['#outputs'][i] = len(dataframe[i].unique())
    return df_nan
