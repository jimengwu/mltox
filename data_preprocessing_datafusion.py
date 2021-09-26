from helper_dataprocessing import *

# -----------------------Step 1: Load Data--------------------------------

# We used three experiment records files from Ecotox with version of 06/11/2020
# from website (https://cfpub.epa.gov/ecotox/):
# [i] species.txt
# [ii] tests.txt
# [iii] results.txt

# In the first file there are information regarding the species of animals: all the
# taxonomic denomination (from domain to variety), other typical denominations
# (Latin name or common name), and the ecotox group (fish, mammalians, birds,
# algae , ...).
# In the second file, tests.txt, there is information regarding the tests: laboratory
# conditions (exposure, control, application frequency, ...), CASRN of the chemical
# compound and the tested animal. Each cross-table reference is based on internal
# values that are unnecessary for the models.
# The last file contains experiment results records: LC50, ACC, EC50, etc.
# Aggregation of information tables on chemicals, species and tests is based on internal keys.


# For information on molecular weight, melting point, water solubility and smiles, we
# used data from CompTox (DSSTox_Predicted_NCCT_Model.zip: DSSToxQueryWPred1.xlsx,
# DSSToxQueryWPred2.xlsx, DSSToxQueryWPred3.xlsx, DSSToxQueryWPred4.xlsx).
# The website to get these files is
# ftp://newftp.epa.gov/COMPTOX/Sustainable_Chemistry_Data/Chemistry_Dashboard.

folder = "..."

DATA_RESULTS_PATH = folder + "results.txt"
DATA_TEST_PATH = folder + "tests.txt"
DATA_SPECIES_PATH = folder + "species.txt"
DATA_PROPERTY_PATH = [
    folder + "DSSToxQueryWPred1.xlsx",
    folder + "DSSToxQueryWPred2.xlsx",
    folder + "DSSToxQueryWPred3.xlsx",
    folder + "DSSToxQueryWPred4.xlsx",
]

tests, species, results, properties = load_raw_data(
    DATA_TEST_PATH, DATA_RESULTS_PATH, DATA_SPECIES_PATH, DATA_PROPERTY_PATH
)

# ---------------------Step 2: Prefiltering-----------------------------------

# After loading the data, we filter the data, select only aquatic species, and limit the
# test results to the concentration format, removed the mortality eperiments.

results_prefiltered = prefilter(
    species, tests, results, label="datafusion", effect="MOR"
)

# merging with the properties
results_prefiltered = results_prefiltered.merge(properties, on="test_cas")


# ---------------------Step 3: Effect & Endpoint Selection; Imputation--------------------------------

# Once loaded the data, we filter the effect and endpoint only if they have enough experiment records.
results_imputated = crosstab_rep_exp(results_prefiltered)

# ---------------------Step 4: Extraction of PubChem2D and molecular descriptors from CASRN------------

# We use the file to extract all information from the CASRN.

# If you want to avoid calculating all Pubchem2d fingerprints and molecular descriptors (about 2 hours)
# you can use the alternative function "process_chemicals" which takes as input a dataset with these
# info already extracted.

# Option 1: get the properties
# results_pub = smiles_to_pubchem(results_imputated)

# Option 2: use the saved file
pubchem = pd.read_csv("data/cas_pub_tot.csv")
results_pub = results_imputated.merge(pubchem[["smiles", "pubchem2d"]], on="smiles")


# extract other molecular properties
results_chem = extract_mol_properties(results_pub)

# ----------------------Step 5: Transformation of chemical features----------------

# Some variables need transformations to regularize their distributions.
# The transformed features are: "bonds_number", "atom_number", "mol_weight" and "WaterSolubility".
# Transformation is logarithmic and then MinMax. For "WaterSolubility" we used the Box-Cox
# transformation to normalize the distribution.

final_results = process_features(results_chem)

final_results.to_csv("datafusion_db_processed.csv")
print("data saved.")

for i in final_results.effect.unique():
    dfr = final_results[final_results.effect == i]

    dfr.to_csv("datafusion_db_processed_" + str(i) + ".csv")
