# Machine Learning for Toxicological Testing

## T-models

We tested five models, T_tuned, T_median, T_median_ml, T_tuned_ml, and T_num_ml. The first two models compared in vitro labels to in vivo labels in the same experiment (here same refers to the same chemical and taxonomy), and the last three models involved a random forest model.

For the first two models, we used different thresholds to classify the in vitro concentrations as binary labels, while the in vivo labels were still classified using 1 mg/L. After this, we compared the in vitro labels to the classified in vivo labels and calculated the number of experiments using the matched in vivo and in vitro labels. We specified the percentage of matched experiments as accuracy.
With the T_tuned model, we tested different thresholds and obtained the best threshold with the highest accuracy for our dataset of 13 mg/L. For the T_median model, we simply used the median concentration value of the input dataset as the threshold.

The model using RF involves the same in vitro classification procedure, after we classify the in vitro concentrations as in vitro labels, we will use the in vitro labels as input features and train the random forest model to predict the in vivo labels using the in vitro labels. In vitro information.

T_tuned_ml uses the best threshold found by T_tuned, while T_median_ml will use the median concentration from the training dataset in T_median.

T_num_ml uses in vitro concentrations as input features.

How to run these models:

T_tuned:

python RASAR_mulneigh_bi_cte_T_null.py   -i .../data/invitro/invivo_repeated_w_invitro_eawag.csv -r 20 -t_ls "tuned"  -dbi "overlap" -wi "own"   -il "label"   -o "T_models/T_null/repeat_own_label"

T_median:

python RASAR_mulneigh_bi_cte_T_null.py   -i .../data/invitro/invivo_repeated_w_invitro_eawag.csv -r 20 -t_ls "median" -dbi "overlap" -wi "own"   -il "label"   -o "T_models/T_median/repeat_own_label"

T_tuned_ml:

python RASAR_mulneigh_bi_cte_T_models.py -i .../data/invitro/invivo_repeated_w_invitro_eawag.csv -r 20 -ah 1  -ap 1  -t_ls "tuned" -dbi "overlap" -wi "own"   -il "label"        -o "T_models/T_tuned_ml/repeat_own_label"

T_median_ml:

python RASAR_mulneigh_bi_cte_T_models.py -i .../data/invitro/invivo_repeated_w_invitro_eawag.csv -r 20 -ah 1  -ap 1  -t_ls "median" -dbi "overlap" -wi "own"   -il "label"        -o "T_models/T_median_ml/repeat_own_label"

T_num_ml:

python RASAR_mulneigh_bi_cte_T_models.py -i .../data/invitro/invivo_repeated_w_invitro_eawag.csv -r 20 -ah 1  -ap 1 -n 1   -dbi "overlap" -wi "own"   -il "number"              -o "T_models/T_num_ml/repeat_own_number"

## Introduction

In our project, we use a subset of the ECOTOX database and we use k-Nearest Neighbors (k-NN), logistic regression, random forest, and two kinds of read-across structure relationships models (simple RASAR and DF RASAR) to predict the effect of untested chemicals on tested species and vice versa.

## Prerequites

### Basic Prerequites

- `Python` (tested on version **_3.6.2_**)
- [pip](https://pip.pypa.io/en/stable/) (tested on version _21.1.2_) (For package installation, if needed)
- `Anaconda` (Test on version _4.2.9_) (More information about Anaconda installation on your OS [here](https://docs.anaconda.com/anaconda/install/)) (For package installation)
- `numpy` (tested on version _1.19.1_)
- `scikit-learn` (tested on version _0.23.2_)
- `scipy`
- `pandas` (tested on version _1.1.3_)
- `h2o` (tested on version _3.32.1.3_) (Only needed for multiclass datafusion model)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

**NOTE**: with this configuration, the `run.py` will run without **preprocessing**

### Preprocessing prerequisites

#### Main database (ECOTOX Knowledge)

The data on the experiments is downloaded from Ecotox. Documentation on the dataset can be found at <https://cfpub.epa.gov/ecotox/>

Download the entire database as an ASCII file from the website, or do it from command line through

```
wget https://gaftp.epa.gov/ecotox/ecotox_ascii_09_15_2020.zip # This is for the version of 15 March 2021
unzip ecotox_ascii_09_15_2020.zip # Decompress
mv ecotox_ascii_09_15_2020 data/raw/ # Move it to the raw data directory
wget https://gaftp.epa.gov/COMPTOX/Sustainable_Chemistry_Data/Chemistry_Dashboard/DSSTox_Predicted_NCCT_Model.zip  # chemical properties
unzip DSSTox_Predicted_NCCT_Model.zip # Decompress
mv DSSTox_Predicted_NCCT_Model data/raw/ # Move it to the raw data directory
```

If having problems with certificates, you can try adding the --no-check-certificate flag.

#### Packages

To run the preprocessing phase the : `rdkit` (Tested on version _2017.09.1_) package and pubchempy(_1.0.4_) are needed.  

To install `rdkit` in your environment use the command

```bash/CMD
conda install -c rdkit rdkit
```

_Note_: the `run.py` will work also without `rdkit` and `pubchempy` if no preprocessing is used. An already preprocessed dataset will be used.

## Usage instruction

1. Open CMD/Bash
2. Activate the environment with needed packages installed
3. Move to the root folder, where the `run.py` is located
4. Execute the command ```python run.py``` with one or more of the following arguments:

```
Mandatory arguments:
  - encoding:
      Select either one of the two possible classification:
        -binary: Execute binary classification
        -multiclass: Execute multiclass (5 class) classification
Optional arguments:    
  -h, --help: show arguments help message and exit
  -preproc:  Execute all the preprocessing steps from the raw dataset. If not set, an already preprocessed dataset will be loaded.
  -c:  Execute all the models using only chemical (c) information. If not set, skip this step.
  -cte:  Execute all the models using chemical, taxanomy and experiment (cte) information. If not set, skip this step.
  -cte_wa:  Execute all the models using cte information and alphas. If not set, skip this step.
```

## Folder structure

    .
    ├── data 
    |   ├── processed                          # Already preprocessed data directly usable
    |   |     ├── final_db_processed.csv          
    │   |     └── cas_to_smiles.csv  
    │   └── raw                                # Raw data (need preprocessing)
    |        ├── results.txt 
    |        ├── species.txt 
    │        └── tests.txt 
    ├── output 
    |    ├── c                                    # Folder to store c results
    |    └── cte                                  # Folder to store cte results
    |    └── cte_wa                               # Folder to store cte_wa results
    |    
    ├── src                                    # Source files
    |    ├── model                               
    |    |    ├── helper_model.py              # algorithm helpers
    │    |    ├── KNN.py
    |    |    ├── LR.py          
    |    |    ├── RF.py    
    |    |    ├── RASAR_simple.py    
    │    |    └── RASAR_df.py
    |    ├── preprocessing                     # Preprocessing algorithm helpers and algorithms
    |         ├── helper_preprocess.py          
    |         ├── helper_chemproperty.py          
    │         └── data_preprocess.py
    ├── run.py                                 # Main entry point for the algorithms
    └── README.md

## Authors

- Jimeng Wu
_Supervisor_: Marco Baity-Jesi
