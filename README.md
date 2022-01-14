# Machine Learning for Toxicological Testing
## Introduction
In our project, we use a subset of the ECOTOX database and we use k-Nearest Neighbors (k-NN) and two kinds of read-across structure relationships models to predict the effect of untested chemicals on tested species and vice versa. 


## Prerequites

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
## Usage instruction
### Main database (ECOTOX Knowledge)
The data on the experiments is downloaded from Ecotox. Documentation on the dataset can be found at https://cfpub.epa.gov/ecotox/

Download the entire database as an ASCII file from the website, or do it from command line through
```
wget https://gaftp.epa.gov/ecotox/ecotox_ascii_03_15_2021.zip # This is for the version of 15 March 2021
unzip ecotox_ascii_03_15_2021.zip # Decompress
mv ecotox_ascii_03_15_2021 data/raw/ # Move it to the raw data directory
wget https://gaftp.epa.gov/COMPTOX/Sustainable_Chemistry_Data/Chemistry_Dashboard/DSSTox_Predicted_NCCT_Model.zip  # chemical properties
unzip DSSTox_Predicted_NCCT_Model.zip # Decompress
mv DSSTox_Predicted_NCCT_Model data/raw/ # Move it to the raw data directory
```
If having problems with certificates, you can try adding the --no-check-certificate flag.


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
    |    ├── knn                               # Folder to store KNN results
    |    └── mf                                # Folder to store MF results
    |    
    ├── src                                    # Source files
    |    ├── knn                               # KNN algorithm helpers and main
    |    |    ├── helper_knn.py          
    │    |    └── main_knn.py
    |    ├── mf                                # MF algorithm helpers and main
    |    |    ├── helper_mf.py          
    │    |    └── main_mf.py
    |    ├── preprocessing                     # Preprocessing algorithm helpers and main
    |    |    ├── helper_preprocessing.py          
    │    |    └── main_preprocessing.py
    |    └── general_helper.py                 # General helper for both algorithms
    ├── run.py                                 # Main entry point for the algorithms
    └── README.md

## Code reproducibility

## Authors
Jimeng Wu
Simone D'Ambrosi
Mentor: Marco Baity-Jesi
