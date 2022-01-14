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
