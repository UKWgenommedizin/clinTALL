# clinTALL: machine learning-driven multimodal based subtype classification and treatment outcome prediction in pediatric T-ALL

clinTALL is a multimodal deep learning framework for survival analysis and classification using gene expression, clinical, variant, and CNV data. It supports hyperparameter optimization, cross-cohort validation, user-model training, and inference using a YAML-driven configuration.

The framework is designed to reproduce the experiments from the associated manuscript while also allowing users to train and apply their own models on custom datasets.


## 1. Running clinTALL
clinTALL is controlled via a single YAML configuration file (config.yaml) that defines
which tasks to run, where data is located, and how models are trained and evaluated.

There are two main options to use clinTALL:
1) Use pretrained models to make predictions for your data (available at https://doi.org/10.5281/zenodo.18002152.)
-- install docker or python environment
-- download the pretrained model from zenodo: https://doi.org/10.5281/zenodo.18002152
-- add your own data into clintall/data/user for prediction. Detailed file description can be found in the section 3.2 below 
-- at the directory where clintall.py is located, execute following command
   ```docker-compose up ```
-- after the running, results can be found in clintall/.  Detailed file description can be found in the section 4.3 below 
   
3) Train a custom model on your data
-- install docker or python environment
-- git clone or download the repository
-- add your own data into clintall/data/user for prediction. Detailed file description can be found in the section 3.2 below
-- at the directory where clintall.py is located, execute following command
   ```docker-compose up ```
-- after the running, results can be found in clintall/.  Detailed file description can be found in the section 4 below 
   

### Quick Start
From the project root:

Configure the options in the config.yaml file, open the commandline/shell/powershell in the clinTall directory then:

- ```docker-compose up ```,
-   or ```python clinTall.py``` if you install an python environment from the requirements.txt locally

By default, clinTall.py will load and execute config.yaml. For more details about the options running clinTall, see the config_example.yaml and the sections below.

> Note: If you don't have a cuda enabled GPU replace the clinTall_requirements.txt with clinTall_requirements_cpu_only.txt (e.g. rename and delete)

### Configuration Overview

ClinTall uses three top-level YAML sections:
```
tasks:            # Which steps of the pipeline to run
DataManager:      # Paths and data-related options
Config:           # Training, optimization, and inference settings
```
#### Supported tasks:
| Task                 | Description                               |
| -------------------- | ----------------------------------------- |
| `hyperopt`           | Hyperparameter optimisation (as in paper) |
| `validate`           | Cross-cohort validation                   |
| `test_user_model`    | Train & test user models, save models     |
| `predict_user_model` | Apply trained user models to user data    |

Multiple tasks can be run sequentially. For predictions with user models (stored under user_model_results) pretrained models are provided and available at https://doi.org/10.5281/zenodo.18002152. (Performance reports for those are in inference_results). These models can be retrained if you provide data for it (under data/user/reference)


For formatting of input data please reference the files data/user/*.tsv (replace with your files). If only specific data modalities 
are being used, specify this in the config.yaml file. Output of the models will be saved in inference_results.

### Reproducibility

> Note: Some GPU ops may still introduce minor nondeterminism. - Run to run variance is observable but small



## 2. Project structure

```
.
├── clinTall.py
├── config.yaml
├── src/
│   └── helpers/
├── data/
│   ├── reference/
│   ├── user/
│   │   └── reference/
│   └── validation/
├── hyperopt_results/
├── validation_results/
├── inference_results/
└── user_model_results/

```

## 3. Data requirements

### 3.1 Reference data (data/reference/)
| File | Description |
|---|---|
| gene_expression.tsv | Gene expression matrix (samples × genes) |
| clinical_data.tsv | Clinical covariates |
| variants_data.tsv | Variant features |
| cnv_data.tsv | CNV features |
| labels.tsv | Binary / categorical labels |
| survival.tsv | Survival time and event indicator |


All files must:
 - Use the same sample IDs as row indices

### 3.2 User data (data/user/)
```
data/user/
├── gene_expression.tsv
├── gene_expression_names.tsv
├── clinical.tsv
├── variants.tsv
├── cnv.tsv
└── reference/
    ├── gene_expression.tsv
    ├── gene_expression_names.tsv
    ├── clinical.tsv
    ├── variants.tsv
    ├── cnv.tsv
    ├── labels.tsv
    ├── survival.tsv
```

Notes:

- User data must contain at least the features present in the reference data (see header of files data/user/reference/*)
- Extra features are allowed but will be ignored
- Gene expression can be provided using either:
    - ensembl_id (default)
    - gene_symbol (via gene_id_or_name in config)
- Users can also provide their own reference data and labels and fit on those.
- Users must not provide every data modality, select what you provided in the config.yaml
- Data should be provided as .csv or .tsv files, this should also be specified in the config.

### Validation data (data/validation/)
```
data/validation/
├── pölönen.tsv
└── hackenhaar.tsv
```

- All cohorts must have identical feature sets.
- One cohort is designated as the training reference.





## 4. Outputs

### 4.1 Hyperparameter optimization

- Best parameters per modality (best_params_*.json)

- Cross-validation metrics

- Risk scores and predicted labels

### 4.2 Validation

- Cross-cohort predictions

### 4.3 User model testing & inference

- Trained models (models_*.pkl)

- User-level predictions (.tsv)

- Aggregated performance summaries

## 5. Dependencies

See clinTall_requirements.txt

## 6. Notes & caveats

- Gene expression input must be integer counts for user-model inference.
- Feature mismatches raise explicit errors.
- CNV data is excluded from “Combined” modality.
- Some functions are placeholders for future extension.
