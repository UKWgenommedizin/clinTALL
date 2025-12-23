import os
import argparse
import importlib.util
from pathlib import Path
import optuna
import json
import sys
import pandas as pd
import pickle
from src.helpers.tabm_competing import *
from src.helpers.tabm_competing_model import *
from src.helpers.training_funcs import *
from src.helpers.training_data_preprocessing import *
import os



class Config:
    """
    Placeholder for future Config implementation.
    Manages configuration settings for the ClinTall model.
    """
    def __init__(self, 
                 datasets_for_hyperopt: list = None,
                 hyperparameter_space: dict = None,
                 optimize_metric_weighting: float = 0.5,
                 num_hyperopt_trials: int = 75,
                 save_dir: str = './results',
                 hyperopt_save_dir: str = './hyperopt_results',
                 validation_save_dir: str = './validation_results',
                 inference_save_dir: str = './prediction_results',
                 user_save_dir: str = './user_models',
                 use_params: bool = False,
                 predict_with_user_model_modalities: list = None,):
        
        self.datasets_for_hyperopt = datasets_for_hyperopt if datasets_for_hyperopt is not None else []
        self.hyperparameter_space = hyperparameter_space 
        self.optimize_metric_weighting = optimize_metric_weighting
        self.num_hyperopt_trials = num_hyperopt_trials
        self.user_model_dir = user_save_dir
        self.save_dir = save_dir
        self.hyperopt_save_dir = hyperopt_save_dir
        self.validation_save_dir = validation_save_dir
        self.inference_save_dir = inference_save_dir
        self.use_params = use_params

        self.predict_with_user_model_modalities = ['gene_expression', 'Combined'] if predict_with_user_model_modalities is None else predict_with_user_model_modalities



    
        pass

class DataManager:
    """
    Manages loading and preprocessing of reference, user, and validation datasets.

    Args:
        reference_data_path (str): Path to the reference dataset.
        user_data_path (str): Path to the user dataset.
        user_reference_data_path (str): Path to the user reference dataset.
        validation_data_path (str): Path to the validation dataset.
        user_data_modalities (list): List of modalities present in the user dataset.
        validation_data_cohorts (list): List of cohorts present in the validation dataset.
        validation_data_reference (str): Reference for validation. (must be in validation_data_cohorts)
        gene_id_or_name (str): Type of gene identifiers used ('ensembl_id' or 'gene_symbol').
    
    Note: The directories should contain preprocessed data files compatible with the model.
    File naming conventions and formats should be adhered to as per the documentation:
     - Gene expression data: CSV/TSV files with samples as rows and genes as columns; Name format: 'gene_expression.csv' or 'gene_expression.tsv'.
     - Clinical data: CSV/TSV files with samples as rows and clinical features as columns; Name format: 'clinical_data.csv' or 'clinical_data.tsv'.
     - Variants data: CSV/TSV files with samples as rows and variant features as columns; Name format: 'variants_data.csv' or 'variants_data.tsv'.
     - CNV data: CSV/TSV files with samples as rows and CNV features as columns; Name format: 'cnv_data.csv' or 'cnv_data.tsv'.
     - Reference directory should contain all four data types and additional labels and survival information as needed. (labels.csv, survival.csv or .tsv)
    Data modalities should be specified as a list of strings, e.g., ['gene_expression', 'clinical_data', combined].
    File Extensions should be .csv or .tsv depending on the delimiter used and consistent across files per directory.
    Validation Data (only gene expression files) should be named 'cohort_name_gene_expression.csv' or 'cohort_name_gene_expression.tsv'.

    """
    def __init__(self, 
                 reference_data_path: str = 'data/reference', 
                 user_data_path: str = 'data/user', 
                 user_reference_data_path: str = 'data/user/reference',
                 validation_data_path: str = 'data/validation', 
                 user_data_modalities: list = ['gene_expression', 'clinical', 'variants', 'cnv'], 
                 validation_data_cohorts: list = ['pölönen', 'hackenhaar'], 
                 ref_file_format: str = 'csv', 
                 user_file_format: str = 'csv', 
                 val_file_format: str = 'csv', 
                 validation_data_reference: str = 'pölönen',
                 gene_id_or_name: str = 'ensembl_id',
                 user_model_dir: str = 'user_models',):
        
        self.user_model_dir = user_model_dir
        self.gene_id_or_name = gene_id_or_name
        if self.gene_id_or_name not in ['ensembl_id', 'gene_symbol']:
            raise ValueError("gene_id_or_name must be either 'ensembl_id' or 'gene_symbol'")
        # Load reference data
        if reference_data_path is not None:
            self.ref_dir = Path(reference_data_path)
            self.ref_data = {
                'gene_expression': pd.read_csv(self.ref_dir / f'gene_expression.{ref_file_format}', index_col=0, sep=',' if ref_file_format == 'csv' else '\t'),
                'clinical': pd.read_csv(self.ref_dir / f'clinical_data.{ref_file_format}', index_col=0, sep=',' if ref_file_format == 'csv' else '\t'),
                'variants': pd.read_csv(self.ref_dir / f'variants_data.{ref_file_format}', index_col=0, sep=',' if ref_file_format == 'csv' else '\t'),
                'cnv': pd.read_csv(self.ref_dir / f'cnv_data.{ref_file_format}', index_col=0, sep=',' if ref_file_format == 'csv' else '\t'),
            }
            self.ref_labels = pd.read_csv(self.ref_dir / f'labels.{ref_file_format}', index_col=0, sep=',' if ref_file_format == 'csv' else '\t')
            self.ref_survival = pd.read_csv(self.ref_dir / f'survival.{ref_file_format}', index_col=0, sep=',' if ref_file_format == 'csv' else '\t')
            #self.user_model_test_gex = pd.read_csv(self.ref_dir / f'polonen_counts_for_user_model_test.{ref_file_format}', index_col=0, sep=',' if ref_file_format == 'csv' else '\t')
            # user_model_test_gex is numeric
            #self.user_model_test_gex = self.user_model_test_gex.apply(pd.to_numeric, errors='raise')
        else:
            print("Warning: No reference data path provided. Reference data will not be loaded. Training tasks may fail.")

        if user_data_path is not None:
            self.user_reference_data = {}
            self.user_data = {}
            self.user_dir = Path(user_data_path)
            self.user_reference_data_path = Path(user_reference_data_path) 
            
            for modality in user_data_modalities:
                if modality == 'gene_expression' and self.gene_id_or_name == 'gene_symbol':
                    self.user_data[modality] = pd.read_csv(self.user_dir / f'{modality}_names.{user_file_format}', index_col=0, sep=',' if user_file_format == 'csv' else '\t')
                else:    
                    self.user_data[modality] = pd.read_csv(self.user_dir / f'{modality}.{user_file_format}', index_col=0, sep=',' if user_file_format == 'csv' else '\t')
            
            if self.user_reference_data_path is not None:
                self.user_reference_data_path = Path(user_reference_data_path)
                for modality in user_data_modalities:
                    if modality == 'gene_expression' and self.gene_id_or_name == 'gene_symbol':
                        self.user_reference_data[modality] = pd.read_csv(self.user_reference_data_path / f'{modality}_names.{user_file_format}', index_col=0, sep=',' if user_file_format == 'csv' else '\t')
                    else:    
                        self.user_reference_data[modality] = pd.read_csv(self.user_reference_data_path / f'{modality}.{user_file_format}', index_col=0, sep=',' if user_file_format == 'csv' else '\t')
            
            self.user_ref_labels = pd.read_csv(self.user_dir / 'reference' / f'labels.{user_file_format}', index_col=0, sep=',' if user_file_format == 'csv' else '\t')
            self.user_ref_survival = pd.read_csv(self.user_dir / 'reference' / f'survival.{user_file_format}', index_col=0, sep=',' if user_file_format == 'csv' else '\t')

            self.check_validity_of_user_data(self.user_reference_data, self.user_data)
        else:
            print("Warning: No user data path provided. User data will not be loaded. Inferencing tasks may fail.")

        # Load validation data
        if validation_data_path is not None:
            self.val_dir = Path(validation_data_path)
            self.val_ref = validation_data_reference
            self.val_data = {}
            for cohort in validation_data_cohorts:
                self.val_data[cohort] = pd.read_csv(self.val_dir / f'{cohort}.{val_file_format}', index_col=0, sep=',' if val_file_format == 'csv' else '\t')
            self.check_validity_of_validation_data(self.val_data, self.val_ref)
        else:
            print("Warning: No validation data path provided. Validation data will not be loaded. Validation tasks may fail.")

        
        

    def get_val_data(self, cohort: str = None):
        if cohort is None:
            raise ValueError("Cohort must be specified to get validation data.")
        if cohort not in self.val_data:
            raise ValueError(f"Cohort '{cohort}' not found in validation data.")
        return self.val_data[cohort]

    def check_validity_of_user_data(self, ref_data: dict, user_data: dict,):
        
        for modality, df in user_data.items():

            if modality not in ref_data:
                raise ValueError(f"Modality '{modality}' in user data not found in reference data.")
            missing_cols = set(ref_data[modality].columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"User data for modality '{modality}' is missing columns: {missing_cols}")
            too_many_cols = set(df.columns) - set(ref_data[modality].columns)
            if too_many_cols:
                raise ValueError(f"User data for modality '{modality}' has unexpected columns: {too_many_cols}")
            
            #make sure columns are in the same order
            if modality != 'gene_expression':
                self.user_data[modality] = df[ref_data[modality].columns]
            
            if modality == 'gene_expression':
                # check if column names match between user and reference data
                ref_gene_ids = set(ref_data[modality].columns)
                user_gene_ids = set(df.columns)
                # if user gene ids are more then reference, but all reference gene ids are in user gene ids, print a warning and ignore extra genes
                if not ref_gene_ids.issubset(user_gene_ids):
                    missing_genes = ref_gene_ids - user_gene_ids
                    raise ValueError(f"User gene expression data is missing genes present in reference data: {missing_genes}")
                else:
                    extra_genes = user_gene_ids - ref_gene_ids
                    if extra_genes:
                        print(f"Warning: User gene expression data has {len(extra_genes)} extra genes not present in reference data. These will be ignored.")
                    self.user_data[modality] = df.drop(columns=extra_genes)
                #check if now both have same columns, in same order
                if not all(self.user_data[modality].columns == ref_data[modality].columns):
                    self.user_data[modality] = self.user_data[modality][ref_data[modality].columns]

    def check_validity_of_validation_data(self, val_data: dict, validation_data_reference: str):
        # check if all sets are the same   
        cols = val_data.get(validation_data_reference, None)
        if cols is None:
            raise ValueError(f"Validation data reference '{validation_data_reference}' not found in validation datasets.")
        else:
            cols= cols.columns.to_list() 
        for cohort, df in val_data.items():
            missing_cols = set(cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Validation data for cohort '{cohort}' is missing columns: {missing_cols}")
            too_many_cols = set(df.columns) - set(cols)
            if too_many_cols:
                raise ValueError(f"Validation data for cohort '{cohort}' has unexpected columns: {too_many_cols}")
            
    def get_ref_data(self, modality: str = None):
        if modality == 'Combined':
            # return all modalities except CNV combined:
            return pd.concat([self.ref_data[mod] for mod in self.ref_data if mod != 'cnv'], axis=1)
        elif modality:
            if modality not in self.ref_data:
                raise ValueError(f"Modality '{modality}' not found in reference data.")
            return self.ref_data.get(modality, None) 
        else:
            raise ValueError("get_ref_data: Modality must be specified.")
        
    def get_ref_labels(self):
        return self.ref_labels

    def get_ref_survival(self):
        return self.ref_survival

    def get_user_data(self, modality: str = None, reference: bool = False):
        if reference:
            if modality == 'Combined':
                # return all modalities except CNV combined:
                return pd.concat([self.user_reference_data[mod] for mod in self.user_reference_data if mod != 'CNV'], axis=1)
            elif modality:
                if modality not in self.user_reference_data:
                    raise ValueError(f"Modality '{modality}' not found in user reference data.")
                return self.user_reference_data.get(modality, None)
            else:
                raise ValueError("get_user_data: Modality must be specified.")
        else:
            if modality == 'Combined':
                # return all modalities except CNV combined:
                return pd.concat([self.user_data[mod] for mod in self.user_data if mod != 'CNV'], axis=1)
            elif modality:
                if modality not in self.user_data:
                    raise ValueError(f"Modality '{modality}' not found in user data.")
                return self.user_data.get(modality, None)

    


                         

class ClinTall:
    """
    Manages the overall workflow for ClinTall, including hyperparameter optimization, validation, user model testing, and inference.
    """
    def __init__(self, list_of_tasks: list, DataManager_args: dict, Config_args: dict):
        self.list_of_tasks = list_of_tasks
        print("Initializing DataManager and Config...")
        self.data_manager = DataManager(**DataManager_args)
        self.config = Config(**Config_args)

    def run(self):
        accepted_tasks = ['hyperopt', 'validate', 'test_user_model', 'predict_user_model']
        for task in self.list_of_tasks:
            if task not in accepted_tasks:
                raise ValueError(f"Task '{task}' is not recognized. Accepted tasks are: {accepted_tasks}")
            print(f"Running task: {task}...")
            if task == 'hyperopt':
                self.run_hyperparameter_optimization()
            elif task == 'validate':
                self.run_validation()
            elif task == 'test_user_model':
                self.test_user_model_on_reference_data()
            elif task == 'predict_user_model':
                self.run_inference()
            print(f"Completed task: {task}.")
        print("All tasks completed.")
        
        return True
                


    def run_hyperparameter_optimization(self):
        print("Starting hyperparameter optimization...")
        datasets_to_use = self.config.datasets_for_hyperopt if hasattr(self.config, 'datasets_for_hyperopt') else None

        labels = self.data_manager.get_ref_labels()
        survival = self.data_manager.get_ref_survival()

        hyperparameter_space = self.config.hyperparameter_space if hasattr(self.config, 'hyperparameter_space') else {}
        optimize_metric_weighting = self.config.optimize_metric_weighting if hasattr(self.config, 'optimize_metric_weighting') else 0.5
        num_trials = self.config.num_hyperopt_trials if hasattr(self.config, 'num_hyperopt_trials') else 75
        
        if datasets_to_use is None:
            datasets_to_use = list(self.data_manager.ref_data.keys()) + ['Combined']

        aggregated_results = {'dataset': [], 'cindex_cv': [], 'accuracy_cv': []}
        all_cv_perfs = {'dataset': [], 'cindex': [], 'accuracy': []}

        for dataset in datasets_to_use:
            data = self.data_manager.get_ref_data(modality=dataset)
            common_samples = data.index.intersection(labels.index).intersection(survival.index)
            data = data.loc[common_samples]
            labels_subset = labels.loc[common_samples]
            survival_subset = survival.loc[common_samples]
            params_path = self.data_manager.ref_dir 
            if self.config.use_params:
                print(f"Using predefined parameters for dataset: {dataset}")
                with open(f'{params_path}/best_params_{dataset}.json', 'r') as f:
                    params = json.load(f)

            print(f"Starting hyperparameter optimization for dataset: {dataset} with {data.shape[0]} samples and {data.shape[1]} features.")
            results, params = hyperparameter_optimisation(data, labels_subset, survival_subset, 
                                                hyperparameter_space, optimize_metric_weighting, num_trials, 
                                                self.config.hyperopt_save_dir, dataset_name=dataset, 
                                                use_params=params if self.config.use_params else False)
            # Save best parameters
            with open(f'{self.config.hyperopt_save_dir}/best_params_{dataset}.json', 'w') as f:
                json.dump(params, f, indent=4)

            # Aggregate results
            aggregated_results['dataset'].append(dataset)
            aggregated_results['cindex_cv'].append(results['cindex_cv'])
            aggregated_results['accuracy_cv'].append(results['accuracy_cv'])


            all_cv_perfs['dataset'].extend([dataset]*5)
            all_cv_perfs['cindex'].extend(results['all_cindexes'])
            all_cv_perfs['accuracy'].extend(results['all_accuracies'])

            # Save Predictions
            all_risks = pd.concat(results['all_risks'])
            all_risks.to_csv(f'{self.config.hyperopt_save_dir}/risks_{dataset}.csv', index=True)
            all_pred_labels = pd.concat(results['all_pred_labels'])
            all_pred_labels.to_csv(f'{self.config.hyperopt_save_dir}/pred_labels_{dataset}.csv', index=True)
            eta_per_cause = pd.concat(results['eta_per_cause'])
            eta_per_cause.to_csv(f'{self.config.hyperopt_save_dir}/eta_per_cause_{dataset}.csv', index=True)

        # Save total Results
        all_cv_perfs_df = pd.DataFrame(all_cv_perfs)
        all_cv_perfs_df.to_csv(f'{self.config.hyperopt_save_dir}/all_cv_perfs.csv', index=True)
        aggregated_results_df = pd.DataFrame(aggregated_results)
        aggregated_results_df.to_csv(f'{self.config.hyperopt_save_dir}/aggregated_hyperopt_results.csv', index=True)
            
     
    def run_validation(self):

        print("Starting validation...")
        labels = self.data_manager.get_ref_labels()
        survival = self.data_manager.get_ref_survival()
        train_data = self.data_manager.get_val_data(cohort=self.data_manager.val_ref)

        for name, test_data in self.data_manager.val_data.items():
            if name == self.data_manager.val_ref:
                continue  # skip reference dataset used for training
            common_samples = train_data.index.intersection(labels.index).intersection(survival.index)
            train_data = train_data.loc[common_samples]
            labels = labels.loc[common_samples]
            survival = survival.loc[common_samples]

            fit_and_validate(train_data, test_data, name, common_samples, labels, survival, save_dir=self.config.validation_save_dir)
        
    def run_inference(self):
        models_to_use = self.config.predict_with_user_model_modalities
        print("Starting inference...")
        for modality in models_to_use:
            print(f"Predicting with user model for modality: {modality}...")
            if modality == 'Combined':
                gene_data = self.data_manager.get_user_data(modality='gene_expression')
                other_data = pd.concat([self.data_manager.get_user_data(mod) for mod in self.data_manager.user_data if mod not in ['cnv', 'gene_expression']], axis=1)
            elif modality == 'gene_expression':
                gene_data = self.data_manager.get_user_data(modality='gene_expression')#self.data_manager.user_model_test_gex#
                other_data = None
            else:
                gene_data = None
                other_data = self.data_manager.get_user_data(modality=modality)
            with open(f'{self.config.user_model_dir}/models_{modality}.pkl', 'rb') as f:
                models = pickle.load(f)
            #with open(f'{self.config.inference_save_dir}/gene_preprocessors_{modality}.pkl', 'rb') as f:
            #    gene_preprocessors = pickle.load(f)

            common_samples = None
            if gene_data is not None and other_data is not None:
                common_samples = gene_data.index.intersection(other_data.index)
                gene_data = gene_data.loc[common_samples]
                other_data = other_data.loc[common_samples]
            elif gene_data is not None and other_data is None:
                common_samples = gene_data.index
            else:
                common_samples = other_data.index

            if not np.all(np.equal(np.mod(gene_data.values, 1), 0)):
                raise ValueError("Input count data must be integers - Test in clinTall.py:run_inference")
        
            results = predict_with_user_model(models=models, #gene_preprocessors=gene_preprocessors,
                                    X_gene=gene_data, X_other=other_data,)
            
            for key, df in results.items():
                df.to_csv(f'{self.config.inference_save_dir}/user_model_{modality}_{key}.tsv', index=True, sep='\t')


    def test_user_model_on_reference_data(self):
        print("Testing user model on reference data...")
        datasets_to_use = self.config.datasets_for_hyperopt if hasattr(self.config, 'datasets_for_hyperopt') else None

        labels = self.data_manager.user_ref_labels
        survival = self.data_manager.user_ref_survival

        #maybe add later
        #hyperparameter_space = self.config.hyperparameter_space if hasattr(self.config, 'hyperparameter_space') else {}
        #optimize_metric_weighting = self.config.optimize_metric_weighting if hasattr(self.config, 'optimize_metric_weighting') else 0.5
        #num_trials = self.config.num_hyperopt_trials if hasattr(self.config, 'num_hyperopt_trials') else 75
        
        if datasets_to_use is None:
            datasets_to_use = list(self.data_manager.user_reference_data.keys()) + ['Combined']

        aggregated_results = {'dataset': [], 'cindex_cv': [], 'accuracy_cv': []}
        all_cv_perfs = {'dataset': [], 'cindex': [], 'accuracy': []}

        for dataset in datasets_to_use:
            if dataset != 'Combined' and dataset != 'gene_expression':
                data = self.data_manager.get_user_data(modality=dataset)
                gene_data = None
            elif dataset == 'gene_expression':
                data = None
                gene_data = self.data_manager.user_reference_data['gene_expression']
            elif dataset == 'Combined':
                data = pd.concat([self.data_manager.get_user_data(mod) for mod in self.data_manager.user_reference_data if mod not in ['cnv', 'gene_expression']], axis=1)
                gene_data = self.data_manager.user_reference_data['gene_expression']
            else:
                raise ValueError(f"Dataset '{dataset}' not recognized for user model testing.")
            
            if data is not None and gene_data is not None:
                common_samples = data.index.intersection(gene_data.index).intersection(labels.index).intersection(survival.index)
            elif data is None and gene_data is not None:
                common_samples = gene_data.index.intersection(labels.index).intersection(survival.index)
            elif data is not None and gene_data is None:
                 common_samples = data.index.intersection(labels.index).intersection(survival.index)

            print(f"Testing user model on dataset: {dataset} with {len(common_samples)} common samples.")
            results = fit_tabm_csr_user_model(labels.loc[common_samples], survival.loc[common_samples], 
                                              X_gene=gene_data.loc[common_samples] if gene_data is not None else None, 
                                              X_other=data.loc[common_samples] if data is not None else None,
                                              epochs = 1000,
                                              patience=50,
                                              save_models=True
                                              )
            aggregated_results['dataset'].append(dataset)
            aggregated_results['cindex_cv'].append(results['cindex_cv'])
            aggregated_results['accuracy_cv'].append(results['accuracy_cv'])
            all_cv_perfs['dataset'].extend([dataset]*5)
            all_cv_perfs['cindex'].extend(results['all_cindexes'])
            all_cv_perfs['accuracy'].extend(results['all_accuracies'])
            with open(f'{self.config.user_model_dir}/models_{dataset}.pkl', 'wb') as f:
                pickle.dump(results['models'], f)
            #with open(f'{self.config.inference_save_dir}/gene_preprocessors_{dataset}.pkl', 'wb') as f:
            #    pickle.dump(results['gene_preprocessors'], f)
            #with open(f'{self.config.inference_save_dir}/parameters_{dataset}.pkl', 'wb') as f:
            #    pickle.dump(results['parameters'], f)
            del results # free up space

        pd.DataFrame(aggregated_results).to_csv(f'{self.config.inference_save_dir}/user_model_test_aggregated_results.csv', index=True)
        pd.DataFrame(all_cv_perfs).to_csv(f'{self.config.inference_save_dir}/user_model_test_all_cv_perfs.csv', index=True)
            # Not save for testing

import yaml
from pathlib import Path

def run_clintall_from_yaml(config_path):
    """
    Load a YAML configuration file and run ClinTall accordingly.

    Expected YAML structure:
      tasks: list
      DataManager: dict
      Config: dict
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- basic validation ---
    if "tasks" not in cfg:
        raise ValueError("YAML config must define a 'tasks' list")

    if "DataManager" not in cfg:
        raise ValueError("YAML config must define a 'DataManager' section")

    if "Config" not in cfg:
        raise ValueError("YAML config must define a 'Config' section")

    tasks = cfg["tasks"]
    data_manager_args = cfg["DataManager"]
    config_args = cfg["Config"]

    if not isinstance(tasks, list):
        raise TypeError("'tasks' must be a list")

    if not isinstance(data_manager_args, dict):
        raise TypeError("'DataManager' must be a dictionary")

    if not isinstance(config_args, dict):
        raise TypeError("'Config' must be a dictionary")

    # --- construct and run ---
    clintall = ClinTall(
        list_of_tasks=tasks,
        DataManager_args=data_manager_args,
        Config_args=config_args
    )

    return clintall.run()

if __name__ == "__main__":
    set_all_seeds(42)
    # Despite setting seeds, some operations may still introduce non-determinism?
    # Results seem not reproducible to the exact number across runs.
    
    # Ignore specific warnings from rtdl_num_embeddings
    # Some features are in a shape rtdl_num_embeddings infomrms it would be the same as using sklearn.MinMaxScaler
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="rtdl_num_embeddings"
    )

    run_clintall_from_yaml('config.yaml')

    # hyperopt, validate are the two tasks to generate results for the paper
    """ClinTall(list_of_tasks=['test_user_model', 'predict_user_model',], 
             DataManager_args={
                 'reference_data_path': 'data/reference',
                 'user_data_path': 'data/user',
                 'validation_data_path': 'data/validation',
                 'user_data_modalities': ['gene_expression', 'clinical', 'variants', 'cnv'],
                 'validation_data_cohorts': ['pölönen', 'bfm', 'hackenhaar'],
                 'ref_file_format': 'tsv',
                 'user_file_format': 'tsv',
                 'val_file_format': 'tsv',
                 'validation_data_reference': 'pölönen'
             },
             Config_args={
                 'datasets_for_hyperopt': ['Combined', 'cnv', 'gene_expression', 'clinical', 'variants', ],
                 'hyperparameter_space': None,
                 'optimize_metric_weighting': 0.5,
                 'num_hyperopt_trials': 75,
                 'hyperopt_save_dir': './hyperopt_results',
                 'use_params': True
             }).run()"""