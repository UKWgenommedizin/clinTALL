from ast import Dict
from time import sleep
from typing import Optional
#from array_api_compat import device
import pandas as pd
import sys
from pathlib import Path

from pyparsing import Any
sys.path.append(str(Path(__file__).resolve().parent.parent))
import gc
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from src.helpers.training_data_preprocessing import categorize_columns, get_data_tabm, ColumnCategories
#from src.helpers.tabmhelpers import *   
from src.helpers.tabm_competing import MultiTaskTabMWrapperWithCauses  # wrapper for TabM outputs
from src.helpers.tabm_competing_model import CoxPH_CompetingRisk_TabM    # main trainer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# optional survival metrics
try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    SKSURV_AVAILABLE = True
except Exception:
    SKSURV_AVAILABLE = False

# TabM / rtdl imports (unchanged)
import rtdl_num_embeddings
import tabm
import torchtuples as tt
import optuna
from src.helpers.training_data_preprocessing import categorize_columns, get_data_tabm, ColumnCategories, prepare_tabm_inputs
#from src.helpers.tabmhelpers import prepare_tabm_inputs#, get_data_tabm
import os
import random 

def set_all_seeds(seed: int = 42):
    """
    Sets seeds for all random number generators used in PyTorch training
    to maximize reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    print(f"Setting seed to {seed}...")
    
    # 1. Python built-in random module
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    
    # 4. CUDA (GPU) Randomness
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

        
    print("Seeds set successfully.")


def calculate_weights(y_train, device='cpu'):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def make_relaxed_stratification_labels(subtypes, events, min_per_group=5):
    """
    Returns a stratification vector where:
        1. composite (subtype_event) is used if frequent enough
        2. else fallback to 'Other'
    """
    df = pd.DataFrame({
        "sub": subtypes.astype(str),
        "evt": events.astype(str),
    })

    # initial composite label
    df["combo"] = df["sub"] + "_" + df["evt"]

    # Compute counts
    combo_counts = df["combo"].value_counts()
    
    # Identify rare combos (count < min_per_group)
    rare_combos = set(combo_counts[combo_counts < min_per_group].index)

    strat = []

    for i, row in df.iterrows():
        c = row["combo"]
        
        if c in rare_combos:
             strat.append('Other')
        else:
            strat.append(c)

    return pd.Series(strat)

def get_relaxed_stratified_split(X, subtypes, events, n_splits=5, seed=123):
    strat_labels = make_relaxed_stratification_labels(subtypes, events, min_per_group=n_splits)
    if strat_labels.value_counts().min() < n_splits:
        print("Warning: Still too few samples, falling back to coarse stratification.")
        fallback = pd.Series(subtypes.astype(str))
        if fallback.value_counts().min() < n_splits:
            print("Fallback also too small. Using plain KFold.")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return kf.split(X)
        strat_labels = fallback
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return kf.split(X, strat_labels)


def _align_xy(dfs, labels):
    common_idx = labels.index.astype(str)
    for df in dfs:
        common_idx = common_idx.intersection(df.index.astype(str))
    if len(common_idx) == 0:
        raise ValueError("No common samples between selected datasets and labels after alignment.")
    aligned_dfs = [df.loc[common_idx].sort_index() for df in dfs]
    X = pd.concat(aligned_dfs, axis=1)
    y = labels.loc[common_idx].sort_index()
    X = X.dropna(axis=1, how="all").fillna(0)
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(np.int8)
        elif not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X, y

def _read_labels(labels_path: Path, label_column: str = "Reviewed.subtype") -> pd.Series:
    df = pd.read_pickle(labels_path)
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in {labels_path}. Available: {list(df.columns)}")
    ser = df[label_column].astype(str)
    return ser

def _filter_labels(labels: pd.Series, exclude_terms: None, include_terms: None) -> pd.Series:
    if not exclude_terms and not include_terms:
        return labels
    if include_terms:
        mask = pd.Series(False, index=labels.index)
        for term in include_terms:
            mask |= labels.str.contains(str(term), na=False)
        return labels[mask]
    mask = pd.Series(True, index=labels.index)
    for term in exclude_terms:
        mask &= ~labels.str.contains(str(term), na=False)
    return labels[mask]

def fit_tabm_csr(
        X, labels, df_clin,
        days_col='EFS_days',
        cause_col='Event_Type',
        n_splits=5, test_size=0.2,
        epochs=1, batch_size=512,
        patience=None, seed=42,
        labels_col='subtype', params=None, trial=None, 
        save = False, inference=False, seed_i = 0, alpha=0.15, load_model = None, user_model=False,
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    if load_model is not None:
        params = load_model['parameters']

    l_e = LabelEncoder()

    labels = pd.Series(l_e.fit_transform(labels.values.squeeze()), index=labels.index)


    if params is None:
        params = {}
    if patience is None:
        patience = max(min(params.get('epochs', epochs)//10, 100), 10)

    seed = seed + seed_i + 5

    lr = params.get('lr', 1e-3)
    weight_decay = params.get('weight_decay', 3e-4)
    d_block = params.get('d_block', 512)
    d_embedding = params.get('d_embedding', 16)
    alpha = params.get('alpha', alpha)
    n_bins = params.get('n_bins', 32)
    activation = params.get('activation', False)
    epochs = params.get('epochs', epochs)
    dropout = params.get('dropout', 0.2)
    n_blocks = params.get('n_blocks', 2)
    class_weighting = params.get('class_weighting', True)
    event_weighting = params.get('event_weighting', False)

    print(f"[CSR] Params: {params}")
    # EVENT TYPE: 0 = censored, 1..K = causes
    events = []
    loe = [np.nan, 'Relapse', 'Toxic Death', 'Induction failure',
        'Second Malignant Neoplasm', 'Death NOS', 'Induction death']
    for et in df_clin[cause_col]:
        if et == 'Induction death':
            et = 'Death NOS'  # Merge these two causes
        events.append(loe.index(et))
    event_type = np.array(events).astype('int32')

    durations = df_clin[days_col].values.astype('float32')
    #event_type = df_clin[cause_col].values.astype('int32')

    K = int(event_type.max())
    labels = labels.values.astype('int64')
    if labels_col in X.columns:
        X = X.drop(columns=[labels_col])

    splits = get_relaxed_stratified_split(X, labels, event_type, n_splits=n_splits, seed=seed)

    NNUM, NCAT, _, _, CATCARDINALITIES, categories = get_data_tabm(X)
    X_num_all, X_cat_all = prepare_tabm_inputs(X, numeric_cols=categories.numeric, categorical_cols=categories.categorical, cat_as_codes=True)

    print(f"Data prepared for TabM: NNUM={NNUM}, NCAT={NCAT}, samples={X.shape[0]}, features={X.shape[1]}")

    cidxs, cidxs_rev, accs = [], [], []
    models = []
    all_risks, all_pred_labels = [], []
    eta =[]

    event_weights = calculate_weights(event_type, device=device)
    class_weights = calculate_weights(labels, device=device)
    print(f"Event weights: {event_weights}")
    print(f"Class weights: {class_weights}")
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\nFold {fold}/{n_splits}")

        if not inference:
            strat_label = make_relaxed_stratification_labels(labels[train_idx], event_type[train_idx], min_per_group=2)
            tr_idx, val_idx = train_test_split(train_idx, test_size=test_size, random_state=seed, stratify=strat_label)

        else:
            tr_idx = train_idx
            val_idx = test_idx

        y_tr = (durations[tr_idx], event_type[tr_idx], labels[tr_idx])
        y_val = (durations[val_idx], event_type[val_idx], labels[val_idx])


        print(f"Fold {fold} event counts by cause (train):")
        for c in range(1, K+1):
            count = np.sum(event_type[train_idx] == c)
            print(f"  Cause {c}: {count} events")

        

        # embeddings
        if NNUM > 0:
            X_train_t = torch.tensor(X_num_all[tr_idx], dtype=torch.float32, device=device)
            num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                rtdl_num_embeddings.compute_bins(
                    X_train_t,
                    n_bins=n_bins,
                ),
                d_embedding=d_embedding,
                activation=activation,
                version='B',
            )
        else:
            num_embeddings = None

        NUM_CLASSES = len(np.unique(labels))
        D_OUT = K + NUM_CLASSES

        if params=={}:
            print("Using default TabM parameters.")
            net_base = tabm.TabM.make(
                n_num_features=NNUM,
                cat_cardinalities=CATCARDINALITIES,
                d_out=D_OUT,
                num_embeddings=num_embeddings,
            ).to(device)
        else:
            net_base = tabm.TabM.make(
                n_num_features=NNUM,
                cat_cardinalities=CATCARDINALITIES,
                d_out=D_OUT,
                num_embeddings=num_embeddings,
                d_block=d_block,
                dropout=dropout,
                n_blocks=n_blocks,
            ).to(device)

        # small init
        with torch.no_grad():
            for p in net_base.output.parameters():
                p.mul_(0.1)

        # wrap TabM
        net = MultiTaskTabMWrapperWithCauses(net_base, num_causes=K).to(device)

        # Create trainer wrapper
        trainer = CoxPH_CompetingRisk_TabM(net=net, K=K, num_classes=NUM_CLASSES, device=device, alpha=alpha, 
                                           class_weights=class_weights if class_weighting else None, event_weights=event_weights if event_weighting else None,)
        
        if load_model is not None:
            print("Loading model weights from provided state_dict...")
            trainer.load_state_dict(load_model['model_state_dict'])

        # prepare inputs for trainer
        X_tr = (None if NNUM == 0 else X_num_all[tr_idx], None if NCAT == 0 else X_cat_all[tr_idx])
        X_val = (None if NNUM == 0 else X_num_all[val_idx], None if NCAT == 0 else X_cat_all[val_idx])
        X_te  = (None if NNUM == 0 else X_num_all[test_idx], None if NCAT == 0 else X_cat_all[test_idx])

        # train (full-batch)
        history = trainer.fit(
            X_tr, y_tr,
            val_input=X_val, val_target=y_val,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            alpha=alpha, tie_method='breslow',
            early_stopping=True, patience=patience, verbose=True, early_stopping_metric='combined',
        )

        # calibrate classification temperature on validation
        trainer.calibrate_temperature(X_val, y_val[-1])

        # fit per-cause horizon calibrators on validation set
        #HORIZONS = [30, 180, 365, 1095]
        #trainer.fit_cif_calibrators(X_val, y_val, horizons=HORIZONS, method='isotonic', min_samples=30)

        # evaluate on test
        eta_test = trainer.predict_eta(X_te)   # (N_test, K)

        # Fix: Clip eta to prevent overflow in exp (which causes sksurv to crash)
        # exp(50) is ~5e21, which fits in float32. exp(89) overflows float32.
        eta_test = np.clip(eta_test, -75, 75)

        risk = np.sum(np.exp(eta_test), axis=1)

        # Fix: Ensure risk is finite and handle NaNs
        if not np.all(np.isfinite(risk)):
            risk = np.nan_to_num(risk, nan=0.0, posinf=1e30, neginf=0.0)


        if SKSURV_AVAILABLE:
            y_test = Surv.from_arrays(event=(event_type[test_idx] > 0).astype(bool), time=durations[test_idx])
            c = concordance_index_censored(y_test['event'], y_test['time'], risk)[0]
            c_rev = concordance_index_censored(y_test['event'], y_test['time'], -risk)[0]
        else:
            # fallback: compute simple concordance via lifelines or skip
            c = float('nan'); c_rev = float('nan')

        cidxs.append(c); cidxs_rev.append(c_rev)
        print(f"Fold {fold} C-index={c:.3f} rev={c_rev:.3f}")

        # classification predictions
        preds = trainer.predict_risk(X_te)   # dict with 'logits'
        logits_test = preds['logits']        # (N_test, C)


        samples = X.index[test_idx]
        #cif_horizons = []
        #for cause in range(K):
        #    for horizon in HORIZONS:
        #        cif_hrzns = trainer.predict_calibrated_cif(X_te, cause_index=cause, horizon=horizon)
        #        cif_horizons.append(pd.DataFrame(cif_hrzns, index=samples, columns=[f"cif_cause_{loe[cause+1]}_horizon_{horizon}"]))
        #cif_horizons = pd.concat(cif_horizons, axis=1)
        #horizons_predictions.append(cif_horizons)

        # apply temperature scaling if calibrated
        if trainer.temperature_calibrated_:
            logits_test = logits_test / trainer.temperature_
        probs = np.exp(logits_test - logits_test.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        pred_labels = probs.argmax(axis=1)
        acc = accuracy_score(labels[test_idx], pred_labels)
        accs.append(acc)
        print(f"Fold {fold} ACC={acc:.3f}")

        # optuna pruning
        if trial is not None:
            trial.report((0.5*c + 0.5*acc), step=fold-1)
            if trial.should_prune():
                print(f"Trial pruned at fold {fold}")
                raise optuna.exceptions.TrialPruned()

        # record outputs
        samples = X.index[test_idx]
        all_risks.append(pd.DataFrame({'sample': samples, 'risk': risk}))
        label_df = pd.DataFrame(probs, index=samples, columns=[f"prob_{l_e.inverse_transform([i])[0]}" for i in range(probs.shape[1])])
        all_pred_labels.append(label_df)
        eta.append(pd.DataFrame(eta_test, index=samples, columns=[f"eta_cause_{loe[i+1]}" for i in range(eta_test.shape[1])]))
        if save:
            models.append(trainer)
        # move net to CPU and free GPU memory
        net.to('cpu')
        if save is False:
            del trainer, net, net_base
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== CV results ===")
    print(f"C-index mean: {np.nanmean(cidxs):.3f} ± {np.nanstd(cidxs):.3f}")
    print(f"ACC mean:     {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    return {
        "cindex_cv": (float(np.nanmean(cidxs)), float(np.nanstd(cidxs))),
        "accuracy_cv": (float(np.mean(accs)), float(np.std(accs))),
        'all_cindexes': cidxs,
        'all_accuracies': accs,
        "all_risks": all_risks,
        "all_pred_labels": all_pred_labels,
        "parameters": params,
        "eta_per_cause": eta,
        "models": models if save else None,
        'label_encoder': l_e,
    }

def hyperparameter_optimisation(data, labels, survival, 
                                hyperparameter_space, optimize_metric_weighting, num_trials, 
                                save_dir, dataset_name='dataset', use_params=False):
    

    print(f"Starting tuning on: {dataset_name} with {data.shape[0]} samples and {data.shape[1]} features.")
    resulsts_dir = save_dir  # add one with default params
    study_name = f'tabm_csr_study_{dataset_name}'
    STORAGE_URL = f'sqlite:///{resulsts_dir}/optuna_studies_{dataset_name}.db'  # Using SQLite for simplicity; replace with your DB URL
    
    # check or create directory
    if not os.path.exists(resulsts_dir):
        os.makedirs(resulsts_dir)

    if use_params == False:
        #create objective function
        def objective(trial, params=hyperparameter_space):
            params = {
                        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                        'd_block': trial.suggest_categorical('d_block', [128, 256, 512]),
                        'd_embedding': trial.suggest_categorical('d_embedding', [8, 12, 16]),
                        'alpha': trial.suggest_float('alpha', 0.1, 0.9),
                        'n_bins': trial.suggest_categorical('n_bins', [16, 32, 64]),
                        'activation': trial.suggest_categorical('activation', [True, False]),
                        'epochs': trial.suggest_categorical('epochs', [1000]),
                        'dropout': trial.suggest_float('dropout', 0.05, 0.25),
                        'n_blocks': trial.suggest_categorical('n_blocks', [2, 3, 4]),
                        'class_weighting': trial.suggest_categorical('class_weighting', [True, False]),
                        'event_weighting': trial.suggest_categorical('event_weighting', [True, False]),
                    } if params is None else params
            results = fit_tabm_csr(data, labels, survival, params=params, n_splits=3, trial=trial)
            c_index = results['cindex_cv'][0]
            accuracy = results['accuracy_cv'][0]
            return optimize_metric_weighting * c_index + (1 - optimize_metric_weighting) * accuracy

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
                #pruner = None  # Disable pruning for testing
                
        try:
            study = optuna.create_study(
                        study_name=study_name,
                        storage=STORAGE_URL,         # <-- Tells Optuna where to save the state
                        direction='maximize',
                        pruner=pruner,
                        load_if_exists=True          # <-- CRUCIAL: Loads study if it exists
                    )
            print(f"Study '{study_name}' loaded or created. Current trials: {len(study.trials)}")
        except Exception as e:
            print(f"Failed to connect to storage: {e}")
            # Fallback to in-memory study if DB connection fails (not resumable)
            study = optuna.create_study(direction='maximize', pruner=pruner)

        n_trials = num_trials

        study.optimize(objective, n_trials=n_trials)
        study.trials_dataframe().to_csv(f'{resulsts_dir}/optuna_study_results_{dataset_name}.csv')

        print("Best trial:")
        trial = study.best_trial
        print(trial.value, trial.params)
        params = trial.params
    
    else:
        params = use_params
        
    best_results = fit_tabm_csr(data, labels, survival, params=params, n_splits=5, seed_i = 0)#
    return best_results, params
    

def fit_and_validate(dataset_train, dataset_test, name, common_samples, labels, survival,
                     save_dir):
       

            dataset_test.columns = dataset_train.columns  # align columns

            print(f"Training samples: {dataset_train.shape}, Test samples: {dataset_test.shape}")
           
            best_params = {
            'activation': False,
            'alpha': 0.10049254277748548,
            'd_block': 256,
            'd_embedding': 32,
            'dropout': 0.15598660989523558,
            'epochs': 1000,
            'lr': 0.0032687665714687124,
            'n_bins': 64,
            'weight_decay': 0.00011976371067215448,
            'n_blocks': 2}
            print('Datasets for inference loaded.')

            # proceed with model training on dataset_train and inference on dataset_test
            results = fit_tabm_csr(dataset_train, labels.loc[dataset_train.index], survival.loc[dataset_train.index], 
                                params=best_params, n_splits=5, epochs=1, save=True, inference=True)
            l_e = results['label_encoder']

            print("Model training completed. Proceeding with inference on test set...")


            scaled_logits_agg = []
            eta_agg = []
            risk_score_agg = []
            total_risk_agg = []
            probs_agg = []

            # Pre-process test data ONCE outside the loop for efficiency
            NNUM, NCAT, _, _, CATCARDINALITIES, categories = get_data_tabm(dataset_test)
            categories.numeric = categories.numeric + categories.categorical
            categories.categorical = []
            X_num_test, X_cat_test = prepare_tabm_inputs(dataset_test, numeric_cols=categories.numeric, categorical_cols=categories.categorical, cat_as_codes=True)
            X_te  = (None if NNUM == 0 else X_num_test, None)
            print(f"Prepared test inputs for model inference. NNUM={NNUM}, NCAT={NCAT}")

            for model in results['models']:
                # predict
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.net.to(device)
                
                # Get all predictions
                preds = model.predict_all(X_te, horizons=[30, 180, 365, 1095])
                
                # 1. Classification: Use Softmax Probabilities
                logits_scaled = torch.tensor(preds['scaled_logits'])
                probs = torch.softmax(logits_scaled, dim=1).numpy()
                probs_agg.append(probs)


                # --- Collect all outputs ---

                scaled_logits_agg.append(preds['scaled_logits'])
                eta_agg.append(preds['eta'])
                risk_score_agg.append(preds['risk_score'])
                total_risk_agg.append(preds['total_risk_score'])

                model.net.to('cpu')
                del model
                gc.collect()
                torch.cuda.empty_cache()

            # --- Aggregate (Mean across folds) ---
            
            # 1. Classification
            probs_final = np.mean(np.array(probs_agg), axis=0)

            scaled_logits_final = np.mean(np.array(scaled_logits_agg), axis=0)
            
            # 2. Survival Risk
            # Average the Risk Score (Hazard Ratio) for ensembles
            risk_score_final = np.mean(np.array(risk_score_agg), axis=0)
            total_risk_final = np.mean(np.array(total_risk_agg), axis=0)
            
            # Calculate Eta from averaged risk (log-scale)
            eta_final = np.log(risk_score_final + 1e-9)

        
            # --- Save to CSV ---
            
            loe = [np.nan, 'Relapse', 'Toxic Death', 'Induction failure',
                'Second Malignant Neoplasm', 'Death NOS', 'Induction death']
            
            # 1. Probabilities
            pd.DataFrame(probs_final, index=dataset_test.index, 
                        columns=[f"prob_{l_e.inverse_transform([i])[0]}" for i in range(probs_final.shape[1])]
                        ).to_csv(f'{save_dir}/predictions_probs_{name}.csv', index=True)
            # 2. Scaled Logits
            pd.DataFrame(scaled_logits_final, index=dataset_test.index,
                        columns=[f"scaled_logit_{l_e.inverse_transform([i])[0]}" for i in range(scaled_logits_final.shape[1])]
                        ).to_csv(f'{save_dir}/predictions_scaled_logits_{name}.csv', index=True)
            # 3. Eta (Log Risk)
            pd.DataFrame(eta_final, index=dataset_test.index, 
                        columns=[f"eta_cause_{loe[i+1]}" for i in range(eta_final.shape[1])]
                        ).to_csv(f'{save_dir}/eta_per_cause_{name}.csv', index=True)
            # 4. Risk Scores (Hazard Ratios) & Total Risk
            df_risk = pd.DataFrame(risk_score_final, index=dataset_test.index, 
                                columns=[f"risk_score_cause_{loe[i+1]}" for i in range(risk_score_final.shape[1])])
            df_risk['total_risk_score'] = total_risk_final
            df_risk.to_csv(f'{save_dir}/risk_scores_{name}.csv', index=True)


def fit_tabm_csr_user_model(
    labels: pd.Series,
    df_clin: pd.DataFrame,
    X_gene: Optional[pd.DataFrame] = None,
    X_other: Optional[pd.DataFrame] = None,
    gene_preprocessor: Optional[Any] = None,
    # Clinical data columns
    days_col: str = 'EFS_days',
    cause_col: str = 'Event_Type',
    labels_col: str = 'subtype',
    # CV settings
    n_splits: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
    seed_offset: int = 0,
    # Training hyperparameters
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 3e-4,
    patience: Optional[int] = None,
    # Model architecture
    d_block: int = 512,
    d_embedding: int = 16,
    n_blocks: int = 2,
    dropout: float = 0.2,
    n_bins: int = 32,
    activation: bool = False,
    # Loss weighting
    alpha: float = 0.15,
    class_weighting: bool = True,
    event_weighting: bool = False,
    # Other options
    params: Optional[Dict] = None,
    trial: Optional[Any] = None,
    save_models: bool = False,
    inference_mode: bool = False,
    load_model: Optional[Dict] = None,
    # Cause mapping
    cause_mapping: Optional[Dict] = None,
)-> Dict:
    """
    Train TabM model with competing risks on gene expression + other data.
    
    Args:
        labels: Target labels for classification
        df_clin: Clinical data with survival information
        X_gene: Gene expression data (will be preprocessed), optional
        X_other: Other data (no preprocessing, e.g., categorical/one-hot), optional
        gene_preprocessor: sklearn-style preprocessor for gene expression (e.g., StandardScaler), optional
        days_col: Column name for survival times
        cause_col: Column name for event types
        labels_col: Column name for labels (to exclude from features)
        n_splits: Number of CV folds
        test_size: Validation set size
        seed: Random seed
        seed_offset: Offset for seed (for multiple runs)
        epochs: Training epochs
        batch_size: Batch size (currently unused, full-batch training)
        lr: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience (auto if None)
        d_block: Hidden dimension
        d_embedding: Embedding dimension
        n_blocks: Number of blocks
        dropout: Dropout rate
        n_bins: Number of bins for numerical embeddings
        activation: Use activation in embeddings
        alpha: Weight for classification loss (1-alpha for survival)
        class_weighting: Use class weights
        event_weighting: Use event weights
        params: Parameter dict to override defaults
        trial: Optuna trial
        save_models: Save trained models
        inference_mode: Use train set for validation
        load_model: Pre-trained model
        cause_mapping: Custom event type mapping

        
    Returns:
        Dictionary with CV results, predictions, and models
    """
    # check input for nans, infs:
    if X_gene is not None:
        if not np.isfinite(X_gene.values).all():
            raise ValueError("X_gene contains NaN or infinite values")
        if X_gene.isna().any().any():
            raise ValueError("X_gene contains NaN values")
    if X_other is not None:
        if not np.isfinite(X_other.values).all():
            raise ValueError("X_other contains NaN or infinite values")
        if X_other.isna().any().any():
            raise ValueError("X_other contains NaN values")

    # Validate required imports
    required = [get_relaxed_stratified_split, get_data_tabm, prepare_tabm_inputs,
                make_relaxed_stratification_labels, tabm, rtdl_num_embeddings,
                MultiTaskTabMWrapperWithCauses, CoxPH_CompetingRisk_TabM]
    if any(imp is None for imp in required):
        raise ValueError("All required function/module imports must be provided")
    
    # Validate input data
    if X_gene is None and X_other is None:
        raise ValueError("At least one of X_gene or X_other must be provided")
    
    # Validate index alignment
    if X_gene is not None and X_other is not None:
        if not X_gene.index.equals(X_other.index):
            raise ValueError("X_gene and X_other must have the same index")
        reference_index = X_gene.index
    elif X_gene is not None:
        reference_index = X_gene.index
    else:
        reference_index = X_other.index
    
    if not reference_index.equals(labels.index):
        raise ValueError("Feature data and labels must have the same index")
    

    if gene_preprocessor is None and X_gene is not None:
        from src.helpers.normalizers import LogCPMNormalizer
        gene_preprocessor = LogCPMNormalizer()

    # Validate gene preprocessor
    if X_gene is not None and gene_preprocessor is None:
        raise ValueError("gene_preprocessor must be provided when X_gene is provided")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Override parameters if provided
    if params is not None:
        lr = params.get('lr', lr)
        weight_decay = params.get('weight_decay', weight_decay)
        d_block = params.get('d_block', d_block)
        d_embedding = params.get('d_embedding', d_embedding)
        alpha = params.get('alpha', alpha)
        n_bins = params.get('n_bins', n_bins)
        activation = params.get('activation', activation)
        epochs = params.get('epochs', epochs)
        dropout = params.get('dropout', dropout)
        n_blocks = params.get('n_blocks', n_blocks)
        class_weighting = params.get('class_weighting', class_weighting)
        event_weighting = params.get('event_weighting', event_weighting)
    
    # Load model parameters if provided
    if load_model is not None:
        params = load_model['parameters']
        lr = params.get('lr', lr)
        weight_decay = params.get('weight_decay', weight_decay)
        d_block = params.get('d_block', d_block)
        d_embedding = params.get('d_embedding', d_embedding)
        alpha = params.get('alpha', alpha)
        n_bins = params.get('n_bins', n_bins)
        activation = params.get('activation', activation)
        epochs = params.get('epochs', epochs)
        dropout = params.get('dropout', dropout)
        n_blocks = params.get('n_blocks', n_blocks)
    
    # Set patience
    if patience is None:
        patience = max(min(epochs // 10, 100), 10)
    
    # Adjust seed
    seed = seed + seed_offset + 5
    
    print(f"[CSR] Training parameters:")
    print(f"  epochs={epochs}, lr={lr}, weight_decay={weight_decay}")
    print(f"  d_block={d_block}, d_embedding={d_embedding}, n_blocks={n_blocks}, dropout={dropout}")
    print(f"  alpha={alpha}, n_bins={n_bins}, activation={activation}")
    print(f"  class_weighting={class_weighting}, event_weighting={event_weighting}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = pd.Series(
        label_encoder.fit_transform(labels.values.squeeze()), 
        index=labels.index
    )
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    
    # Process competing risks data
    if cause_mapping is None:
        cause_mapping = {
            np.nan: 0,
            'Relapse': 1,
            'Toxic Death': 2,
            'Induction failure': 3,
            'Second Malignant Neoplasm': 4,
            'Death NOS': 5,
            'Induction death': 5  # Merge with Death NOS
        }
    
    events = []
    for et in df_clin[cause_col]:
        events.append(cause_mapping.get(et, 0))
    event_type = np.array(events).astype('int32')
    durations = df_clin[days_col].values.astype('float32')
    
    K = int(event_type.max())
    print(f"\nNumber of competing causes: {K}")
    for c in range(K + 1):
        count = np.sum(event_type == c)
        cause_name = "censored" if c == 0 else f"cause {c}"
        print(f"  {cause_name}: {count} events")
    
    # Remove labels column if present
    if X_gene is not None and labels_col in X_gene.columns:
        X_gene = X_gene.drop(columns=[labels_col])
    if X_other is not None and labels_col in X_other.columns:
        X_other = X_other.drop(columns=[labels_col])
    
    print(f"\nData shapes:")
    if X_gene is not None:
        print(f"  X_gene: {X_gene.shape}")
    if X_other is not None:
        print(f"  X_other: {X_other.shape}")
    
    # Generate CV splits (on combined data for stratification)
    # Combine available data
    data_parts = []
    if X_gene is not None:
        data_parts.append(X_gene)
    if X_other is not None:
        data_parts.append(X_other)
    X_combined = pd.concat(data_parts, axis=1)
    labels_array = labels_encoded.values.astype('int64')
    
    splits = get_relaxed_stratified_split(
        X_combined, labels_array, event_type, 
        n_splits=n_splits, seed=seed
    )
    
    # Get TabM data format
    NNUM, NCAT, _, _, CATCARDINALITIES, categories = get_data_tabm(X_combined)
    
    event_weights = calculate_weights(event_type, device) if event_weighting else None
    class_weights = calculate_weights(labels_array, device) if class_weighting else None
    
    if event_weights is not None:
        print(f"\nEvent weights: {event_weights}")
    if class_weights is not None:
        print(f"Class weights: {class_weights}")
    
    # Storage for results
    results = {
        'cindexes': [],
        'accuracies': [],
        'all_risks': [],
        'all_pred_labels': [],
        'eta_per_cause': [],
        'models': [] if save_models else None,
        'gene_preprocessors': [],
    }
    
    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_splits}")
        print(f"{'='*60}")
        
        # Split train into train/val
        if not inference_mode:
            strat_label = make_relaxed_stratification_labels(
                labels_array[train_idx], 
                event_type[train_idx], 
                min_per_group=2
            )
            tr_idx, val_idx = train_test_split(
                train_idx, 
                test_size=test_size, 
                random_state=seed, 
                stratify=strat_label
            )
        else:
            tr_idx = train_idx
            val_idx = test_idx
        
        # Preprocess gene expression data if provided
        if X_gene is not None:
            # Fit on training data only
            gene_preprocessor.fit(X_gene.iloc[tr_idx])
            

            # Transform all splits
            X_gene_tr = pd.DataFrame(
                gene_preprocessor.transform(X_gene.iloc[tr_idx]),
                index=X_gene.iloc[tr_idx].index,
                columns=X_gene.columns
            )
            X_gene_val = pd.DataFrame(
                gene_preprocessor.transform(X_gene.iloc[val_idx]),
                index=X_gene.iloc[val_idx].index,
                columns=X_gene.columns
            )
            X_gene_te = pd.DataFrame(
                gene_preprocessor.transform(X_gene.iloc[test_idx]),
                index=X_gene.iloc[test_idx].index,
                columns=X_gene.columns
            )

            # check processed for nan/infs
            if not np.isfinite(X_gene_tr.values).all():
                #print(X_gene_tr.head())
                #print(X_gene_tr.describe())
                #print('Columns with NaNs:', X_gene_tr.columns[X_gene_tr.isna().any()].tolist())
                #print('Columns with Infs:', X_gene_tr.columns[np.isinf(X_gene_tr.values).any(axis=0)].tolist())
                #print('Rows with NaNs:', X_gene_tr.index[X_gene_tr.isna().any(axis=1)].tolist())
                #print('Rows with Infs:', X_gene_tr.index[np.isinf(X_gene_tr.values).any(axis=1)].tolist())
                raise ValueError("Processed X_gene_tr contains NaN or infinite values")
            if not np.isfinite(X_gene_val.values).all():
                raise ValueError("Processed X_gene_val contains NaN or infinite values")
            if not np.isfinite(X_gene_te.values).all():
                raise ValueError("Processed X_gene_te contains NaN or infinite values")
            if X_gene_tr.isna().any().any():
                raise ValueError("Processed X_gene_tr contains NaN values")
            if X_gene_val.isna().any().any():
                raise ValueError("Processed X_gene_val contains NaN values")
            if X_gene_te.isna().any().any():
                raise ValueError("Processed X_gene_te contains NaN values")
            # get min and max for debug
            print(f"X_gene_tr min: {X_gene_tr.values.min()}, max: {X_gene_tr.values.max()}")
            print(f"X_gene_val min: {X_gene_val.values.min()}, max: {X_gene_val.values.max()}")
            print(f"X_gene_te min: {X_gene_te.values.min()}, max: {X_gene_te.values.max()}")
        
        # Combine gene + other data
        data_parts_tr, data_parts_val, data_parts_te = [], [], []
        
        if X_gene is not None:
            data_parts_tr.append(X_gene_tr)
            data_parts_val.append(X_gene_val)
            data_parts_te.append(X_gene_te)
        
        if X_other is not None:
            data_parts_tr.append(X_other.iloc[tr_idx])
            data_parts_val.append(X_other.iloc[val_idx])
            data_parts_te.append(X_other.iloc[test_idx])
        
        X_tr_combined = pd.concat(data_parts_tr, axis=1)
        X_val_combined = pd.concat(data_parts_val, axis=1)
        X_te_combined = pd.concat(data_parts_te, axis=1)
        
        # Prepare targets
        y_tr = (durations[tr_idx], event_type[tr_idx], labels_array[tr_idx])
        y_val = (durations[val_idx], event_type[val_idx], labels_array[val_idx])

        
        print(f"TabM format: NNUM={NNUM}, NCAT={NCAT}, samples={X_tr_combined.shape[0]}, features={X_tr_combined.shape[1]}")
        print(f'Unique values in CATCARDINALITIES: {set(CATCARDINALITIES)}')
        
        # Prepare TabM inputs
        X_num_tr, X_cat_tr = prepare_tabm_inputs(
            X_tr_combined, 
            numeric_cols=categories.numeric, 
            categorical_cols=categories.categorical, 
            cat_as_codes=True
        )
        X_num_val, X_cat_val = prepare_tabm_inputs(
            X_val_combined,
            numeric_cols=categories.numeric,
            categorical_cols=categories.categorical,
            cat_as_codes=True
        )
        X_num_te, X_cat_te = prepare_tabm_inputs(
            X_te_combined,
            numeric_cols=categories.numeric,
            categorical_cols=categories.categorical,
            cat_as_codes=True
        )
        
        # Create embeddings
        num_embeddings = None
        if NNUM > 0:
            X_train_t = torch.tensor(X_num_tr, dtype=torch.float32, device=device)
            num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                rtdl_num_embeddings.compute_bins(X_train_t, n_bins=n_bins),
                d_embedding=d_embedding,
                activation=activation,
                version='B',
            )
        
        # Build model
        D_OUT = K + num_classes
        
        net_base = tabm.TabM.make(
            n_num_features=NNUM,
            cat_cardinalities=CATCARDINALITIES,
            d_out=D_OUT,
            num_embeddings=num_embeddings,
            d_block=d_block,
            dropout=dropout,
            n_blocks=n_blocks,
        ).to(device)
        
        # Initialize output layer with small weights
        with torch.no_grad():
            for p in net_base.output.parameters():
                p.mul_(0.1)
        
        # Wrap model
        net = MultiTaskTabMWrapperWithCauses(net_base, num_causes=K).to(device)
        
        # Create trainer
        trainer = CoxPH_CompetingRisk_TabM(
            net=net, 
            K=K, 
            num_classes=num_classes, 
            device=device, 
            alpha=alpha,
            class_weights=class_weights,
            event_weights=event_weights,
        )
        
        # Load pre-trained weights if provided
        if load_model is not None:
            print("Loading model weights...")
            trainer.load_state_dict(load_model['model_state_dict'])
        
        # Prepare trainer inputs
        X_tr_tuple = (X_num_tr if NNUM > 0 else None, X_cat_tr if NCAT > 0 else None)
        X_val_tuple = (X_num_val if NNUM > 0 else None, X_cat_val if NCAT > 0 else None)
        X_te_tuple = (X_num_te if NNUM > 0 else None, X_cat_te if NCAT > 0 else None)
        
        print(f"  Input tuples prepared: num is None={X_tr_tuple[0] is None}, cat is None={X_tr_tuple[1] is None}")
        # Train
        history = trainer.fit(
            X_tr_tuple, y_tr,
            val_input=X_val_tuple, val_target=y_val,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            alpha=alpha,
            tie_method='breslow',
            early_stopping=True,
            patience=patience,
            verbose=True,
            early_stopping_metric='combined',
        )
        
        # Calibrate temperature on validation set
        trainer.calibrate_temperature(X_val_tuple, y_val[-1])
        
        # Evaluate on test set
        eta_test = trainer.predict_eta(X_te_tuple)
        eta_test = np.clip(eta_test, -75, 75)  # Prevent overflow
        
        risk = np.sum(np.exp(eta_test), axis=1)
        risk = np.nan_to_num(risk, nan=0.0, posinf=1e30, neginf=0.0)
        
        # Calculate concordance index
        if SKSURV_AVAILABLE:
            y_test_surv = Surv.from_arrays(
                event=(event_type[test_idx] > 0).astype(bool),
                time=durations[test_idx]
            )
            cindex = concordance_index_censored(
                y_test_surv['event'], 
                y_test_surv['time'], 
                risk
            )[0]
        else:
            cindex = float('nan')
        
        results['cindexes'].append(cindex)
        print(f"Fold {fold} C-index: {cindex:.4f}")
        
        # Classification predictions
        preds = trainer.predict_risk(X_te_tuple)
        logits_test = preds['logits']
        
        # Apply temperature scaling
        if trainer.temperature_calibrated_:
            logits_test = logits_test / trainer.temperature_
        
        # Compute probabilities
        logits_max = logits_test.max(axis=1, keepdims=True)
        probs = np.exp(logits_test - logits_max)
        probs = probs / probs.sum(axis=1, keepdims=True)
        pred_labels = probs.argmax(axis=1)
        
        accuracy = accuracy_score(labels_array[test_idx], pred_labels)
        results['accuracies'].append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        
        # Optuna pruning
        if trial is not None:
            combined_metric = 0.5 * cindex + 0.5 * accuracy
            trial.report(combined_metric, step=fold - 1)
            if trial.should_prune():
                print(f"Trial pruned at fold {fold}")
                raise trial.__class__.TrialPruned()
        
        # Store predictions
        samples = X_combined.index[test_idx]
        
        results['all_risks'].append(
            pd.DataFrame({'sample': samples, 'risk': risk})
        )
        
        prob_cols = [f"prob_{label_encoder.inverse_transform([i])[0]}" 
                     for i in range(probs.shape[1])]
        results['all_pred_labels'].append(
            pd.DataFrame(probs, index=samples, columns=prob_cols)
        )
        
        eta_cols = [f"eta_cause_{i+1}" for i in range(eta_test.shape[1])]
        results['eta_per_cause'].append(
            pd.DataFrame(eta_test, index=samples, columns=eta_cols)
        )
        
        # Save model if requested
        if save_models:
            trainer.label_encoder_ = label_encoder
            trainer.gene_preprocessor_ = gene_preprocessor
            trainer.cause_mapping_ = cause_mapping
            results['models'].append(trainer)
            #results['gene_preprocessors'].append(gene_preprocessor)
        
        # Cleanup
        net.to('cpu')
        if not save_models:
            del trainer, net, net_base
        gc.collect()
        torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Cross-Validation Results")
    print(f"{'='*60}")
    print(f"C-index:  {np.nanmean(results['cindexes']):.4f} ± "
          f"{np.nanstd(results['cindexes']):.4f}")
    print(f"Accuracy: {np.mean(results['accuracies']):.4f} ± "
          f"{np.std(results['accuracies']):.4f}")
    
    # Store final parameters
    final_params = {
        'lr': lr, 'weight_decay': weight_decay, 'd_block': d_block,
        'd_embedding': d_embedding, 'alpha': alpha, 'n_bins': n_bins,
        'activation': activation, 'epochs': epochs, 'dropout': dropout,
        'n_blocks': n_blocks, 'class_weighting': class_weighting,
        'event_weighting': event_weighting
    }
    
    return {
        'cindex_cv': (float(np.nanmean(results['cindexes'])), 
                     float(np.nanstd(results['cindexes']))),
        'accuracy_cv': (float(np.mean(results['accuracies'])), 
                       float(np.std(results['accuracies']))),
        'all_cindexes': results['cindexes'],
        'all_accuracies': results['accuracies'],
        'all_risks': results['all_risks'],
        'all_pred_labels': results['all_pred_labels'],
        'eta_per_cause': results['eta_per_cause'],
        'parameters': final_params,
        'models': results['models'],
        'label_encoder': label_encoder,
        #'gene_preprocessors': results['gene_preprocessors'] if X_gene is not None else None,
    }

def predict_with_user_model(
    models: Any = None,
    #params: Dict,
    #gene_preprocessors: Optional[Any] = None,
    #df_clin: pd.DataFrame,
    X_gene: Optional[pd.DataFrame] = None,
    X_other: Optional[pd.DataFrame] = None,
    days_col: str = 'EFS_days',
    cause_col: str = 'Event_Type',
    labels_col: str = 'subtype',
) -> Dict:
    """
    Make predictions with a trained TabM competing risks model.
    
    Args:
        model: Trained TabM model
        params: Model parameters
        gene_preprocessor: Preprocessor for gene expression data
        df_clin: Clinical data
        X_gene: Gene expression data, optional
        X_other: Other data, optional
        days_col: Column name for survival times
        cause_col: Column name for event types
        labels_col: Column name for labels (to exclude from features)
        
    Returns:
        Dictionary with predictions
    """
    if models is None:
        raise ValueError("models must be provided")
    # Validate input for nans, infs:
    if X_gene is not None:
        if not np.isfinite(X_gene.values).all():
            raise ValueError("X_gene contains NaN or infinite values")
        if X_gene.isna().any().any():
            raise ValueError("X_gene contains NaN values")
    if X_other is not None:
        if not np.isfinite(X_other.values).all():
            raise ValueError("X_other contains NaN or infinite values")
        if X_other.isna().any().any():
            raise ValueError("X_other contains NaN values")
    # Validate input data
    if X_gene is None and X_other is None:
        raise ValueError("At least one of X_gene or X_other must be provided")
    
    # Validate index alignment
    if X_gene is not None and X_other is not None:
        if not X_gene.index.equals(X_other.index):
            raise ValueError("X_gene and X_other must have the same index")
        reference_index = X_gene.index
    elif X_gene is not None:
        reference_index = X_gene.index
    else:
        reference_index = X_other.index
    

    
    if not np.all(np.equal(np.mod(X_gene.values, 1), 0)):
                raise ValueError("Input count data must be integers - Test in training_funcs.py")

    results = {
        'scaled_logits': None,
        'probs': None,
        'eta': None,
        'risk_score': None,
        'total_risk_score': None,
    }
    
    for model in models:
        gene_preprocessor = model.gene_preprocessor_
        X_gene_m = pd.DataFrame(gene_preprocessor.transform(X_gene), index=X_gene.index, columns=X_gene.columns) if X_gene is not None else None

        # Combine data
        data_parts = []
        if X_gene_m is not None:
            data_parts.append(X_gene_m)
        if X_other is not None:
            data_parts.append(X_other)
        X_combined = pd.concat(data_parts, axis=1)

        # Prepare TabM inputs
        NNUM, NCAT, _, _, CATCARDINALITIES, categories = get_data_tabm(X_combined)
        X_num, X_cat = prepare_tabm_inputs(X_combined, 
                                           numeric_cols=categories.numeric, 
                                           categorical_cols=categories.categorical, 
                                           cat_as_codes=True)
        X_te_tuple = (X_num if NNUM > 0 else None, X_cat if NCAT > 0 else None)
        print(f"Prepared inputs for prediction. NNUM={NNUM}, NCAT={NCAT}")
        # Make predictions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.net.to(device)
        
        preds = model.predict_all(X_te_tuple, horizons=[30, 180, 365, 1095])
        
        # Collect outputs
        scaled_logits = torch.tensor(preds['scaled_logits'])
        probs = torch.softmax(scaled_logits, dim=1).numpy()
        eta = preds['eta']
        risk_score = preds['risk_score']
        total_risk_score = preds['total_risk_score']
        results['scaled_logits'] = results['scaled_logits'] + scaled_logits if results['scaled_logits'] is not None else scaled_logits
        results['probs'] = results['probs'] + probs if results['probs'] is not None else probs
        results['eta'] = results['eta'] + eta if results['eta'] is not None else eta
        results['risk_score'] = results['risk_score'] + risk_score if results['risk_score'] is not None else risk_score
        results['total_risk_score'] = results['total_risk_score'] + total_risk_score if results['total_risk_score'] is not None else total_risk_score
        
        label_encoder = model.label_encoder_
        cause_mapping = model.cause_mapping_
        model.net.to('cpu')
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Average over models
    cause_mapping = {v: k for k, v in cause_mapping.items() if k != 'Induction death'}  # Invert mapping, exclude censored
    num_models = len(models)
    results['scaled_logits'] = results['scaled_logits'] / num_models
    results['probs'] = results['probs'] / num_models
    results['eta'] = results['eta'] / num_models
    results['risk_score'] = results['risk_score'] / num_models
    results['total_risk_score'] = results['total_risk_score'] / num_models

    for key, value in results.items():
        results[key] = pd.DataFrame(value, index=X_combined.index)
        if key == 'probs':
            results[key].columns = [f"prob_{label_encoder.inverse_transform([i])[0]}" for i in range(results[key].shape[1])]
        if key == 'eta':
            results[key].columns = [f"eta_cause_{cause_mapping[i+1]}" for i in range(results[key].shape[1])]
    return results