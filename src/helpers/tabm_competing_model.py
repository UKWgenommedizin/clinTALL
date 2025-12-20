import time
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score # Added

# Try importing sksurv for C-index
try:
    from sksurv.metrics import concordance_index_censored
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False

# import helpers from the drop-in module you saved earlier
from src.helpers.tabm_competing import (
    cox_ph_loss_tabm,
    reduce_loghaz_to_eta,
    compute_breslow_baseline_hazards,
    predict_cif_from_breslow,
    fit_calibrator_on_predicted_cif,
    predict_calibrated_cif,
    CompetingRiskTabMLoss
)

# Optional: a small progress print helper
def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class CoxPH_CompetingRisk_TabM:
    """
    Wrapper for TabM backbone + cause-specific Cox heads.

    Parameters
    ----------
    net : torch.nn.Module
        A model that on forward(num_tensor, cat_tensor) returns (log_h, logits),
        where log_h shape is (B, H, K) or (B, K) and logits shape is (B, H, C) or (B, C).
    K : int
        Number of competing causes.
    num_classes : int
        Number of classes for classification head.
    device : str or torch.device
        Device to run training/prediction on.
    """

    def __init__(self, net, K: int, num_classes: int, device=None, class_weights=None, event_weights=None, alpha=0.4):
        self.net = net
        self.K = int(K)
        self.num_classes = int(num_classes)
        self.device = torch.device(device) if device is not None else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        # move model to device
        self.net.to(self.device)

        # storage for baseline hazards and calibrators
        self.event_times_ = None
        self.H0_ = None
        self.dH0_ = None
        # dict keyed by (horizon, cause_index) -> calibrator object
        self.horizon_calibrators_ = {}
        # temperature for classification calibration
        self.temperature_ = 1.0
        self.temperature_calibrated_ = False
        self.class_weights = class_weights
        self.event_weights = event_weights
        self.loss_fn = CompetingRiskTabMLoss(K, alpha=alpha, class_weights=class_weights, event_weights=event_weights)
        # keep last training history (list of dicts)
        self.history_ = []

    # ---------------------------
    # internal helpers
    # ---------------------------
    def _forward_full(self, input_tuple):
        """Forward pass on entire dataset returning log_h (torch), logits (torch)."""
        X_num, X_cat = input_tuple
        # convert to tensors (allow None)
        if X_num is None:
            num_t = None
        else:
            num_t = torch.as_tensor(X_num, dtype=torch.float32, device=self.device)
        if X_cat is None:
            cat_t = None
        else:
            cat_t = torch.as_tensor(X_cat, dtype=torch.int64, device=self.device)

        # forward through net (may return tuple)
        self.net.train()  # ensure layers like dropout active if desired (we use full-batch gradient)
        log_h, logits = self.net(num_t, cat_t)
        # ensure returned tensors are on device and floats
        if isinstance(log_h, (list, tuple)):
            log_h = torch.stack(log_h, dim=0)
        log_h = log_h.to(self.device)
        logits = logits.to(self.device)
        return log_h, logits

    def _eval_forward_numpy(self, input_tuple):
        """
        Eval-mode forward that returns numpy arrays (averaged heads by taking mean across head dim if present).
        Returns: (eta_numpy (N,K), logits_numpy (N,C))
        """
        self.net.eval()
        X_num, X_cat = input_tuple
        with torch.no_grad():
            num_t = None if X_num is None else torch.as_tensor(X_num, dtype=torch.float32, device=self.device)
            cat_t = None if X_cat is None else torch.as_tensor(X_cat, dtype=torch.int64, device=self.device)
            preds = self.net(num_t, cat_t)
            log_h = preds[0]
            logits = preds[1]
            # average heads if present
            if torch.is_tensor(log_h) and log_h.ndim == 3:
                eta_t = log_h.mean(dim=1)
            else:
                eta_t = log_h
            if torch.is_tensor(logits) and logits.ndim == 3:
                logits_t = logits.mean(dim=1)
            else:
                logits_t = logits
            eta_np = eta_t.detach().cpu().numpy()
            logits_np = logits_t.detach().cpu().numpy()
        return eta_np, logits_np

    # ---------------------------
    # training
    # ---------------------------
    def fit(self,
            train_input,
            train_target,
            val_input=None,
            val_target=None,
            epochs=200,
            lr=1e-3,
            weight_decay=1e-4,
            alpha=0.4,
            tie_method='breslow',
            early_stopping=True,
            patience=20,
            early_stopping_metric='loss', # Options: 'loss', 'combined'
            verbose=True):
        """
        Train the model using full-batch Cox partial likelihood + classification loss.

        Parameters
        ----------
        train_input: tuple (X_num, X_cat)
        train_target: tuple (durations, event_type, labels)
            event_type must be integers where 0 = censored, 1..K = cause index.
        val_input / val_target: optional validation set used for early stopping and calibrations
        epochs: number of epochs
        lr, weight_decay: optimizer params
        alpha: mixing weight (alpha * survival_loss + (1-alpha) * classification_loss)
        tie_method: 'breslow' or 'efron' for Cox loss
        early_stopping: whether to use validation early stopping
        patience: patience for early stopping
        early_stopping_metric: 'loss' (minimize val_loss) or 'combined' (maximize val_acc + val_cindex)
        """
        # unpack
        X_num_tr, X_cat_tr = train_input
        durations_tr, event_type_tr, labels_tr = train_target

        N_train = len(durations_tr)
        if verbose:
            print(f"[{_now()}] Starting training: N_train={N_train}, epochs={epochs}, device={self.device}")

        # move to device for full-batch gradients
        # We'll perform full-batch updates each epoch for correct risk sets.
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        # Initialize tracking variables
        best_val_loss = float('inf')
        best_val_score = float('-inf') # For combined metric (maximize)
        best_state = None
        epochs_no_improve = 0
        history = []

        for epoch in range(1, epochs + 1):
            self.net.train()
            # forward full training set (returns raw log_h, logits as tensors)
            #preds = self._forward_full((X_num_tr, X_cat_tr))
            log_h_t, logits_t = self._forward_full((X_num_tr, X_cat_tr))
            # average heads if necessary for survival
            if torch.is_tensor(log_h_t) and log_h_t.ndim == 3:
                eta = log_h_t.mean(dim=1)  # (N, K)
            else:
                eta = log_h_t

            # classification logits averaged
            if torch.is_tensor(logits_t) and logits_t.ndim == 3:
                logits_mean = logits_t.mean(dim=1)
            else:
                logits_mean = logits_t

            # ensure durations and event_type tensors on device
            durations_tensor = torch.as_tensor(durations_tr, dtype=torch.float32, device=self.device)
            event_type_tensor = torch.as_tensor(event_type_tr, dtype=torch.long, device=self.device)
            labels_tensor = torch.as_tensor(labels_tr, dtype=torch.long, device=self.device)

            # compute survival loss: sum over causes
            loss_surv = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            for c in range(self.K):
                # event mask for cause c+1 (1..K)
                event_mask_c = (event_type_tensor == (c + 1)).long()
                scores_c = eta[:, c]
                loss_c = cox_ph_loss_tabm(scores_c, durations_tensor, event_mask_c, tie_method=tie_method, reduction='mean')
                if self.event_weights is not None:
                    loss_surv = loss_surv + loss_c * self.event_weights[c + 1]
                else:
                    loss_surv = loss_surv + loss_c

            # classification loss
            loss_cls = nn.functional.cross_entropy(logits_mean, labels_tensor, weight=self.class_weights)

            loss = alpha * loss_surv + (1.0 - alpha) * loss_cls

            #loss = self.loss_fn(preds, durations_tensor, event_type_tensor, labels_tensor,)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- validation loss if provided (use same metrics)
            val_loss = None
            val_acc = 0.0
            val_cindex = 0.0
            
            if val_input is not None and val_target is not None:
                # eval forward on validation set
                self.net.eval()
                with torch.no_grad():
                    eta_val, logits_val = self._eval_forward_numpy(val_input)
                    # convert to tensors for computing cox loss
                    eta_val_t = torch.as_tensor(eta_val, dtype=torch.float32, device=self.device)
                    logits_val_t = torch.as_tensor(logits_val, dtype=torch.float32, device=self.device)
                    durations_val = torch.as_tensor(val_target[0], dtype=torch.float32, device=self.device)
                    event_type_val = torch.as_tensor(val_target[1], dtype=torch.long, device=self.device)
                    labels_val = torch.as_tensor(val_target[2], dtype=torch.long, device=self.device)

                    loss_surv_val = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                    for c in range(self.K):
                        mask_c = (event_type_val == (c + 1)).long()
                        loss_c_val = cox_ph_loss_tabm(eta_val_t[:, c], durations_val, mask_c, tie_method=tie_method, reduction='mean')
                        if self.event_weights is not None:
                            loss_surv_val = loss_surv_val + loss_c_val * self.event_weights[c + 1]
                        else:
                            loss_surv_val = loss_surv_val + loss_c_val
                    loss_cls_val = nn.functional.cross_entropy(logits_val_t, labels_val, weight=self.class_weights)
                    val_loss = alpha * loss_surv_val + (1.0 - alpha) * loss_cls_val
                    #val_loss = self.loss_fn((eta_val_t, logits_val_t), durations_val, event_type_val, labels_val)
                    val_loss = float(val_loss.detach().cpu().numpy())

                    # --- NEW: Calculate Metrics for 'combined' mode ---
                    if early_stopping_metric == 'combined':
                        # 1. Accuracy
                        preds_cls = np.argmax(logits_val, axis=1)
                        val_acc = accuracy_score(val_target[2], preds_cls)

                        # 2. C-index (Any Event)
                        # We use Total Risk = sum(exp(eta))
                        # Clip to prevent overflow during unstable training epochs
                        eta_val_safe = np.clip(eta_val, -50, 50)
                        eta_val_safe = np.exp(eta_val_safe)
                        risk_score = np.sum(eta_val_safe, axis=1)
                        
                        if SKSURV_AVAILABLE:
                            # Create boolean event indicator (any cause > 0)
                            any_event = (val_target[1] > 0).astype(bool)
                            try:
                                val_cindex = concordance_index_censored(any_event, val_target[0], risk_score)[0]
                            except ValueError:
                                val_cindex = 0.5 # Fallback if calculation fails (e.g. all censored)
                        else:
                            val_cindex = 0.5 # Neutral score if library missing

            # record history
            history.append({
                'epoch': epoch,
                'loss': float(loss.detach().cpu().numpy()),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_cindex': val_cindex
            })

            if verbose and epoch % max(1, epochs // 10) == 0:
                if early_stopping_metric == 'combined':
                    print(f"[{_now()}] Epoch {epoch}/{epochs} loss={history[-1]['loss']:.4f} val_loss={val_loss:.4f} val_score={(val_acc+val_cindex):.4f}")
                else:
                    print(f"[{_now()}] Epoch {epoch}/{epochs} loss={history[-1]['loss']:.4f} val_loss={val_loss:.4f}")

            # early stopping logic
            if val_loss is not None:
                improved = False
                
                if early_stopping_metric == 'combined':
                    # MAXIMIZE (Accuracy + C-index)
                    current_score = val_acc + val_cindex
                    if current_score > best_val_score + 1e-4: # small epsilon
                        best_val_score = current_score
                        improved = True
                else:
                    # MINIMIZE (Loss)
                    if val_loss < best_val_loss - 1e-12:
                        best_val_loss = val_loss
                        improved = True

                if improved:
                    best_state = copy.deepcopy(self.net.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if early_stopping and epochs_no_improve >= patience:
                    if verbose:
                        if early_stopping_metric == 'combined':
                             print(f"[{_now()}] Early stopping after epoch {epoch}, best_val_score={best_val_score:.4f}")
                        else:
                             print(f"[{_now()}] Early stopping after epoch {epoch}, best_val_loss={best_val_loss:.6f}")
                    break

        # restore best weights if available
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.history_ = history
        if verbose:
            print(f"[{_now()}] Training complete. epochs_run={len(history)}")

        # after training compute baseline hazards on train set
        try:
            self.fit_baselines(train_input, train_target)
        except Exception as e:
            print(f"[{_now()}] Warning: baseline estimation failed: {e}")

        return history

    # ---------------------------
    # baseline estimation & CIF
    # ---------------------------
    def fit_baselines(self, train_input, train_target):
        """
        Compute Breslow baseline cumulative hazards using the entire training set.
        Stores event_times_, H0_, dH0_ on the object.
        """
        # predict eta on training set (numpy)
        eta_train, _ = self._eval_forward_numpy(train_input)  # (N, K)
        durations_tr = np.asarray(train_target[0])
        events_tr = np.asarray(train_target[1]).astype(int)

        event_times, H0, dH0 = compute_breslow_baseline_hazards(durations_tr, events_tr, eta_train)
        self.event_times_ = event_times
        self.H0_ = H0
        self.dH0_ = dH0
        return event_times, H0, dH0

    def predict_eta(self, input_tuple):
        """
        Return eta (N, K) numpy array = averaged log_h across heads.
        """
        eta_np, _ = self._eval_forward_numpy(input_tuple)
        return eta_np

    def predict_cif(self, input_tuple, horizons, cause_index=0):
        """
        Predict raw (uncalibrated) CIF for cause_index (0-based) at given horizon(s).
        horizons: scalar or iterable of times.
        Returns numpy array shape (N,) or (N, len(horizons)).
        """
        if self.event_times_ is None or self.H0_ is None or self.dH0_ is None:
            raise RuntimeError("Baselines not fitted. Run fit_baselines(...) after training before predicting CIF.")
        eta_np = self.predict_eta(input_tuple)  # (N, K)
        cif = predict_cif_from_breslow(self.event_times_, self.H0_, self.dH0_, eta_np, horizons, cause_index=cause_index)
        return cif

    # ---------------------------
    # calibration
    # ---------------------------
    def fit_cif_calibrators(self, val_input, val_target, horizons, method='isotonic', min_samples=50):
        """
        Fit per-horizon per-cause calibrators on validation data.
        Stores calibrators in self.horizon_calibrators_ keyed by (horizon, cause_index).
        """
        if self.event_times_ is None:
            raise RuntimeError("Baselines not fitted. Call fit_baselines(...) first.")

        # compute eta and raw predicted CIFs for validation set
        eta_val, _ = self._eval_forward_numpy(val_input)
        durations_val = np.asarray(val_target[0])
        events_val = np.asarray(val_target[1]).astype(int)

        # for each horizon & cause, fit calibrator
        for h in np.atleast_1d(horizons):
            for c in range(self.K):
                raw_cif = predict_cif_from_breslow(self.event_times_, self.H0_, self.dH0_, eta_val, h, cause_index=c)
                try:
                    calibrator, usable_idx = fit_calibrator_on_predicted_cif(raw_cif, durations_val, events_val, h, cause_index=c, method=method, min_samples=min_samples)
                    self.horizon_calibrators_[(int(h), int(c))] = {
                        'calibrator': calibrator,
                        'usable_n': int(len(usable_idx)),
                        'method': method
                    }
                except ValueError as e:
                    # Not enough samples; store None (user can inspect)
                    self.horizon_calibrators_[(int(h), int(c))] = {
                        'calibrator': None,
                        'usable_n': 0,
                        'method': method,
                        'error': str(e)
                    }

        return self.horizon_calibrators_

    def predict_calibrated_cif(self, input_tuple, horizon=None, cause_index=0):
        """
        Returns calibrated probabilities for the given horizon and cause.
        Requires that fit_cif_calibrators(...) has been called and that baseline hazards exist.
        """
        if horizon is None:
            raise ValueError("horizon must be specified for calibrated CIF prediction.")
        
        key = (int(horizon), int(cause_index))
        if key not in self.horizon_calibrators_:
            raise ValueError(f"No calibrator found for horizon {horizon}, cause {cause_index}. Call fit_cif_calibrators(...) first.")
        cal_info = self.horizon_calibrators_[key]
        calibrator = cal_info.get('calibrator', None)
        if calibrator is None:
            raise RuntimeError(f"Calibrator for horizon {horizon}, cause {cause_index} is missing or not fitted: {cal_info}")
        
        raw_cif = self.predict_cif(input_tuple, horizon, cause_index=cause_index)

        if not np.all(np.isfinite(raw_cif)):
        # Option 1: Replace non-finite with 0 or a safe value
            raw_cif = np.nan_to_num(raw_cif, nan=0.0, posinf=1.0, neginf=0.0)
        # Option 2: Or raise an error for debugging
        # raise ValueError("raw_cif contains NaN or Inf values!")
        
        calibrated = predict_calibrated_cif(calibrator, raw_cif)
        return calibrated

    # ---------------------------
    # classification temperature scaling
    # ---------------------------
    def calibrate_temperature(self, val_input, val_labels, max_iter=200):
        """
        Temperature scaling for classification logits using validation set.
        Stores temperature_ and sets temperature_calibrated_=True.
        """
        # get logits on validation set
        _, logits_val = self._eval_forward_numpy(val_input)  # (N, C)
        logits_t = torch.as_tensor(logits_val, dtype=torch.float32, device=self.device)
        labels_t = torch.as_tensor(val_labels, dtype=torch.long, device=self.device)

        # param = log_temp for stability
        log_temp = torch.nn.Parameter(torch.zeros(1, device=self.device))
        optimizer = torch.optim.LBFGS([log_temp], max_iter=max_iter, line_search_fn='strong_wolfe')

        loss_fn = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            T = torch.exp(log_temp)
            logits_scaled = logits_t / T
            loss = loss_fn(logits_scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature_ = float(torch.exp(log_temp).detach().cpu().item())
        self.temperature_calibrated_ = True
        return self.temperature_

    # ---------------------------
    # misc: predict total risk or per-cause
    # ---------------------------
    def predict_risk(self, input_tuple):
        """
        Return a dictionary with:
            'eta': (N, K) array (log-risk per cause),
            'total_risk': (N,) array (sum of hazards as proxy risk),
            'logits': (N, C) raw logits (temperature NOT applied)
        """
        eta, logits = self._eval_forward_numpy(input_tuple)
        eta_safe = np.clip(eta, -50, 50)
        # total risk: sum of exp(eta) (PH model gives hazard proportional to baseline*exp(eta); baseline unknown here)
        total_hazard = np.sum(np.exp(eta_safe), axis=1)
        return {'eta': eta, 'total_hazard': total_hazard, 'logits': logits}

    # ---------------------------
    # misc: predict total risk or per-cause
    # ---------------------------
    def predict_risk(self, input_tuple):
        """
        Return a dictionary with:
            'eta': (N, K) array (log-risk per cause),
            'total_risk': (N,) array (sum of hazards as proxy risk),
            'logits': (N, C) raw logits (temperature NOT applied)
        """
        eta, logits = self._eval_forward_numpy(input_tuple)
        # total risk: sum of exp(eta) (PH model gives hazard proportional to baseline*exp(eta); baseline unknown here)
        eta_safe = np.clip(eta, -50, 50)
        total_hazard = np.sum(np.exp(eta_safe), axis=1)
        return {'eta': eta, 'total_hazard': total_hazard, 'logits': logits}

    def predict_all(self, input_tuple, horizons=[30, 180, 365, 1095]):
        """Get all model outputs in one pass.
        
        Parameters
        ----------
        input_tuple : tuple
            (X_num, X_cat)
        horizons : float or list of floats, optional
            Time horizons to evaluate baseline-referenced cumulative hazards.
            
        Returns
        -------
        dict
            - 'eta': (N, K) log-hazards
            - 'risk_score': (N, K) exp(eta) (hazard ratio)
            - 'total_risk_score': (N,) sum of risk_scores
            - 'logits': (N, C) raw classification logits
            - 'scaled_logits': (N, C) temperature-scaled logits
            - 'cumulative_hazard': (N, K, len(horizons)) [Only if horizons provided]
              This is H0(t) * exp(eta), the absolute cumulative hazard. """
        eta, logits = self._eval_forward_numpy(input_tuple)
        
        # Hazard ratios (risk scores)
        # exp(eta) is the factor multiplying the baseline hazard H0(t)
        # We clip to avoid overflow, consistent with other methods
        eta_safe = np.clip(eta, -50, 50)
        risk_score = np.exp(eta_safe)
        total_risk_score = np.sum(risk_score, axis=1)

        # Classification
        scaled_logits = logits / self.temperature_

        result = {
            'eta': eta,
            'risk_score': risk_score,
            'total_risk_score': total_risk_score,
            'logits': logits,
            'scaled_logits': scaled_logits
        }

        # Calculate baseline-referenced cumulative hazard if horizons provided
        if horizons is not None:
            if self.H0_ is None:
                raise RuntimeError("Baselines not fitted. Run fit_baselines(...) first.")
            
            horizons_arr = np.atleast_1d(horizons)
            # Find indices in event_times_ corresponding to horizons
            # event_times_ is sorted. searchsorted returns insertion point.
            # We want the index of the largest event time <= horizon.
            # side='right' gives index i such that a[i-1] <= v < a[i]
            idx = np.searchsorted(self.event_times_, horizons_arr, side='right') - 1
            
            # Gather H0 for each cause at these indices
            # H0_ is shape (K, M)
            # We want output (K, len(horizons))
            H0_at_t = np.zeros((self.K, len(horizons_arr)))
            
            # Valid indices are >= 0. If idx is -1, it means horizon < first event -> H0=0
            valid_mask = idx >= 0
            valid_indices = idx[valid_mask]
            
            if len(valid_indices) > 0:
                H0_at_t[:, valid_mask] = self.H0_[:, valid_indices]

            # Cumulative Hazard H(t) = H0(t) * exp(eta)
            # risk_score: (N, K)
            # H0_at_t: (K, T)
            # Result: (N, K, T)
            cumulative_hazard = risk_score[:, :, None] * H0_at_t[None, :, :]
            
            result['cumulative_hazard'] = cumulative_hazard
            result['horizons'] = horizons_arr

        return result
    # ---------------------------
    # utility to save/load model weights
    # ---------------------------
    def state_dict(self):
        return {
            'net': self.net.state_dict(),
            'event_times': self.event_times_,
            'H0': self.H0_,
            'dH0': self.dH0_,
            'horizon_calibrators': self.horizon_calibrators_,
            'temperature': self.temperature_,
        }

    def load_state_dict(self, state):
        if 'net' in state:
            self.net.load_state_dict(state['net'])
        self.event_times_ = state.get('event_times', None)
        self.H0_ = state.get('H0', None)
        self.dH0_ = state.get('dH0', None)
        self.horizon_calibrators_ = state.get('horizon_calibrators', {})
        self.temperature_ = state.get('temperature', 1.0)
        return self
