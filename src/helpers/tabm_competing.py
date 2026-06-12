import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


EPS = 1e-12


# -------------------------------------------------------------------------
# 1) Cause-Specific Cox PH Loss (torch)
# -------------------------------------------------------------------------
def cox_ph_loss_tabm(scores, durations, events, tie_method='breslow',
                     reduction='mean', eps=1e-8,):
    """
    Negative partial log-likelihood for Cox PH.
    This is the cleaned version designed for cause-specific Cox.

    scores: (N,) log-risk (eta)
    durations: (N,)
    events: (N,) binary 0/1 for a single cause
    """

    if not torch.is_tensor(scores):
        scores = torch.as_tensor(scores, dtype=torch.get_default_dtype())
    if not torch.is_tensor(durations):
        durations = torch.as_tensor(durations, dtype=torch.get_default_dtype())
    if not torch.is_tensor(events):
        events = torch.as_tensor(events, dtype=torch.get_default_dtype())

    device = scores.device
    scores = scores.view(-1).to(device) #* -1.0  # Negate for risk score
    durations = durations.view(-1).to(device)
    events = events.view(-1).float().to(device)

    N = scores.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=device)

    num_events = torch.clamp(events.sum(), min=0.0)
    if num_events.item() == 0:
        return torch.tensor(0.0, device=device)

    # Sort by time descending (standard for risk set construction)
    durations_sorted, order = torch.sort(durations, descending=True)
    scores_sorted = scores[order]
    events_sorted = events[order]
    

    exp_scores = torch.exp(scores_sorted)
    denom_cumsum = torch.cumsum(exp_scores, dim=0)
    #if torch.any(denom_cumsum < 1e-8):
        #print("Warning: Small values in denom_cumsum, possible instability in Cox loss.")

    # Breslow is stable
    if tie_method == 'breslow':
        term = events_sorted * (scores_sorted - torch.log(denom_cumsum + eps))
        
        loss = -torch.sum(term)

    # Efron (more correct for ties but more expensive)
    elif tie_method == 'efron':
        loss_val = torch.tensor(0.0, device=device)
        i = 0
        Nsorted = durations_sorted.shape[0]

        while i < Nsorted:
            t_i = durations_sorted[i]
            j = i + 1
            while j < Nsorted and durations_sorted[j] == t_i:
                j += 1

            idx = torch.arange(i, j, device=device)
            d = events_sorted[idx].sum()

            if d == 0:
                i = j
                continue

            event_mask = events_sorted[idx] == 1.0
            sum_scores_event = scores_sorted[idx][event_mask].sum()
            denom = denom_cumsum[i]

            tied_exp_sum = exp_scores[idx].sum()
            d_int = int(d.item())

            # Efron correction
            l = torch.arange(0, d_int, device=device, dtype=denom.dtype)
            denom_l = denom - (l * tied_exp_sum / float(d_int))
            denom_l = torch.clamp(denom_l, min=eps)
            log_sum = torch.log(denom_l).sum()

            loss_val += -(sum_scores_event - log_sum)
            i = j

        loss = loss_val
    else:
        raise ValueError("tie_method must be 'breslow' or 'efron'")

    if reduction == 'mean':
        loss = loss / (num_events + eps)
    elif reduction == 'none':
        pass
    elif reduction == 'sum':
        pass
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    return loss


# -------------------------------------------------------------------------
# 2) Reduce TabM output -> eta (numpy)
# -------------------------------------------------------------------------
def reduce_loghaz_to_eta(log_h):
    """
    Handles shapes:
        (N, K), (N, H, K), (N,), (N,K,1)
    Returns:
        (N, K) numpy array
    """
    if isinstance(log_h, torch.Tensor):
        arr = log_h.detach().cpu().numpy()
    else:
        arr = np.asarray(log_h)

    if arr.ndim == 3:          # (N, H, K)
        arr = arr.mean(axis=1)
    elif arr.ndim == 2:
        pass
    elif arr.ndim == 1:        # (N,)
        arr = arr.reshape(-1, 1)
    else:
        raise ValueError(f"Unexpected shape for log_h: {arr.shape}")

    return arr.astype(float)


# -------------------------------------------------------------------------
# 3) Breslow baseline estimation for competing risks
# -------------------------------------------------------------------------
def compute_breslow_baseline_hazards(durations, events, eta, times=None):
    """
    durations: (N,)
    events: (N,) ints {0=censor, 1..K}
    eta: (N, K)
    Returns:
        event_times: (M,)
        H0: (K, M)
        dH0: (K, M)
    """
    durations = np.asarray(durations).astype(float)
    events = np.asarray(events).astype(int)
    eta = np.asarray(eta).astype(float)

    # Fix: Clip eta to prevent overflow in exp()
    # This ensures stability even if the model outputs extreme values
    eta = np.clip(eta, -50, 50)

    N, K = eta.shape

    if times is None:
        mask = events != 0
        event_times = np.unique(durations[mask])
        event_times.sort()
    else:
        event_times = np.unique(np.asarray(times))
        event_times.sort()

    M = len(event_times)
    H0 = np.zeros((K, M))
    dH0 = np.zeros((K, M))
    exp_eta = np.exp(eta)

    for j, t_j in enumerate(event_times):
        risk_idx = np.where(durations >= t_j)[0]

        for k in range(K):
            d_kj = np.sum((durations == t_j) & (events == (k + 1)))
            if d_kj == 0:
                continue

            denom = np.sum(exp_eta[risk_idx, k])
            denom = max(denom, EPS)

            dH_kj = d_kj / denom
            dH0[k, j] = dH_kj

            H0[k, j] = H0[k, j-1] + dH_kj if j > 0 else dH_kj

    return event_times, H0, dH0


# -------------------------------------------------------------------------
# 4) Predict CIF
# -------------------------------------------------------------------------
def predict_cif_from_breslow(event_times, H0, dH0, eta, horizons, cause_index=0):
    """
    eta: (N,K)
    Returns CIF_k(t) for each horizon.
    """
    horizons_arr = np.atleast_1d(horizons)
    
    # Fix: Clip eta to prevent overflow in exp()
    eta = np.clip(eta, -50, 50)

    N, K = eta.shape
    M = len(event_times)

    exp_eta = np.exp(eta)
    cif = np.zeros((N, len(horizons_arr)))

    H0_prev = np.zeros_like(H0)
    if M > 0:
        H0_prev[:, 1:] = H0[:, :-1]

    for hi, h in enumerate(horizons_arr):
        valid_j = np.where(event_times <= h)[0]
        if valid_j.size == 0:
            continue

        for j in valid_j:
            H_prev_j = H0_prev[:, j]
            S_prev = np.exp(-np.sum(exp_eta * H_prev_j[np.newaxis, :], axis=1))

            dHkj = dH0[cause_index, j]
            contrib = S_prev * (dHkj * exp_eta[:, cause_index])
            cif[:, hi] += contrib

    return cif[:, 0] if np.isscalar(horizons) else cif


# -------------------------------------------------------------------------
# 5) Calibration functions
# -------------------------------------------------------------------------
def fit_calibrator_on_predicted_cif(predicted_cif, durations, events, horizon,
                                    cause_index=0, method='isotonic',
                                    min_samples=50):
    """
    Fits a calibrator mapping raw CIF -> calibrated risk at a horizon.
    """
    pred = np.asarray(predicted_cif).ravel()
    durations = np.asarray(durations)
    events = np.asarray(events).astype(int)

    # Construct binary outcome for horizon
    pos = (durations <= horizon) & (events == (cause_index + 1))
    neg = durations > horizon
    mask = pos | neg
    usable = np.where(mask)[0]

    if usable.size < min_samples:
        raise ValueError(f"Too few samples for calibration at horizon={horizon}")

    X = pred[usable].reshape(-1, 1)
    y = pos[usable].astype(int)

    if method == 'logistic':
        clf = LogisticRegression(max_iter=500)
        clf.fit(X, y)
        return clf, usable
    else:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(X.ravel(), y)
        return iso, usable


def predict_calibrated_cif(calibrator, predicted_cif):
    pred = np.asarray(predicted_cif).ravel()
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(pred.reshape(-1, 1))[:, 1]
    return calibrator.predict(pred)


# -------------------------------------------------------------------------
# 6) Multi-cause TabM wrapper
# -------------------------------------------------------------------------
class MultiTaskTabMWrapperWithCauses(nn.Module):
    """
    Wrap TabM base model so that:
        forward -> (log_h, logits)
    where:
        log_h:   (B, H, K)
        logits:  (B, H, C)
    """
    def __init__(self, tabm_model, num_causes):
        super().__init__()
        self.tabm = tabm_model
        self.num_causes = int(num_causes)

    def forward(self, x_num, x_cat):
        out = self.tabm(x_num, x_cat)  # (B, H, K + C)
        K = self.num_causes
        log_h = out[..., :K]
        logits = out[..., K:]
        return log_h, logits


# -------------------------------------------------------------------------
# 7) Cause-Specific TabM Loss
# -------------------------------------------------------------------------
class CompetingRiskTabMLoss(nn.Module):
    def __init__(self, K, alpha=0.4, mode='multiclass',
                 event_weights=None, class_weights=None,
                 tie_method='breslow'):
        super().__init__()
        self.K = int(K)
        self.alpha = float(alpha)
        self.mode = mode
        self.tie_method = tie_method

        self.register_buffer(
            'event_weights',
            torch.as_tensor(event_weights) if event_weights is not None else None
        )
        self.register_buffer(
            'class_weights',
            torch.as_tensor(class_weights) if class_weights is not None else None
        )

    def forward(self, preds, durations, event_type, labels):
        log_h, logits = preds

        # average heads
        if log_h.ndim == 3:
            eta = log_h.mean(1)   # (B, K)
        else:
            eta = log_h

        if logits.ndim == 3:
            logits = logits.mean(1)

        durations = durations.view(-1)
        event_type = event_type.view(-1)

        loss_surv = 0.0

        eta = torch.clamp(eta, min=-50, max=50)

        for c in range(self.K):
            mask_c = (event_type == (c + 1)).long()
            loss_c = cox_ph_loss_tabm(
                eta[:, c], durations, mask_c,
                tie_method=self.tie_method
            )
            if self.event_weights is not None:
                loss_c = loss_c * self.event_weights[c]
            loss_surv += loss_c

        if self.mode == 'multiclass':
            loss_cls = nn.functional.cross_entropy(logits, labels.long(), weight=self.class_weights)
        else:
            loss_cls = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float()
            )

        return self.alpha * loss_surv + (1 - self.alpha) * loss_cls
