import json, numpy as np, joblib
from pathlib import Path

LEVELS = ["Kn","Cm","Ap","An","Sn","Ev"]

def _load_cdfs(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cdfs = []
    for d in raw:
        xs = np.asarray(d["xs"], dtype=float)
        cdf = np.asarray(d["cdf"], dtype=float)
        cdfs.append((xs, cdf))
    return cdfs

def apply_cdfs(S, cdfs):
    S = np.nan_to_num(np.asarray(S, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    S01 = np.zeros_like(S, dtype=float)
    for j in range(S.shape[1]):
        xs, cdf = cdfs[j]
        S01[:, j] = np.interp(S[:, j], xs, cdf, left=cdf[0], right=cdf[-1])
    return np.clip(S01, 0.0, 1.0)

class BT4Scorer:
    def __init__(self, bundle_dir, use_oof_thresholds=False):
        bundle = Path(bundle_dir)
        self.model = joblib.load(bundle / "model.pkl")
        self.cdfs  = _load_cdfs(bundle / "calibration_cdfs.json")
        with open(bundle / "thresholds.json", "r", encoding="utf-8") as f:
            thr = json.load(f)
        if use_oof_thresholds and "thresholds_oof_aligned" in thr:
            self.thresholds = np.asarray(thr["thresholds_oof_aligned"], dtype=float)
            self.thr_source = "OOF_ALIGNED"
        else:
            self.thresholds = np.asarray(thr["thresholds_tuning"], dtype=float)
            self.thr_source = "TUNING"

    def score_raw(self, X):
        # Uses decision_function if available; else predict_proba; else predict.
        m = self.model
        if hasattr(m, "decision_function"):
            S = m.decision_function(X)
        elif hasattr(m, "predict_proba"):
            P = m.predict_proba(X)
            # OneVsRest: list of (n,2) arrays → take prob of class 1
            if isinstance(P, list):
                S = np.column_stack([p[:,1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel() for p in P])
            else:
                S = P
        else:
            S = m.predict(X).astype(float)
        S = np.asarray(S, dtype=float)
        if S.ndim == 1: S = S.reshape(-1, len(LEVELS))
        return S

    def score_calibrated(self, X):
        S = self.score_raw(X)
        return apply_cdfs(S, self.cdfs)

    def predict_binary(self, X):
        S01 = self.score_calibrated(X)
        Yb = (S01 >= self.thresholds.reshape(1, -1)).astype(int)
        return Yb, S01

    def suggest_binary(self, X, alpha=0.85):
        S01 = self.score_calibrated(X)
        thr_soft = self.thresholds * float(alpha)
        Yb = (S01 >= thr_soft.reshape(1, -1)).astype(int)
        return Yb, S01