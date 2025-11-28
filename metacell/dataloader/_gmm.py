# GMM-based adaptive filtering and scoring of signal peaks.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def filter_and_score_gmm(
    lif_peak_data: pd.DataFrame,
    max_peaks: int = 800,
    intensity_col: str = "corrected_intensity",
    time_col: str = "times",
    min_p_high: float = 0.50,        # at least 50% prob of 'high-intensity' component
    weights: tuple = (0.5, 0.3, 0.2),# (w_prob, w_zsig, w_perc)
    log_eps: float = 1e-12           # avoid log(0)
) -> tuple[pd.DataFrame, dict]:
    """
    GMM-based adaptive filtering and scoring of signal peaks.
    Returns:
      kept: filtered peaks (belonging to high component with prob >= min_p_high), sorted by score desc
      info: fit & threshold info
    """
    if "orig_index" not in lif_peak_data.columns:
        lif_peak_data = lif_peak_data.copy()
        lif_peak_data["orig_index"] = lif_peak_data.index
    
    if intensity_col not in lif_peak_data.columns:
        raise KeyError(f"Column '{intensity_col}' not found.")
    
    # --- 1) Take top-N by intensity ---
    top = lif_peak_data.sort_values(intensity_col, ascending=False).head(int(max_peaks)).copy()

    # --- 2) Fit 2-component GMM in log-space ---
    x = top[intensity_col].to_numpy(float)
    pos_min = np.nanmin(x[x > 0]) if np.any(x > 0) else 1.0
    x_clipped = np.clip(x, a_min=max(pos_min*1e-6, log_eps), a_max=None)
    logx = np.log(x_clipped).reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(logx)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.reshape(2, 1, 1).squeeze())

    resp = gmm.predict_proba(logx)       # posterior prob for each component
    labels = gmm.predict(logx)
    high_comp = int(np.argmax(means))    # larger-mean component is "high-intensity"
    p_high = resp[:, high_comp]

    # --- 3) Keep high-comp points with sufficient posterior ---
    low_comp = 1 - high_comp
    mu_low = float(means[low_comp])
    keep_mask = (p_high >= float(min_p_high)) & (logx.flatten() > mu_low)
    kept = top.loc[keep_mask].copy()

    # --- 4) Score: combine prob + z-score(sigmoid) + percentile rank ---
    mu_h, sd_h = float(means[high_comp]), float(stds[high_comp] if np.isfinite(stds[high_comp]) else 1.0)
    z_high_all = (logx.flatten() - mu_h) / (sd_h + 1e-12)
    z_high = z_high_all[keep_mask]

    ranks = kept[intensity_col].rank(pct=True, ascending=True).to_numpy()
    z_sig = 1.0 / (1.0 + np.exp(-z_high))

    w_prob, w_zsig, w_perc = weights
    score = w_prob * p_high[keep_mask] + w_zsig * z_sig + w_perc * ranks

    kept["p_high"] = p_high[keep_mask]
    kept["score"] = score
    kept = kept.sort_values("score", ascending=False).reset_index(drop=True)

    info = {
        "method_used": "gmm",
        "selected_topN": int(max_peaks),
        "kept_count": int(len(kept)),
        "min_p_high": float(min_p_high),
        "weights": tuple(float(w) for w in weights),
        "gmm_means_log": [float(m) for m in means],
        "gmm_stds_log":  [float(s) for s in stds],
        "high_comp_index": int(high_comp)
    }
    # also return objects needed for plots
    extras = {
        "gmm": gmm,
        "logx_all": logx.flatten(),
        "top_df": top,
        "p_high_all": p_high,
        "keep_mask": keep_mask
    }
    return kept, info, extras

def _gaussian_pdf(x, mu, sigma):
    return (1.0 / (np.sqrt(2.0*np.pi)*sigma)) * np.exp(-0.5*((x-mu)/sigma)**2)

def plot_hist_with_gmm(logx, gmm, bins=50):
    """
    Plot histogram of log(intensity) with fitted GMM component curves + mixture curve.
    """
    plt.figure()
    # histogram
    plt.hist(logx, bins=bins, density=True, alpha=0.5)
    xs = np.linspace(np.min(logx), np.max(logx), 400)
    # mixture curve
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.reshape(2, 1, 1).squeeze())
    comp1 = weights[0] * _gaussian_pdf(xs, means[0], stds[0])
    comp2 = weights[1] * _gaussian_pdf(xs, means[1], stds[1])
    mixture = comp1 + comp2

    plt.plot(xs, comp1)
    plt.plot(xs, comp2)
    plt.plot(xs, mixture)
    plt.xlabel("log(intensity)")
    plt.ylabel("Density")
    plt.title("Histogram of log(intensity) with GMM fit")
    plt.show()

def plot_posterior_scatter(logx, p_high, keep_mask, min_p_high):
    """
    Scatter: log(intensity) vs posterior P(high component); kept points marked by threshold.
    """
    plt.figure()
    plt.scatter(logx[~keep_mask], p_high[~keep_mask], s=10)
    plt.scatter(logx[keep_mask], p_high[keep_mask], s=10)
    plt.axhline(min_p_high, linestyle="--")
    plt.xlabel("log(intensity)")
    plt.ylabel("Posterior P(high component)")
    plt.title("Posterior probability of high-intensity component")
    plt.show()

def plot_score_distribution(kept_df):
    """
    Plot histogram of the score among kept peaks.
    """
    plt.figure()
    plt.hist(kept_df["score"].to_numpy(), bins=40)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Score distribution (kept peaks)")
    plt.show()