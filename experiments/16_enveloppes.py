# %% [markdown]
# # Experiment 16 — Enveloppes patient + fiabilité médecin
#
# **Date:** 2025-01-07
# **Base:** Exp 15 (courbe MLE log-normale asymétrique)
#
# ## Nouvelles features : enveloppes hautes et fiabilité rater
#
# Le bruit tire souvent les mesures vers le bas et presque jamais vers le haut.
# Les mesures les plus hautes d'un patient (et celles autour de chaque visite)
# sont donc les plus fiables. Chaque médecin a aussi son propre "plafond".
#
# Après le calcul de mle / mle_slope, pour chaque patient (visites triées par âge) :
#
# **Enveloppe patient :**
#   am = âge moyen du patient, tendance = mle_slope * (age - am),
#   zd = z - tendance (résidu détrienté).
#   Statistiques globales de zd → exp(stat + tendance) :
#     médiane, q75, q90, max, moyenne des 2 plus grands.
#   Plus écart-type et nombre de zd valides.
#
# **Enveloppe locale :**
#   Fenêtres centrées de 3, 5, 7 visites (min 1 valeur) sur zd : max et moyenne.
#   Cummax de zd et cummax inverse.
#   zd des visites -2, -1, +1, +2 (valeur ou NaN).
#   Tout → exp(stat + tendance).
#
# **Fiabilité médecin :**
#   Sur le train du fold uniquement, u = z - log(target) (target > 3).
#   Par rater_id : médiane, écart-type, q10, q90 de u → colonnes r_med, r_sd, r_q10, r_q90.
#   Valeur globale si médecin inconnu.
#   zr = z - r_q90 - tendance.
#   Statistiques de zr → exp(stat + tendance) :
#     max du patient, moyenne des 2 plus grands, moyenne,
#     max sur fenêtre de 3, cummax, cummax inverse.
#
# La boucle par patient de _add_mle_features est parallélisée avec joblib.
# Tout le reste (folds, stacking 2 étages, lissage) est identique à exp 15.

# %%
import numpy as np
import pandas as pd
import skore
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.utils.validation import check_is_fitted
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset
from parkinsons.evaluate import get_cv_folds
from parkinsons.hub import load_skore_credentials
from parkinsons.features import build_features, FEATURE_COLS
from parkinsons.postprocess import smooth_by_age_poly

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

DATA_DIR = PROJECT_ROOT / "data"
X_train, y_train = load_dataset(DATA_DIR)
print(f"Train: {X_train.shape}")

folds = get_cv_folds(X_train, y_train)

# %% [markdown]
# ## Correction de réponse individuelle au traitement (identique exp 14)

# %%
def _apply_patient_response_correction(X_fe: pd.DataFrame) -> pd.DataFrame:
    """Corrige on_corrected et on_time_corrected par le facteur de réponse k."""
    df = X_fe.copy()
    k_map = {}
    n_map = {}

    for pid, grp in df.groupby("patient_id"):
        mask = (
            grp["on_time_corrected"].notna()
            & grp["off_corrected"].notna()
            & (grp["off_corrected"] > 1)
        )
        valid = grp.loc[mask]
        n_valid = len(valid)

        if n_valid == 0:
            k_map[pid] = 1.0
            n_map[pid] = 0
        else:
            ratio = valid["on_time_corrected"] / valid["off_corrected"]
            ratio = ratio.clip(lower=1e-6)
            log_sum = float(np.log(ratio).sum())
            k_map[pid] = float(np.exp(log_sum / (n_valid + 1)))
            n_map[pid] = n_valid

    k_series = df["patient_id"].map(k_map).clip(lower=1e-6)
    df["on_corrected"]      = df["on_corrected"] / k_series
    df["on_time_corrected"] = df["on_time_corrected"] / k_series
    df["patient_k_response"] = k_series
    df["patient_k_n_visits"] = df["patient_id"].map(n_map).astype(float)
    return df


# %% [markdown]
# ## Courbe ON/OFF par patient (identique exp 14)

# %%
def _add_onoff_curve(df: pd.DataFrame, min_pts_for_poly2: int = 4) -> pd.DataFrame:
    """Courbe ON+OFF combinée par patient."""
    df = df.copy()
    curve_vals = np.full(len(df), np.nan)

    for pid, grp in df.groupby("patient_id"):
        ages_visit = grp["age"].values
        pos = df.index.get_indexer(grp.index)

        on_valid  = grp[["age", "on_corrected"]].dropna()
        off_valid = grp[["age", "off_corrected"]].dropna()

        ages_all = np.concatenate([on_valid["age"].values, off_valid["age"].values])
        vals_all = np.concatenate([on_valid["on_corrected"].values,
                                   off_valid["off_corrected"].values])
        nv = len(ages_all)

        if nv == 0:
            continue
        elif nv == 1:
            curve_vals[pos] = vals_all[0]
        else:
            degree = 2 if nv >= min_pts_for_poly2 else 1
            degree = min(degree, nv - 1)
            coeffs = np.polyfit(ages_all, vals_all, degree)
            curve_vals[pos] = np.polyval(coeffs, ages_visit)

    df["onoff_curve_at_age"]      = curve_vals
    df["on_residual_from_curve"]  = df["on_corrected"]  - df["onoff_curve_at_age"]
    df["off_residual_from_curve"] = df["off_corrected"] - df["onoff_curve_at_age"]
    return df


# %% [markdown]
# ## Courbe MLE par patient (exp 15) + enveloppes hautes (exp 16)

# %%
def _compute_z(df: pd.DataFrame) -> pd.Series:
    """Calcule z = moyenne de log(off_corrected) et log(on_time_corrected) par ligne.

    Les valeurs manquantes ou <= 0 sont ignorées dans la moyenne.
    Retourne NaN si aucune valeur valide.
    """
    log_off = np.where(
        df["off_corrected"].notna() & (df["off_corrected"] > 0),
        np.log(df["off_corrected"].clip(lower=1e-9)),
        np.nan,
    )
    log_on = np.where(
        df["on_time_corrected"].notna() & (df["on_time_corrected"] > 0),
        np.log(df["on_time_corrected"].clip(lower=1e-9)),
        np.nan,
    )
    stacked = np.stack([log_off, log_on], axis=1)
    return pd.Series(np.nanmean(stacked, axis=1), index=df.index)


class _Density:
    """Densite tabulee sur une grille (classe au niveau du module pour etre picklable / envoyable sur skore)."""

    def __init__(self, grid, values):
        self.grid = np.asarray(grid, dtype=float)
        self.values = np.asarray(values, dtype=float)

    def __call__(self, u):
        u_arr = np.asarray(u, dtype=float)
        return np.maximum(np.interp(u_arr, self.grid, self.values), 1e-4)


def _build_residual_density(df_train: pd.DataFrame, y_train_series: pd.Series):
    """Estime la densite de u = z - log(target) sur le train (target > 3).

    Retourne un objet _Density appelable : densite(u) -> float (vectorise).
    Grille : -4 a 1.5, largeur de noyau 0.04, plancher 1e-4.
    """
    z_tr = _compute_z(df_train)
    log_target = np.log(y_train_series.clip(lower=1e-9))

    mask = (y_train_series > 3) & z_tr.notna()
    u_vals = (z_tr[mask] - log_target[mask]).values

    grid = np.linspace(-4.0, 1.5, 1000)
    if len(u_vals) < 5:
        return _Density(grid, np.full_like(grid, 1e-4))

    kde = gaussian_kde(u_vals, bw_method=0.04 / u_vals.std(ddof=1) if u_vals.std(ddof=1) > 0 else 0.04)
    density_grid = np.maximum(kde(grid), 1e-4)
    return _Density(grid, density_grid)


def _compute_log_slope_stats(df_train: pd.DataFrame, y_train_series: pd.Series):
    """Médiane et écart-type des pentes de log(target) ~ âge par patient.

    Utilise uniquement les patients avec >= 4 visites ayant target disponible.
    """
    slopes = []
    tmp = df_train[["patient_id", "age"]].copy()
    tmp["_logy"] = np.log(y_train_series.clip(lower=1e-9)).values

    for pid, grp in tmp.groupby("patient_id"):
        v = grp[["age", "_logy"]].dropna()
        if len(v) >= 4:
            slope = float(np.polyfit(v["age"].values, v["_logy"].values, 1)[0])
            slopes.append(slope)

    if len(slopes) < 2:
        return 0.0, 1.0

    slopes_arr = np.array(slopes)
    return float(np.median(slopes_arr)), float(np.std(slopes_arr, ddof=1))


def _fit_patient_mle_curve(
    z_patient: np.ndarray,
    ages_patient: np.ndarray,
    density_fn,
    slope_median: float,
    slope_std: float,
) -> tuple[float, float, float]:
    """Optimise a, b, c pour un patient par max de log-vraisemblance.

    x = age - mean(age)
    Objectif : minimiser - sum(log dens(z - (a + b*x + c*x²)))
               + 0.5*((b - slope_median)/slope_std)²
               + 0.5*(c/0.01)²
    Si < 6 visites avec z : c est fixé à 0 (optimisation 2D).
    3 points de départ : a = median(z) + {+0.1, +0.2, -0.2}, meilleur conservé.
    """
    valid = ~np.isnan(z_patient)
    z_v = z_patient[valid]
    x_v = ages_patient[valid] - ages_patient[valid].mean() if valid.sum() > 0 else np.array([])

    use_c = valid.sum() >= 6

    def neg_loglik_3d(params):
        a, b, c = params
        mu = a + b * x_v + c * x_v ** 2
        u = z_v - mu
        ll = np.sum(np.log(density_fn(u)))
        pen_b = 0.5 * ((b - slope_median) / max(slope_std, 1e-6)) ** 2
        pen_c = 0.5 * (c / 0.01) ** 2
        return -(ll - pen_b - pen_c)

    def neg_loglik_2d(params):
        a, b = params
        mu = a + b * x_v
        u = z_v - mu
        ll = np.sum(np.log(density_fn(u)))
        pen_b = 0.5 * ((b - slope_median) / max(slope_std, 1e-6)) ** 2
        return -(ll - pen_b)

    if len(z_v) == 0:
        return 0.0, slope_median, 0.0

    z_med = float(np.median(z_v))
    start_offsets = [+0.1, +0.2, -0.2]

    best_val = np.inf
    best_params = (z_med, slope_median, 0.0)

    for offset in start_offsets:
        if use_c:
            x0 = [z_med + offset, slope_median, 0.0]
            res = minimize(neg_loglik_3d, x0, method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
            params = (res.x[0], res.x[1], res.x[2])
            val = res.fun
        else:
            x0 = [z_med + offset, slope_median]
            res = minimize(neg_loglik_2d, x0, method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
            params = (res.x[0], res.x[1], 0.0)
            val = res.fun

        if val < best_val:
            best_val = val
            best_params = params

    return best_params  # (a, b, c)


def _build_rater_stats(df_train: pd.DataFrame, z_train: pd.Series, y_train_series: pd.Series) -> dict:
    """Calcule la fiabilité médecin sur le train du fold.

    u = z - log(target) (target > 3).
    Par rater_id : médiane, écart-type, q10, q90 de u.
    Retourne un dict avec clés 'r_med', 'r_sd', 'r_q10', 'r_q90' (dict rater->valeur)
    et 'global_*' pour les valeurs de repli.
    """
    mask = (y_train_series > 3) & z_train.notna()
    tmp = df_train[["rater_id"]].copy()
    tmp = tmp[mask]
    u = (z_train[mask] - np.log(y_train_series[mask].clip(lower=1e-9))).values
    tmp = tmp.copy()
    tmp["u"] = u

    r_med, r_sd, r_q10, r_q90 = {}, {}, {}, {}
    for rid, grp in tmp.groupby("rater_id"):
        uv = grp["u"].values
        r_med[rid] = float(np.median(uv))
        r_sd[rid]  = float(np.std(uv, ddof=1)) if len(uv) > 1 else 0.0
        r_q10[rid] = float(np.percentile(uv, 10))
        r_q90[rid] = float(np.percentile(uv, 90))

    g_med = float(np.median(u)) if len(u) > 0 else 0.0
    g_sd  = float(np.std(u, ddof=1)) if len(u) > 1 else 0.0
    g_q10 = float(np.percentile(u, 10)) if len(u) > 0 else 0.0
    g_q90 = float(np.percentile(u, 90)) if len(u) > 0 else 0.0

    return {
        "r_med": r_med, "r_sd": r_sd, "r_q10": r_q10, "r_q90": r_q90,
        "global_med": g_med, "global_sd": g_sd, "global_q10": g_q10, "global_q90": g_q90,
    }


def _window_stats(arr: np.ndarray, center: int, half: int) -> tuple[float, float]:
    """Max et moyenne des valeurs valides dans arr[center-half : center+half+1]."""
    lo = max(0, center - half)
    hi = min(len(arr), center + half + 1)
    window = arr[lo:hi]
    valid = window[~np.isnan(window)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.max(valid)), float(np.mean(valid))


def _process_one_patient(
    pid,
    grp_idx,
    z_vals,        # np.ndarray aligned to grp_idx
    ages,          # np.ndarray aligned to grp_idx
    mle_slope_vals,  # np.ndarray aligned to grp_idx
    rater_ids,     # np.ndarray aligned to grp_idx
    rater_stats,   # dict or None
):
    """Calcule toutes les enveloppes pour un patient.

    Retourne un dict col -> np.ndarray (même longueur que grp_idx, dans l'ordre d'arrivée).
    """
    n = len(grp_idx)
    # Tri par âge pour les calculs séquentiels
    sort_order = np.argsort(ages, kind="stable")
    inv_order  = np.argsort(sort_order, kind="stable")  # pour revenir à l'ordre d'origine

    ages_s    = ages[sort_order]
    z_s       = z_vals[sort_order]
    b_s       = mle_slope_vals[sort_order]  # mle_slope (identique pour tout le patient)
    rater_s   = rater_ids[sort_order]

    # Tendance et résidu détrienté
    am        = float(np.mean(ages_s))
    b_val     = float(b_s[0]) if not np.isnan(b_s[0]) else 0.0
    trend_s   = b_val * (ages_s - am)      # tendance linéaire
    zd_s      = z_s - trend_s              # résidu détrienté

    # ------------------------------------------------------------------ #
    # 1. Enveloppe patient (statistiques globales de zd)
    # ------------------------------------------------------------------ #
    valid_zd = zd_s[~np.isnan(zd_s)]
    n_valid  = len(valid_zd)

    if n_valid >= 1:
        zd_med  = float(np.median(valid_zd))
        zd_q75  = float(np.percentile(valid_zd, 75))
        zd_q90  = float(np.percentile(valid_zd, 90))
        zd_max  = float(np.max(valid_zd))
        zd_top2 = float(np.mean(np.sort(valid_zd)[-2:])) if n_valid >= 2 else zd_max
        zd_std  = float(np.std(valid_zd, ddof=1)) if n_valid >= 2 else 0.0
    else:
        zd_med = zd_q75 = zd_q90 = zd_max = zd_top2 = 0.0
        zd_std = 0.0

    env_med  = np.exp(zd_med  + trend_s)
    env_q75  = np.exp(zd_q75  + trend_s)
    env_q90  = np.exp(zd_q90  + trend_s)
    env_max  = np.exp(zd_max  + trend_s)
    env_top2 = np.exp(zd_top2 + trend_s)

    # ------------------------------------------------------------------ #
    # 2. Enveloppe locale
    # ------------------------------------------------------------------ #
    win_max3  = np.full(n, np.nan)
    win_avg3  = np.full(n, np.nan)
    win_max5  = np.full(n, np.nan)
    win_avg5  = np.full(n, np.nan)
    win_max7  = np.full(n, np.nan)
    win_avg7  = np.full(n, np.nan)

    for i in range(n):
        m3, a3 = _window_stats(zd_s, i, 1)
        m5, a5 = _window_stats(zd_s, i, 2)
        m7, a7 = _window_stats(zd_s, i, 3)
        t = trend_s[i]
        win_max3[i] = np.exp(m3 + t) if not np.isnan(m3) else np.nan
        win_avg3[i] = np.exp(a3 + t) if not np.isnan(a3) else np.nan
        win_max5[i] = np.exp(m5 + t) if not np.isnan(m5) else np.nan
        win_avg5[i] = np.exp(a5 + t) if not np.isnan(a5) else np.nan
        win_max7[i] = np.exp(m7 + t) if not np.isnan(m7) else np.nan
        win_avg7[i] = np.exp(a7 + t) if not np.isnan(a7) else np.nan

    # Cummax forward et backward
    cummax_fwd = np.full(n, np.nan)
    cummax_bwd = np.full(n, np.nan)
    running_max = -np.inf
    for i in range(n):
        if not np.isnan(zd_s[i]):
            running_max = max(running_max, zd_s[i])
        if running_max > -np.inf:
            cummax_fwd[i] = np.exp(running_max + trend_s[i])

    running_max = -np.inf
    for i in range(n - 1, -1, -1):
        if not np.isnan(zd_s[i]):
            running_max = max(running_max, zd_s[i])
        if running_max > -np.inf:
            cummax_bwd[i] = np.exp(running_max + trend_s[i])

    # Lags de zd : visites -2, -1, +1, +2
    zd_lag_m2 = np.full(n, np.nan)
    zd_lag_m1 = np.full(n, np.nan)
    zd_lag_p1 = np.full(n, np.nan)
    zd_lag_p2 = np.full(n, np.nan)
    for i in range(n):
        t = trend_s[i]
        if i >= 2 and not np.isnan(zd_s[i - 2]):
            zd_lag_m2[i] = np.exp(zd_s[i - 2] + t)
        if i >= 1 and not np.isnan(zd_s[i - 1]):
            zd_lag_m1[i] = np.exp(zd_s[i - 1] + t)
        if i + 1 < n and not np.isnan(zd_s[i + 1]):
            zd_lag_p1[i] = np.exp(zd_s[i + 1] + t)
        if i + 2 < n and not np.isnan(zd_s[i + 2]):
            zd_lag_p2[i] = np.exp(zd_s[i + 2] + t)

    # ------------------------------------------------------------------ #
    # 3. Fiabilité médecin : zr = z - r_q90 - tendance
    # ------------------------------------------------------------------ #
    if rater_stats is not None:
        r_q90_arr = np.array([
            rater_stats["r_q90"].get(r, rater_stats["global_q90"])
            for r in rater_s
        ], dtype=float)
    else:
        r_q90_arr = np.zeros(n)

    zr_s = z_s - r_q90_arr - trend_s  # résidu après plafond médecin

    valid_zr = zr_s[~np.isnan(zr_s)]
    n_valid_r = len(valid_zr)

    zr_max_pat  = float(np.max(valid_zr))   if n_valid_r >= 1 else 0.0
    zr_top2_pat = float(np.mean(np.sort(valid_zr)[-2:])) if n_valid_r >= 2 else zr_max_pat
    zr_avg_pat  = float(np.mean(valid_zr))  if n_valid_r >= 1 else 0.0

    env_r_max  = np.exp(zr_max_pat  + trend_s)
    env_r_top2 = np.exp(zr_top2_pat + trend_s)
    env_r_avg  = np.exp(zr_avg_pat  + trend_s)

    # Fenêtre 3 sur zr
    env_r_win3_max = np.full(n, np.nan)
    for i in range(n):
        m, _ = _window_stats(zr_s, i, 1)
        env_r_win3_max[i] = np.exp(m + trend_s[i]) if not np.isnan(m) else np.nan

    # Cummax forward et backward de zr
    env_r_cummax_fwd = np.full(n, np.nan)
    env_r_cummax_bwd = np.full(n, np.nan)
    running_max = -np.inf
    for i in range(n):
        if not np.isnan(zr_s[i]):
            running_max = max(running_max, zr_s[i])
        if running_max > -np.inf:
            env_r_cummax_fwd[i] = np.exp(running_max + trend_s[i])

    running_max = -np.inf
    for i in range(n - 1, -1, -1):
        if not np.isnan(zr_s[i]):
            running_max = max(running_max, zr_s[i])
        if running_max > -np.inf:
            env_r_cummax_bwd[i] = np.exp(running_max + trend_s[i])

    # Rater stats per-row (r_med, r_sd, r_q10, r_q90)
    if rater_stats is not None:
        r_med_arr = np.array([rater_stats["r_med"].get(r, rater_stats["global_med"]) for r in rater_s], dtype=float)
        r_sd_arr  = np.array([rater_stats["r_sd"].get(r,  rater_stats["global_sd"])  for r in rater_s], dtype=float)
        r_q10_arr = np.array([rater_stats["r_q10"].get(r, rater_stats["global_q10"]) for r in rater_s], dtype=float)
        r_q90_arr_out = r_q90_arr
    else:
        r_med_arr = r_sd_arr = r_q10_arr = r_q90_arr_out = np.zeros(n)

    # ------------------------------------------------------------------ #
    # Remettre dans l'ordre d'origine
    # ------------------------------------------------------------------ #
    def _reorder(a):
        return a[inv_order]

    result = {
        # Enveloppe patient
        "env_pat_med":  _reorder(env_med),
        "env_pat_q75":  _reorder(env_q75),
        "env_pat_q90":  _reorder(env_q90),
        "env_pat_max":  _reorder(env_max),
        "env_pat_top2": _reorder(env_top2),
        "zd_std":       _reorder(np.full(n, zd_std)),
        "zd_n":         _reorder(np.full(n, float(n_valid))),
        # Enveloppe locale
        "env_win3_max": _reorder(win_max3),
        "env_win3_avg": _reorder(win_avg3),
        "env_win5_max": _reorder(win_max5),
        "env_win5_avg": _reorder(win_avg5),
        "env_win7_max": _reorder(win_max7),
        "env_win7_avg": _reorder(win_avg7),
        "env_cummax":   _reorder(cummax_fwd),
        "env_cummax_r": _reorder(cummax_bwd),
        "zd_lag_m2":    _reorder(zd_lag_m2),
        "zd_lag_m1":    _reorder(zd_lag_m1),
        "zd_lag_p1":    _reorder(zd_lag_p1),
        "zd_lag_p2":    _reorder(zd_lag_p2),
        # Fiabilité médecin per-row
        "r_med":        _reorder(r_med_arr),
        "r_sd":         _reorder(r_sd_arr),
        "r_q10":        _reorder(r_q10_arr),
        "r_q90":        _reorder(r_q90_arr_out),
        # Enveloppes zr
        "env_r_max":        _reorder(env_r_max),
        "env_r_top2":       _reorder(env_r_top2),
        "env_r_avg":        _reorder(env_r_avg),
        "env_r_win3_max":   _reorder(env_r_win3_max),
        "env_r_cummax":     _reorder(env_r_cummax_fwd),
        "env_r_cummax_r":   _reorder(env_r_cummax_bwd),
    }
    return result


def _add_mle_features(
    df: pd.DataFrame,
    density_fn,
    slope_median: float,
    slope_std: float,
    rater_stats: dict | None = None,
) -> pd.DataFrame:
    """Ajoute mle, mle_slope, z_resid et toutes les enveloppes à df (tous les patients).

    La boucle par patient est parallélisée avec joblib (n_jobs=-1).
    Les lignes sont remises dans l'ordre d'origine après traitement.
    """
    df = df.copy()
    z = _compute_z(df)

    mle_vals    = np.full(len(df), np.nan)
    slope_vals  = np.full(len(df), np.nan)
    z_resid     = np.full(len(df), np.nan)

    # Calcul MLE séquentiel (optimisation Nelder-Mead non-parallélisable facilement)
    # Les positions absolues sont nécessaires pour scatter-back
    patient_groups = []
    for pid, grp in df.groupby("patient_id"):
        pos   = df.index.get_indexer(grp.index)
        ages  = grp["age"].values
        z_pat = z.loc[grp.index].values
        age_mean = ages.mean()
        x     = ages - age_mean

        a, b, c = _fit_patient_mle_curve(z_pat, ages, density_fn, slope_median, slope_std)

        mu = a + b * x + c * x ** 2
        mle_vals[pos]   = np.exp(mu)
        slope_vals[pos] = b
        z_resid[pos]    = np.where(~np.isnan(z_pat), z_pat - mu, np.nan)

        patient_groups.append((pid, grp.index, z_pat, ages, slope_vals[pos], grp))

    df["mle"]       = mle_vals
    df["mle_slope"] = slope_vals
    df["z_resid"]   = z_resid

    # --- Enveloppes : traitement parallèle par patient ---
    # Prépare les arguments (arrays déjà calculés)
    tasks = []
    for pid, grp_idx, z_pat, ages, b_pat, grp in patient_groups:
        rater_ids_pat = grp["rater_id"].values if "rater_id" in grp.columns else np.zeros(len(grp), dtype=object)
        tasks.append((pid, grp_idx, z_pat, ages, b_pat, rater_ids_pat, rater_stats))

    results = Parallel(n_jobs=-1)(
        delayed(_process_one_patient)(pid, grp_idx, zp, ag, bp, ri, rs)
        for pid, grp_idx, zp, ag, bp, ri, rs in tasks
    )

    # Toutes les colonnes enveloppe
    env_cols = list(results[0].keys()) if results else []
    env_arrays = {col: np.full(len(df), np.nan) for col in env_cols}

    for (pid, grp_idx, *_), res in zip(tasks, results):
        pos = df.index.get_indexer(grp_idx)
        for col, arr in res.items():
            env_arrays[col][pos] = arr

    for col, arr in env_arrays.items():
        df[col] = arr

    return df


# %% [markdown]
# ## Feature engineering (base exp 14 + features MLE + enveloppes)

# %%
def add_extra_features_v16(X_fe, X_raw, y_train=None, saved_stats=None):
    """Features exp 15 avec enveloppes hautes et fiabilité médecin ajoutées."""
    df = X_fe.copy()
    df["patient_id"] = X_raw["patient_id"].values
    df["rater_id"]   = X_raw["rater_id"].values
    df["gene"]       = X_raw["gene"].fillna("Unknown").values
    df["cohort"]     = X_raw["cohort"].values

    # ---- Correction de réponse individuelle (AVANT toutes les features patient) ----
    df = _apply_patient_response_correction(df)

    # -- exp-06 lags --
    ds = df.sort_values(["patient_id", "age"])
    g = ds.groupby("patient_id")
    ds["off_corrected_prev2"]     = g["off_corrected"].shift(2)
    ds["on_corrected_prev2"]      = g["on_corrected"].shift(2)
    ds["off_lag_delta"]           = ds["off_corrected"] - ds["off_corrected_prev"]
    ds["inter_visit_gap"]         = (ds["age"] - g["age"].shift(1)).clip(lower=0)
    ds["off_lag_delta_per_year"]  = ds["off_lag_delta"] / ds["inter_visit_gap"].replace(0, np.nan)
    ds["off_corrected_next2"]     = g["off_corrected"].shift(-2)
    ds = ds.loc[df.index]
    for c in ["off_corrected_prev2", "on_corrected_prev2", "off_lag_delta",
              "inter_visit_gap", "off_lag_delta_per_year", "off_corrected_next2"]:
        df[c] = ds[c].values

    df["patient_on_max"]  = df["patient_id"].map(df.groupby("patient_id")["on_corrected"].max())
    df["patient_on_q75"]  = df["patient_id"].map(df.groupby("patient_id")["on_corrected"].quantile(0.75))
    df["patient_on_std"]  = df["patient_id"].map(df.groupby("patient_id")["on_corrected"].std().fillna(0))

    on_slopes = {}
    for pid, gp in df.groupby("patient_id"):
        v = gp[["on_corrected", "age"]].dropna()
        on_slopes[pid] = float(np.polyfit(v["age"].values, v["on_corrected"].values, 1)[0]) if len(v) >= 2 else np.nan
    med_on_sl = float(np.nanmedian(list(on_slopes.values())))
    for pid in on_slopes:
        if np.isnan(on_slopes[pid]):
            on_slopes[pid] = med_on_sl
    df["patient_on_slope"] = df["patient_id"].map(on_slopes)

    df["ledd_off_ratio"]     = df["ledd"] / df["off_corrected"].replace(0, np.nan)
    df["ledd_per_off_trend"] = df["ledd"] / df["patient_trend_at_age"].replace(0, np.nan)
    df["patient_off_iqr"]    = df["patient_id"].map(
        df.groupby("patient_id")["off_corrected"].apply(
            lambda x: float(np.nanpercentile(x.dropna().values, 75) - np.nanpercentile(x.dropna().values, 25))
            if x.notna().sum() >= 2 else 0.0
        )
    )

    ds2 = df.sort_values(["patient_id", "age"])
    g2  = ds2.groupby("patient_id")
    ds2["off_cummin"]             = g2["off_corrected"].cummin()
    ds2["on_time_corrected_prev"] = g2["on_time_corrected"].shift(1)
    ds2 = ds2.loc[df.index]
    df["off_cummin"]             = ds2["off_cummin"].values
    df["on_time_corrected_prev"] = ds2["on_time_corrected_prev"].values

    # -- exp-07 --
    sl2 = {}
    for pid, gp in df.groupby("patient_id"):
        v = gp[["off_corrected", "age"]].dropna()
        sl2[pid] = float(np.polyfit(v["age"].values, v["off_corrected"].values, 2)[0]) if len(v) >= 4 else np.nan
    med_sl2 = float(np.nanmedian(list(sl2.values())))
    for pid in sl2:
        if np.isnan(sl2[pid]):
            sl2[pid] = med_sl2
    df["patient_slope2"] = df["patient_id"].map(sl2)

    ds3 = df.sort_values(["patient_id", "age"])
    g3  = ds3.groupby("patient_id")
    ds3["off_residual_prev"]      = g3["off_residual_from_trend"].shift(1)
    ds3["on_time_corrected_next"] = g3["on_time_corrected"].shift(-1)
    ds3["ledd_prev"]              = g3["ledd"].shift(1)
    ds3["ledd_next"]              = g3["ledd"].shift(-1)
    ds3["off_lag_accel"]          = (ds3["off_corrected"] - ds3["off_corrected_prev"]) - \
                                    (ds3["off_corrected_prev"] - ds3["off_corrected_prev2"])
    ds3 = ds3.loc[df.index]
    for c in ["off_residual_prev", "on_time_corrected_next", "ledd_prev", "ledd_next", "off_lag_accel"]:
        df[c] = ds3[c].values

    df["patient_ledd_mean"] = df["patient_id"].map(df.groupby("patient_id")["ledd"].mean())
    df["patient_ledd_max"]  = df["patient_id"].map(df.groupby("patient_id")["ledd"].max())
    pl_std = df.groupby("patient_id")["ledd"].std().fillna(1.0)
    df["patient_ledd_std"]  = df["patient_id"].map(pl_std)
    df["ledd_zscore"]       = (df["ledd"] - df["patient_ledd_mean"]) / df["patient_ledd_std"].replace(0, np.nan)

    # -- exp-08 fold-aware stats --
    if y_train is not None:
        dfy = df.copy()
        dfy["_t"] = y_train.values
        rm, rs, rc = {}, {}, {}
        for rid, gr in dfy.groupby("rater_id"):
            v = gr["_t"].dropna()
            rm[rid] = float(v.mean()) if len(v) > 0 else np.nan
            rs[rid] = float(v.std())  if len(v) > 1 else 0.0
            rc[rid] = len(v)
        grm = float(np.nanmean(list(rm.values())))
        grs = float(np.nanmean(list(rs.values())))
        gom = df.groupby("gene")["off_corrected"].mean().to_dict()
        gsm = {g_: float(np.mean([on_slopes[p] for p in gr["patient_id"].unique() if p in on_slopes]) or np.nan)
               for g_, gr in df.groupby("gene")}
        com = df.groupby("cohort")["off_corrected"].mean().to_dict()
        cos = df.groupby("cohort")["off_corrected"].std().to_dict()
        csm = {c_: float(np.mean([on_slopes[p] for p in gr["patient_id"].unique() if p in on_slopes]) or np.nan)
               for c_, gr in df.groupby("cohort")}
        saved_stats_out = {"rm": rm, "rs": rs, "rc": rc, "grm": grm, "grs": grs,
                           "gom": gom, "gsm": gsm, "com": com, "cos": cos, "csm": csm}
    else:
        saved_stats_out = None

    st = saved_stats if saved_stats is not None else saved_stats_out
    if st:
        df["rater_target_mean"]     = df["rater_id"].map(st["rm"]).fillna(st["grm"])
        df["rater_target_std"]      = df["rater_id"].map(st["rs"]).fillna(st["grs"])
        df["rater_target_count"]    = df["rater_id"].map(st["rc"]).fillna(0)
        df["gene_off_mean_feat"]    = df["gene"].map(st["gom"])
        df["gene_slope_mean_feat"]  = df["gene"].map(st["gsm"])
        df["cohort_off_mean_feat"]  = df["cohort"].map(st["com"])
        df["cohort_off_std_feat"]   = df["cohort"].map(st["cos"])
        df["cohort_slope_mean_feat"]= df["cohort"].map(st["csm"])
    else:
        for c in ["rater_target_mean", "rater_target_std", "rater_target_count",
                  "gene_off_mean_feat", "gene_slope_mean_feat",
                  "cohort_off_mean_feat", "cohort_off_std_feat", "cohort_slope_mean_feat"]:
            df[c] = np.nan

    mn = df.groupby("patient_id")["age"].min()
    mx = df.groupby("patient_id")["age"].max()
    ot = (mx - mn).clip(lower=0.1)
    df["patient_observation_time"] = df["patient_id"].map(ot)
    df["visits_per_year"]          = df["patient_n_visits"] / df["patient_observation_time"]
    df["dd_ledd_interaction"]      = df["disease_duration"] * df["ledd"].fillna(0)
    df["dd_ledd_velocity"]         = df["disease_duration"] * df["ledd_velocity"].fillna(0)

    ds4 = df.sort_values(["patient_id", "age"])
    ds4["off_running_std"] = ds4.groupby("patient_id")["off_corrected"].expanding().std().reset_index(level=0, drop=True)
    ds4 = ds4.loc[df.index]
    df["off_running_std"] = ds4["off_running_std"].values

    # -- exp-09 --
    ds5 = df.sort_values(["patient_id", "age"])
    g5  = ds5.groupby("patient_id")
    sh_off = g5["off_corrected"].shift(1)
    sh_on  = g5["on_corrected"].shift(1)
    ds5["off_ewma"] = sh_off.groupby(ds5["patient_id"]).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    ds5["on_ewma"]  = sh_on.groupby(ds5["patient_id"]).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    ds5 = ds5.loc[df.index]
    df["off_ewma"] = ds5["off_ewma"].values
    df["on_ewma"]  = ds5["on_ewma"].values

    pk = {}
    for pid, gp in df.groupby("patient_id"):
        v = gp[["on_corrected", "on_time_corrected"]].dropna()
        if len(v) >= 2:
            pk[pid] = float((v["on_corrected"] / v["on_time_corrected"].replace(0, np.nan)).median())
        else:
            pk[pid] = np.nan
    med_pk = float(np.nanmedian(list(pk.values())))
    for pid in pk:
        if np.isnan(pk[pid]):
            pk[pid] = med_pk
    df["patient_pk_ratio"] = df["patient_id"].map(pk)

    if st:
        df["off_vs_gene_mean"]   = df["off_corrected"] - df["gene_off_mean_feat"]
        df["off_vs_cohort_mean"] = df["off_corrected"] - df["cohort_off_mean_feat"]
    else:
        df["off_vs_gene_mean"]   = np.nan
        df["off_vs_cohort_mean"] = np.nan

    df["visit_rev_index"] = df["patient_n_visits"] - df["visit_index"]

    ds6 = df.sort_values(["patient_id", "age"])
    sh_ledd = ds6.groupby("patient_id")["ledd"].shift(1)
    ds6["ledd_ewma"] = sh_ledd.groupby(ds6["patient_id"]).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    ds6 = ds6.loc[df.index]
    df["ledd_ewma"] = ds6["ledd_ewma"].values

    # -- exp-11b : courbe ON/OFF combinée --
    df = _add_onoff_curve(df)

    df = df.drop(columns=["patient_id", "rater_id", "gene", "cohort", "patient_ledd_std"])
    return df, saved_stats_out


# %%
EXP06_COLS = ["off_corrected_prev2", "on_corrected_prev2", "off_lag_delta", "inter_visit_gap",
              "off_lag_delta_per_year", "off_corrected_next2", "patient_on_max", "patient_on_q75",
              "patient_on_std", "patient_on_slope", "ledd_off_ratio", "ledd_per_off_trend",
              "patient_off_iqr", "off_cummin", "on_time_corrected_prev"]
EXP07_COLS = ["patient_slope2", "off_residual_prev", "on_time_corrected_next", "ledd_prev", "ledd_next",
              "off_lag_accel", "patient_ledd_mean", "patient_ledd_max", "ledd_zscore",
              "rater_target_mean", "rater_target_std", "rater_target_count"]
EXP08_COLS = ["gene_off_mean_feat", "gene_slope_mean_feat", "cohort_off_mean_feat", "cohort_off_std_feat",
              "cohort_slope_mean_feat", "patient_observation_time", "visits_per_year",
              "dd_ledd_interaction", "dd_ledd_velocity", "off_running_std"]
EXP09_COLS = ["off_ewma", "on_ewma", "patient_pk_ratio", "off_vs_gene_mean", "off_vs_cohort_mean",
              "visit_rev_index", "ledd_ewma"]
EXP11B_COLS = ["onoff_curve_at_age", "on_residual_from_curve", "off_residual_from_curve"]
EXP14_COLS = ["patient_k_response", "patient_k_n_visits"]
# Features MLE (exp 15)
EXP15_COLS = ["mle", "mle_slope", "z_resid"]
# Nouvelles features enveloppes (exp 16)
EXP16_ENV_COLS = [
    "env_pat_med", "env_pat_q75", "env_pat_q90", "env_pat_max", "env_pat_top2",
    "zd_std", "zd_n",
    "env_win3_max", "env_win3_avg", "env_win5_max", "env_win5_avg",
    "env_win7_max", "env_win7_avg",
    "env_cummax", "env_cummax_r",
    "zd_lag_m2", "zd_lag_m1", "zd_lag_p1", "zd_lag_p2",
    "r_med", "r_sd", "r_q10", "r_q90",
    "env_r_max", "env_r_top2", "env_r_avg",
    "env_r_win3_max", "env_r_cummax", "env_r_cummax_r",
]
# Features étage 1 (prédictions stacking)
STACK_COLS = ["s1_pred_curr", "s1_pred_prev", "s1_pred_next"]

ALL_COLS_BASE = FEATURE_COLS + EXP06_COLS + EXP07_COLS + EXP08_COLS + EXP09_COLS + EXP11B_COLS + EXP14_COLS + EXP15_COLS + EXP16_ENV_COLS
ALL_COLS      = ALL_COLS_BASE + STACK_COLS
print(f"Base features (étage 1): {len(ALL_COLS_BASE)}")
print(f"Total features (étage 2): {len(ALL_COLS)}")

# %% [markdown]
# ## Factories de modèles (identiques à exp 14)

# %%
def make_lgbm_stage1():
    """Étage 1 : LightGBM unique."""
    return LGBMRegressor(
        n_estimators=3000, learning_rate=0.008, min_child_samples=40,
        reg_lambda=2.0, num_leaves=127, subsample=0.85, colsample_bytree=0.85,
        subsample_freq=1, n_jobs=-1, random_state=42, verbose=-1,
    )

def make_lgbm_1():
    return LGBMRegressor(
        n_estimators=3000, learning_rate=0.008, min_child_samples=40,
        reg_lambda=2.0, num_leaves=127, subsample=0.85, colsample_bytree=0.85,
        subsample_freq=1, n_jobs=-1, random_state=42, verbose=-1,
    )

def make_lgbm_2():
    return LGBMRegressor(
        n_estimators=3000, learning_rate=0.008, min_child_samples=50,
        reg_lambda=1.5, num_leaves=95, subsample=0.80, colsample_bytree=0.80,
        subsample_freq=1, n_jobs=-1, random_state=2025, verbose=-1,
    )

def make_xgb():
    return XGBRegressor(
        n_estimators=3000, learning_rate=0.008, min_child_weight=40,
        reg_lambda=2.0, max_depth=6, subsample=0.85, colsample_bytree=0.85,
        tree_method="hist", n_jobs=-1, random_state=99, verbosity=0,
    )

# %% [markdown]
# ## Lissage (identique exp 14)

# %%
def _smooth_isotonic_per_patient(predictions: np.ndarray, X_val: pd.DataFrame) -> np.ndarray:
    preds = predictions.copy().astype(float)
    df = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds
    ir = IsotonicRegression(increasing=True)
    for _pid, grp in df.groupby("patient_id"):
        if len(grp) < 2:
            continue
        grp_sorted = grp.sort_values("age")
        fitted = ir.fit_transform(grp_sorted["age"].values, grp_sorted["_pred"].values)
        df.loc[grp_sorted.index, "_pred"] = fitted
    return df["_pred"].values


def smooth_poly2_then_isotonic(predictions: np.ndarray, X_val: pd.DataFrame) -> np.ndarray:
    p1 = smooth_by_age_poly(predictions, X_val, degree=2)
    return _smooth_isotonic_per_patient(p1, X_val)

# %% [markdown]
# ## Utilitaire : lag des prédictions étage 1 par patient

# %%
def _add_stage1_lag_features(
    X_raw: pd.DataFrame,
    s1_preds: np.ndarray,
) -> pd.DataFrame:
    """Ajoute s1_pred_curr/prev/next en triant par âge au sein de chaque patient."""
    df = X_raw[["patient_id", "age"]].copy()
    df["s1_pred_curr"] = s1_preds

    df_sorted = df.sort_values(["patient_id", "age"])
    grp = df_sorted.groupby("patient_id")
    df_sorted["s1_pred_prev"] = grp["s1_pred_curr"].shift(1)
    df_sorted["s1_pred_next"] = grp["s1_pred_curr"].shift(-1)
    df_sorted = df_sorted.loc[df.index]

    result = df.copy()
    result["s1_pred_prev"] = df_sorted["s1_pred_prev"].values
    result["s1_pred_next"] = df_sorted["s1_pred_next"].values
    return result[["s1_pred_curr", "s1_pred_prev", "s1_pred_next"]]

# %% [markdown]
# ## Wrapper sklearn stacking 2 étages (exp 16)

# %%
def _with_pid_rid(X_ext: pd.DataFrame, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Remet patient_id et rater_id (supprimés par add_extra_features_v16) pour les calculs MLE/enveloppes."""
    d = X_ext.copy()
    d["patient_id"] = X_raw["patient_id"].values
    d["rater_id"]   = X_raw["rater_id"].values
    return d


def _mle_stats_and_features(X_ext_fit, X_raw_fit, y_fit, others, rater_stats=None):
    """Estime densité + pente sur (X_ext_fit, y_fit), calcule rater_stats si absent,
    puis ajoute toutes les features MLE + enveloppes à X_ext_fit et aux others.
    Retourne (fit_out, [others_out], stats)."""
    d_fit = _with_pid_rid(X_ext_fit, X_raw_fit)
    y_ser = pd.Series(np.asarray(y_fit), index=d_fit.index)
    dens = _build_residual_density(d_fit, y_ser)
    s_med, s_std = _compute_log_slope_stats(d_fit, y_ser)

    # Calcule les stats rater sur le train si pas encore fait
    if rater_stats is None:
        z_fit = _compute_z(d_fit)
        rater_stats = _build_rater_stats(d_fit, z_fit, y_ser)

    fit_out = _add_mle_features(d_fit, dens, s_med, s_std, rater_stats=rater_stats).drop(
        columns=["patient_id", "rater_id"])
    outs = [
        _add_mle_features(_with_pid_rid(xe, xr), dens, s_med, s_std, rater_stats=rater_stats).drop(
            columns=["patient_id", "rater_id"])
        for xe, xr in others
    ]
    return fit_out, outs, (dens, s_med, s_std, rater_stats)


class StackingPipeline16(BaseEstimator, RegressorMixin):
    """Exp-16 : stacking 2 étages + correction k + features MLE + enveloppes hautes.

    Ajoute les enveloppes patient, locales, et la fiabilité médecin calculées
    à partir du résidu détrienté zd = z - mle_slope*(age - age_moyen).
    """

    def __init__(self, n_inner_folds: int = 4):
        self.n_inner_folds = n_inner_folds

    def _build_features(self, X, y=None, saved_stats=None):
        """Construit features 16 + retourne saved_stats."""
        if y is not None:
            X_fe, _ = build_features(X, y, X_val=None)
            X_ext, stats = add_extra_features_v16(X_fe, X, y_train=y)
        else:
            _, X_fe = build_features(self._X_fit, self._y_fit, X_val=X)
            X_ext, stats = add_extra_features_v16(X_fe, X, saved_stats=saved_stats)
        return X_ext, stats

    def _prepare_for_lgbm(self, X_ext, all_cols, cat_dtypes=None):
        """Sélectionne les colonnes et aligne les dtypes catégoriels."""
        X_sel = X_ext[all_cols].copy()
        if cat_dtypes is not None:
            for c, dtype in cat_dtypes.items():
                if c in X_sel.columns:
                    X_sel[c] = X_sel[c].astype(dtype)
        return X_sel

    def fit(self, X, y, sample_weight=None):
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        y_ser = pd.Series(y_arr)

        # ---- Etape 1 : features de base sur le fold train ----
        X_fe_tr, saved_stats = self._build_features(X, y=y)
        self._X_fit       = X
        self._y_fit       = y
        self._saved_stats = saved_stats

        # ---- Etapes 2-3 : densite/pente + rater_stats sur le train + features MLE/enveloppes ----
        X_fe_tr, _, mle_stats = _mle_stats_and_features(X_fe_tr, X, y_arr, [])
        self._density_fn, self._slope_median, self._slope_std, self._rater_stats = mle_stats

        X_base = X_fe_tr[ALL_COLS_BASE].copy()

        cat_cols = [c for c in ALL_COLS_BASE if str(X_base[c].dtype) == "category"]
        self._cat_dtypes = {c: X_base[c].dtype for c in cat_cols}
        self._cat_codes  = {c: X_base[c].cat.categories for c in cat_cols}

        # ---- Etape 4 : predictions etage 1 OOF sur le train ----
        patient_groups = X["patient_id"].values
        inner_gkf = GroupKFold(n_splits=self.n_inner_folds)
        s1_oof = np.full(len(X), np.nan)

        for inner_tr_idx, inner_val_idx in inner_gkf.split(X, y, groups=patient_groups):
            X_inner_raw_tr  = X.iloc[inner_tr_idx].reset_index(drop=True)
            y_inner_tr      = y_ser.iloc[inner_tr_idx].reset_index(drop=True)
            X_inner_raw_val = X.iloc[inner_val_idx].reset_index(drop=True)

            X_fe_inner_tr, _ = build_features(X_inner_raw_tr, y_inner_tr, X_val=None)
            X_ext_inner_tr, inner_stats = add_extra_features_v16(
                X_fe_inner_tr, X_inner_raw_tr, y_train=y_inner_tr)

            _, X_fe_inner_val = build_features(X_inner_raw_tr, y_inner_tr, X_val=X_inner_raw_val)
            X_ext_inner_val, _ = add_extra_features_v16(
                X_fe_inner_val, X_inner_raw_val, saved_stats=inner_stats)

            # densite/pente + rater_stats sur le train interne uniquement
            X_ext_inner_tr, (X_ext_inner_val,), _ = _mle_stats_and_features(
                X_ext_inner_tr, X_inner_raw_tr, y_inner_tr.values,
                [(X_ext_inner_val, X_inner_raw_val)])

            for c, dtype in self._cat_dtypes.items():
                if c in X_ext_inner_tr.columns:
                    X_ext_inner_tr[c] = X_ext_inner_tr[c].astype(dtype)
                if c in X_ext_inner_val.columns:
                    X_ext_inner_val[c] = X_ext_inner_val[c].astype(dtype)

            m_s1_inner = make_lgbm_stage1()
            m_s1_inner.fit(X_ext_inner_tr[ALL_COLS_BASE], y_inner_tr.values,
                           categorical_feature=cat_cols)
            s1_oof[inner_val_idx] = m_s1_inner.predict(X_ext_inner_val[ALL_COLS_BASE])

        # ---- Etape 5 : lags des predictions OOF ----
        lag_tr = _add_stage1_lag_features(X, s1_oof)

        # ---- Etape 6 : etage 1 complet sur tout le fold train ----
        self.m_s1_ = make_lgbm_stage1()
        self.m_s1_.fit(X_base, y_arr, categorical_feature=cat_cols)

        # ---- Etape 7 : features augmentees pour l'etage 2 ----
        X_aug = X_base.copy()
        for col in STACK_COLS:
            X_aug[col] = lag_tr[col].values

        X_xgb = X_aug.copy()
        for c, dtype in self._cat_dtypes.items():
            X_xgb[c] = X_xgb[c].cat.codes.astype("float32")

        # ---- Etape 8 : etage 2 ----
        self.m1_ = make_lgbm_1()
        self.m1_.fit(X_aug, y_arr, categorical_feature=cat_cols)
        self.m2_ = make_lgbm_2()
        self.m2_.fit(X_aug, y_arr, categorical_feature=cat_cols)
        self.mxgb_ = make_xgb()
        self.mxgb_.fit(X_xgb.values, y_arr)

        self.feature_names_in_ = ALL_COLS
        self.n_features_in_    = len(ALL_COLS)
        return self

    def predict(self, X):
        check_is_fitted(self, "m1_")

        X_ext_val, _ = self._build_features(X, saved_stats=self._saved_stats)
        X_ext_val = _add_mle_features(
            _with_pid_rid(X_ext_val, X),
            self._density_fn, self._slope_median, self._slope_std,
            rater_stats=self._rater_stats,
        ).drop(columns=["patient_id", "rater_id"])
        X_base_val = self._prepare_for_lgbm(X_ext_val, ALL_COLS_BASE, self._cat_dtypes)

        s1_val = self.m_s1_.predict(X_base_val)
        lag_val = _add_stage1_lag_features(X, s1_val)

        X_aug_val = X_base_val.copy()
        for col in STACK_COLS:
            X_aug_val[col] = lag_val[col].values

        p1 = self.m1_.predict(X_aug_val)
        p2 = self.m2_.predict(X_aug_val)

        X_xgb_val = X_aug_val.copy()
        for c, cats in self._cat_codes.items():
            cm = {v: i for i, v in enumerate(cats)}
            X_xgb_val[c] = X_aug_val[c].map(lambda x, cm=cm: cm.get(x, -1)).astype("float32")
        p_xgb = self.mxgb_.predict(X_xgb_val.values)

        preds_raw = (p1 + p2 + p_xgb) / 3.0
        return smooth_poly2_then_isotonic(preds_raw, X)

# %% [markdown]
# ## Evaluation

# %%
print("=" * 60)
print("Exp 16 — Enveloppes patient + fiabilité médecin")
print("=" * 60)

RUN_MANUAL_LOOP = False  # CrossValidationReport ci-dessous lance les folds

fold_rmses = []
for fold_idx, (tr_idx, val_idx) in enumerate(folds if RUN_MANUAL_LOOP else []):
    X_tr  = X_train.iloc[tr_idx].reset_index(drop=True)
    y_tr  = y_train.iloc[tr_idx].reset_index(drop=True)
    X_val = X_train.iloc[val_idx].reset_index(drop=True)
    y_val = y_train.iloc[val_idx].reset_index(drop=True)

    model = StackingPipeline16()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    rmse_fold = root_mean_squared_error(y_val, preds)
    fold_rmses.append(rmse_fold)
    print(f"  Fold {fold_idx + 1}  RMSE = {rmse_fold:.4f}")

if fold_rmses:
    mean_rmse = float(np.mean(fold_rmses))
    std_rmse  = float(np.std(fold_rmses))
    print("-" * 60)
    print(f"  Mean RMSE = {mean_rmse:.4f}  std = {std_rmse:.4f}")

# %% [markdown]
# ## CrossValidationReport

# %%
print("\nRunning CrossValidationReport exp 16 (n_jobs=-1)…")
report = skore.CrossValidationReport(
    StackingPipeline16(),
    X=X_train, y=y_train, splitter=folds, n_jobs=-1,
)
project.put("16_enveloppes", report)
print("Report saved: 16_enveloppes")
rmse_df = report.metrics.rmse()
print(rmse_df)

# Affiche le RMSE de chaque fold depuis le rapport
try:
    fold_rmse_vals = rmse_df["RMSE"].values if "RMSE" in rmse_df.columns else rmse_df.iloc[:, 0].values
    for i, v in enumerate(fold_rmse_vals):
        print(f"  Fold {i + 1}  RMSE = {v:.4f}")
    print(f"  Mean RMSE = {float(np.mean(fold_rmse_vals)):.4f}")
except Exception:
    pass

# %% [markdown]
# ## Génération de la soumission

# %%
from parkinsons.data import load_test_dataset

X_test = load_test_dataset(DATA_DIR)
print(f"Test: {X_test.shape}")

final_model = StackingPipeline16()
final_model.fit(X_train, y_train)
test_preds = final_model.predict(X_test)

# Clip à 0 et sauvegarde directement avec pandas
test_preds_clipped = np.clip(test_preds, 0, None)
submission = pd.DataFrame({
    "id": X_test["id"] if "id" in X_test.columns else X_test.index,
    "updrs_4": test_preds_clipped,
})
import pathlib
pathlib.Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/16_enveloppes.csv", index=False)
print("Submission saved: submissions/16_enveloppes.csv")
