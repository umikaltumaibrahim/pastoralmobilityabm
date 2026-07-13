"""
OUT-OF-SAMPLE TEMPORAL VALIDATION WITH UNCONSTRAINED NULL MODEL
=========================================================
Validates the calibrated model against GPS data withheld from all
calibration steps. Calibration used months [1,4,7,8,9,10]. This phase
uses November (835 fixes) and December (779 fixes) — withheld entirely
from earlier phases

Reads:
    calibration_outputs_final/phase4_abc/phase4_summary_final.json
    gps_data/gps_data.csv  (months 11 and 12 only)


Primary validation metrics (POM-consistent):
    density_correlation  — weighted Spearman on 4 km grid
    savi_tracking        — two-sample KS on CurrentSavi vs GPS_SAVI
    rainfall_tracking    — two-sample KS on CurrentRainfall vs GPS_Rainfall

Supplementary validation metrics (not in calibration fitness):
    bhattacharyya        — KDE-based Bhattacharyya coefficient [0,1]

Density surfaces averaged across replicates before
        scoring to remove stochastic placement noise from metric computation.
Monte Carlo null model, 99 random placements per month; p_null < 0.05
      Equal month weighting for validation (Nov/Dec have similar fix counts).

Agent count: read dynamically from districts_pop vector via agent_allocation.py.
"""

import numpy as np
import pandas as pd
import subprocess
import os
import json
import shutil
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata, ks_2samp, pearsonr
from scipy.stats import gaussian_kde
import warnings

from agent_allocation import load_agent_allocation

warnings.filterwarnings('ignore')



# CONFIGURATION


BASE_DIR  = os.path.abspath(os.getcwd())
JAVA_EXE  = os.path.join(BASE_DIR, "jre", "bin", "java.exe")
CLASSPATH = (f"{os.path.join(BASE_DIR, 'output', 'complete_model.jar')};"
             f"{os.path.join(BASE_DIR, 'headless_lib', '*')}")
SCENARIO  = os.path.join(BASE_DIR, "Geography.rs")

# ---- Validation settings (unchanged from vF) ----------------
AGENT_COUNT = None    # populated at runtime from agent_allocation.py
REPLICATES  = 5       # stochastic replicates for variability assessment
VALID_MONTHS = [11, 12]
ALL_MONTHS   = list(range(1, 13))

# GPS fix counts (full dataset reference)
GPS_FIX_COUNTS = {
    1: 270, 2: 15, 4: 179, 7: 272,
    8: 63, 9: 1046, 10: 138, 11: 835, 12: 779,
}
MIN_GPS_POINTS = 50
MONTH_NAMES = {
    1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
    7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec',
}

# ---- Spatial grid (must match Phase 2) ----------------------
COMPARISON_RES   = 4000    # metres — 4 km grid
SMOOTH_SIGMA     = 3       # Gaussian smoothing cells
WEIGHT_THRESHOLD = 0.02    # relative density threshold for spatial mask

# ---- POM fitness weights 
W_DENSITY  = 0.60
W_SAVI     = 0.20
W_RAINFALL = 0.20

# ---- Null model and bootstrap settings ----------------------
N_NULL           = 99      # random placement runs per month
N_BOOTSTRAP      = 2000    # BCa bootstrap resamples
CI_LEVEL         = 0.95    # confidence interval level
CV_WARN_THRESHOLD = 0.15   # warn if CV > 15% across replicates
BASE_SEED        = 42      # base random seed (replicate r uses BASE_SEED + r)
NULL_MODEL_SEED  = 9999    # separate seed for null model runs

# ---- Metric classification thresholds -----------------------
# ('higher', [STRONG, ACCEPTABLE, WEAK]) or ('lower', [...])
METRIC_THRESHOLDS = {
    'density_spearman':  ('higher', [0.70, 0.55, 0.40]),
    'savi_tracking':     ('higher', [0.75, 0.60, 0.40]),
    'rainfall_tracking': ('higher', [0.75, 0.60, 0.40]),
    'bhattacharyya':     ('higher', [0.70, 0.50, 0.30]),
}


def get_gps_count(m):
    return GPS_FIX_COUNTS.get(m, MIN_GPS_POINTS)


def classify_metric(key, value):
    """Classify metric score as STRONG / ACCEPTABLE / WEAK / POOR."""
    if key not in METRIC_THRESHOLDS or value is None or np.isnan(value):
        return 'N/A'
    direction, (strong, acc, weak) = METRIC_THRESHOLDS[key]
    if direction == 'higher':
        return ('STRONG'     if value >= strong else
                'ACCEPTABLE' if value >= acc    else
                'WEAK'       if value >= weak   else 'POOR')
    else:
        return ('STRONG'     if value <= strong else
                'ACCEPTABLE' if value <= acc    else
                'WEAK'       if value <= weak   else 'POOR')


def bca_ci(data, statistic=np.mean, n_bootstrap=N_BOOTSTRAP,
           ci_level=CI_LEVEL, rng=None):
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval.
    Returns (lower, upper) tuple. Returns (nan, nan) for n < 2.
    """
    from scipy.special import ndtri, ndtr
    if rng is None:
        rng = np.random.default_rng(BASE_SEED)
    data = np.asarray(data)
    n    = len(data)
    if n < 2:
        return (np.nan, np.nan)

    obs  = statistic(data)
    boot = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    prop_less = np.clip(np.mean(boot < obs), 1e-10, 1 - 1e-10)
    z0        = ndtri(prop_less)

    jack      = np.array([statistic(np.delete(data, i)) for i in range(n)])
    jack_mean = jack.mean()
    num       = np.sum((jack_mean - jack) ** 3)
    den       = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a         = num / den if den != 0 else 0.0

    alpha     = (1 - ci_level) / 2

    def adj_pct(z):
        return ndtr(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo_pct = np.clip(adj_pct(ndtri(alpha))     * 100, 0, 100)
    hi_pct = np.clip(adj_pct(ndtri(1 - alpha)) * 100, 0, 100)
    return (float(np.percentile(boot, lo_pct)),
            float(np.percentile(boot, hi_pct)))



# VALIDATION ENGINE


class ValidationEngine:
    """
    Computes all primary and supplementary validation metrics against
    withheld GPS data (months 11 and 12).

    Month aggregation: equal weights across validation months.
        Nov (835 fixes) and Dec (779 fixes) have similar counts, making
        equal weighting more appropriate than sqrt-weighting here.
        (Calibration used sqrt-weighting because fix counts ranged from
        63 to 1046 — a much wider spread requiring explicit moderation.)

    All density and environmental metric logic is identical to Phase 2.
    Bhattacharyya coefficient is a validation-specific diagnostic only
    (not part of the calibration fitness function).
    """

    def __init__(self, gps_df, eval_months=None):
        if eval_months is None:
            eval_months = VALID_MONTHS
        self.eval_months = eval_months

        # Collapse duplicates with mean aggregation on environmental columns
        agg = {'GPS_SAVI': 'mean', 'GPS_Rainfall': 'mean'}
        gps_df = (
            gps_df
            .groupby(['X', 'Y', 'month'], as_index=False)
            .agg({
                **agg,
                **{c: 'first' for c in gps_df.columns
                   if c not in ['X', 'Y', 'month', 'GPS_SAVI', 'GPS_Rainfall']},
            })
        )
        cnt    = (gps_df.groupby(['X', 'Y', 'month'])
                        .size().reset_index(name='weight'))
        gps_df = gps_df.merge(cnt, on=['X', 'Y', 'month'])
        self.gps_df = gps_df

        xmin, xmax = gps_df['X'].min(), gps_df['X'].max()
        ymin, ymax = gps_df['Y'].min(), gps_df['Y'].max()
        self.x_bins = np.arange(xmin, xmax + COMPARISON_RES, COMPARISON_RES)
        self.y_bins = np.arange(ymin, ymax + COMPARISON_RES, COMPARISON_RES)
        self.n_cells = (len(self.x_bins) - 1) * (len(self.y_bins) - 1)
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

        self.gps_density  = {}
        self.weights      = {}
        self.gps_centroids = {}
        self.gps_pts      = {}
        self.gps_savi     = {}
        self.gps_rainfall = {}

        print(f"\n  Validation GPS summary ({len(eval_months)} months):")
        for m in self.eval_months:
            g = gps_df[gps_df['month'] == m].copy()
            n = len(g)
            if n < MIN_GPS_POINTS:
                print(f"    Month {m:2d}: {n:4d} fixes — SKIPPED (< {MIN_GPS_POINTS})")
                continue

            pts = g[['X', 'Y']].values
            w   = g['weight'].values

            H, _, _ = np.histogram2d(
                pts[:, 0], pts[:, 1],
                bins=[self.x_bins, self.y_bins],
                weights=w,
            )
            H = gaussian_filter(H.astype(float), sigma=SMOOTH_SIGMA)
            H = H / (H.sum() + 1e-12)
            W = H.copy()
            W[W < WEIGHT_THRESHOLD * H.max()] = 0.0

            self.gps_density[m]   = H.flatten()
            self.weights[m]       = W.flatten()
            self.gps_centroids[m] = np.average(pts, weights=w, axis=0)
            self.gps_pts[m]       = pts
            self.gps_savi[m]      = g['GPS_SAVI'].dropna().values.astype(float)
            self.gps_rainfall[m]  = g['GPS_Rainfall'].dropna().values.astype(float)

            n_active = int(np.sum(W > 0))
            print(f"    Month {m:2d} ({MONTH_NAMES[m]}): {n:4d} fixes | "
                  f"active cells: {n_active:4,} / {self.n_cells:,} | "
                  f"SAVI pts: {len(self.gps_savi[m])} | "
                  f"Rain pts: {len(self.gps_rainfall[m])}")

    # ----------------------------------------------------------
    # Density surface builder (shared utility)
    # ----------------------------------------------------------

    def build_abm_density(self, a_pts):
        """Build Gaussian-smoothed sum-to-1 histogram on 4 km grid."""
        H, _, _ = np.histogram2d(
            a_pts[:, 0], a_pts[:, 1],
            bins=[self.x_bins, self.y_bins],
        )
        H = gaussian_filter(H.astype(float), sigma=SMOOTH_SIGMA)
        if H.sum() > 0:
            H = H / H.sum()
        return H

    # ----------------------------------------------------------
    # Primary metric 1: Density correlation (Spearman)
    # ----------------------------------------------------------

    def weighted_spearman(self, a_flat, m):
        """Weighted Spearman on GPS-active cells. Rescaled to [0, 1]."""
        mask = self.weights[m] > 0
        if mask.sum() < 10:
            return 0.5
        a, b, w = a_flat[mask], self.gps_density[m][mask], self.weights[m][mask]
        ra, rb  = rankdata(a), rankdata(b)
        sw = w.sum()
        ma = np.sum(w * ra) / sw
        mb = np.sum(w * rb) / sw
        cov = np.sum(w * (ra - ma) * (rb - mb))
        sa  = np.sqrt(np.sum(w * (ra - ma) ** 2))
        sb  = np.sqrt(np.sum(w * (rb - mb) ** 2))
        if sa == 0 or sb == 0:
            return 0.5
        return float((cov / (sa * sb) + 1.0) / 2.0)

    # ----------------------------------------------------------
    # Primary metric 2: SAVI tracking (KS)
    # ----------------------------------------------------------

    def compute_savi_tracking(self, abm_df, m):
        """KS score on SAVI distributions. Score = 1 - KS_distance."""
        abm_m = abm_df[abm_df['month'] == m].copy()
        if 'CurrentSavi' not in abm_m.columns:
            return 0.5
        a = abm_m['CurrentSavi'].dropna().values.astype(float)
        if len(a) < 3:
            return 0.5
        g = self.gps_savi.get(m, np.array([]))
        if len(g) < 3:
            return 0.5
        if len(np.unique(a)) < 2 or len(np.unique(g)) < 2:
            return 0.5
        ks, _ = ks_2samp(a, g)
        return float(1.0 - ks)

    # ----------------------------------------------------------
    # Primary metric 3: Rainfall tracking (KS)
    # ----------------------------------------------------------

    def compute_rainfall_tracking(self, abm_df, m):
        """KS score on rainfall distributions. Score = 1 - KS_distance."""
        abm_m = abm_df[abm_df['month'] == m].copy()
        if 'CurrentRainfall' not in abm_m.columns:
            return 0.5
        a = abm_m['CurrentRainfall'].dropna().values.astype(float)
        if len(a) < 3:
            return 0.5
        g = self.gps_rainfall.get(m, np.array([]))
        if len(g) < 3:
            return 0.5
        if len(np.unique(a)) < 2 or len(np.unique(g)) < 2:
            return 0.5
        ks, _ = ks_2samp(a, g)
        return float(1.0 - ks)

    # ----------------------------------------------------------
    # Primary validation fitness (3 metrics)
    # ----------------------------------------------------------

    def compute_fitness(self, abm_df):
        """
        Validation POM fitness using the same formula as calibration phase.
        Uses equal month weights because Nov and Dec have similar
        GPS fix counts (835 vs 779), unlike calibration months
        (63 to 1046) which required sqrt-weighting.
        """
        monthly_dens = []
        monthly_savi = []
        monthly_rain = []
        months_used  = []

        for m in self.eval_months:
            if m not in self.gps_density:
                continue
            abm_m = abm_df[abm_df['month'] == m]
            if len(abm_m) < 3:
                continue
            pts  = abm_m[['X', 'Y']].values
            H    = self.build_abm_density(pts).flatten()
            monthly_dens.append(self.weighted_spearman(H, m))
            monthly_savi.append(self.compute_savi_tracking(abm_df, m))
            monthly_rain.append(self.compute_rainfall_tracking(abm_df, m))
            months_used.append(m)

        if not months_used:
            return 0.0

        # Equal weights for validation months
        n_m = len(months_used)
        eq_wt = np.ones(n_m) / n_m
        fitness = (W_DENSITY  * np.average(monthly_dens, weights=eq_wt) +
                   W_SAVI     * np.average(monthly_savi, weights=eq_wt) +
                   W_RAINFALL * np.average(monthly_rain, weights=eq_wt))
        return float(fitness)

    # ----------------------------------------------------------
    # Supplementary metrics (validation-specific diagnostics)
    # ----------------------------------------------------------

    def compute_density_pearson(self, H_flat, m):
        """Pearson correlation restricted to GPS-active cells."""
        mask = self.weights[m] > 0
        if mask.sum() < 10:
            return np.nan
        a = H_flat[mask]
        b = self.gps_density[m][mask]
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        r, _ = pearsonr(a, b)
        return float(r) if not np.isnan(r) else np.nan

    def compute_bhattacharyya(self, a_pts, m):
        """
        KDE-based Bhattacharyya coefficient. Grid-resolution independent.
        BC = sum_k sqrt(p_abm_k * p_gps_k). Range [0, 1].
        """
        gps_pts = self.gps_pts.get(m)
        if gps_pts is None or len(a_pts) < 5 or len(gps_pts) < 5:
            return np.nan
        try:
            kde_abm = gaussian_kde(a_pts.T, bw_method='scott')
            kde_gps = gaussian_kde(gps_pts.T, bw_method='scott')
            x_lo = min(a_pts[:, 0].min(), gps_pts[:, 0].min())
            x_hi = max(a_pts[:, 0].max(), gps_pts[:, 0].max())
            y_lo = min(a_pts[:, 1].min(), gps_pts[:, 1].min())
            y_hi = max(a_pts[:, 1].max(), gps_pts[:, 1].max())
            xi   = np.linspace(x_lo, x_hi, 50)
            yi   = np.linspace(y_lo, y_hi, 50)
            xx, yy = np.meshgrid(xi, yi)
            grid   = np.vstack([xx.ravel(), yy.ravel()])
            p_abm  = kde_abm(grid)
            p_gps  = kde_gps(grid)
            p_abm  = p_abm / (p_abm.sum() + 1e-12)
            p_gps  = p_gps / (p_gps.sum() + 1e-12)
            return float(np.sum(np.sqrt(p_abm * p_gps)))
        except Exception:
            return np.nan

    # ----------------------------------------------------------
    # Null model (Monte Carlo random placement)
    # ----------------------------------------------------------

    def compute_null_distribution(self, m, agent_count,
                                  n_null=N_NULL, seed=NULL_MODEL_SEED):
        """
        Generate n_null uniform random placements within the GPS bounding
        box and score each on spatial metrics. Returns dict of null arrays.
        p_null < 0.05 for a metric means the model significantly
        outperforms random placement on that metric.
        """
        if m not in self.gps_density:
            return {}
        rng = np.random.default_rng(seed)
        null_spearman = []
        null_bhat     = []

        for _ in range(n_null):
            rx = rng.uniform(self.xmin, self.xmax, size=agent_count)
            ry = rng.uniform(self.ymin, self.ymax, size=agent_count)
            rand_pts = np.column_stack([rx, ry])
            H_null   = self.build_abm_density(rand_pts).flatten()
            null_spearman.append(self.weighted_spearman(H_null, m))
            null_bhat.append(self.compute_bhattacharyya(rand_pts, m))

        return {
            'null_spearman': np.array(null_spearman),
            'null_bhat':     np.array([x for x in null_bhat if not np.isnan(x)]),
        }

    # ----------------------------------------------------------
    # Ensemble metrics (averaged density surfaces)
    # ----------------------------------------------------------

    def compute_ensemble_metrics(self, density_surfaces_by_month,
                                 abm_pts_by_month=None):
        """
        Average density surfaces across replicates per month, then
        compute spatial metrics on the ensemble mean surface.
        This removes stochastic placement noise from the spatial comparison.
        """
        monthly_sp   = []
        monthly_bhat = []
        monthly_wt   = []
        per_month_ens = {}

        for m in self.eval_months:
            if m not in self.gps_density:
                continue
            surfaces = density_surfaces_by_month.get(m, [])
            if not surfaces:
                continue

            ens_mean = np.mean(np.stack(surfaces, axis=0), axis=0)

            sp_ens   = self.weighted_spearman(ens_mean, m)
            bhat_ens = np.nan
            if abm_pts_by_month and m in abm_pts_by_month:
                stacked  = np.vstack(abm_pts_by_month[m])
                bhat_ens = self.compute_bhattacharyya(stacked, m)

            monthly_sp.append(sp_ens)
            monthly_bhat.append(bhat_ens if not np.isnan(bhat_ens) else 0.0)
            monthly_wt.append(get_gps_count(m))

            per_month_ens[m] = {
                'ens_density_spearman': sp_ens,
                'ens_bhattacharyya':    bhat_ens,
                'n_replicates':         len(surfaces),
            }

        if not monthly_sp:
            return {}, {}

        n_m    = len(monthly_sp)
        eq_wt  = np.ones(n_m) / n_m   # equal weights for validation months
        summary_ens = {
            'ens_density_spearman': float(np.average(monthly_sp,   weights=eq_wt)),
            'ens_bhattacharyya':    float(np.average(monthly_bhat, weights=eq_wt)),
        }
        return summary_ens, per_month_ens

    # ----------------------------------------------------------
    # Full per-replicate metrics
    # ----------------------------------------------------------

    def compute_all_metrics(self, abm_df):
        """
        Compute all primary and supplementary validation metrics for one
        replicate. Returns (summary, per_month, density_surfs, abm_pts, centroids).

        density_surfs and abm_pts are stored for ensemble averaging across
        replicates in the main function.
        """
        monthly_sp   = []
        monthly_savi = []
        monthly_rain = []
        monthly_bhat = []
        monthly_wt   = []
        abm_centroids = {}
        months_used   = []
        per_month     = {}
        density_surfs = {}
        abm_pts_store = {}

        for m in self.eval_months:
            if m not in self.gps_density:
                continue
            abm_m = abm_df[abm_df['month'] == m]
            if len(abm_m) < 3:
                continue

            a_pts = abm_m[['X', 'Y']].values
            H     = self.build_abm_density(a_pts).flatten()

            density_surfs[m]  = H
            abm_pts_store[m]  = a_pts

            sp   = self.weighted_spearman(H, m)
            bhat = self.compute_bhattacharyya(a_pts, m)
            sv   = self.compute_savi_tracking(abm_df, m)
            rn   = self.compute_rainfall_tracking(abm_df, m)

            abm_centroids[m] = a_pts.mean(axis=0)
            months_used.append(m)

            monthly_sp.append(sp)
            monthly_savi.append(sv)
            monthly_rain.append(rn)
            monthly_bhat.append(bhat if not np.isnan(bhat) else 0.0)
            monthly_wt.append(np.sqrt(get_gps_count(m)))

            per_month[m] = {
                'density_spearman':   sp,
                'savi_tracking':      sv,
                'rainfall_tracking':  rn,
                'bhattacharyya':      bhat,
                'class_spearman':     classify_metric('density_spearman',  sp),
                'class_savi':         classify_metric('savi_tracking',      sv),
                'class_rainfall':     classify_metric('rainfall_tracking',  rn),
                'class_bhat':         classify_metric('bhattacharyya',      bhat),
            }

        if not months_used:
            return {}, {}, {}, {}, {}

        # Equal weights — validation months have similar fix counts
        n_m   = len(months_used)
        eq_wt = np.ones(n_m) / n_m

        def wavg(lst):
            return float(np.average(lst, weights=eq_wt))

        summary = {
            'density_correlation': wavg(monthly_sp),
            'savi_tracking':       wavg(monthly_savi),
            'rainfall_tracking':   wavg(monthly_rain),
            'bhattacharyya':       wavg(monthly_bhat),
            'n_months_used':       n_m,
        }
        return summary, per_month, density_surfs, abm_pts_store, abm_centroids


# ============================================================
# REPAST RUNNER AND COLUMN NORMALISER
# ============================================================

def run_repast_simulation(params, strategy_name, run_dir,
                          seed=42, agent_count=None):
    """Write XML and launch Repast headless. Returns CSV path or None."""
    if agent_count is None:
        agent_count = AGENT_COUNT
    if agent_count is None:
        raise RuntimeError("AGENT_COUNT not set.")
    os.makedirs(run_dir, exist_ok=True)

    tree = ET.parse(os.path.join(BASE_DIR, "batch", "batch_params.xml"))
    root = tree.getroot()
    for p in root.findall('parameter'):
        name = p.get('name')
        if name == 'movementStrategy':
            p.set('value', strategy_name)
        elif name == 'numAgents':
            p.set('value', str(agent_count))
        elif name == 'randomSeed':
            p.set('value', str(seed))
        elif name in params:
            p.set('value', str(round(float(params[name]), 6)))
    tree.write(os.path.join(run_dir, "batch_params.xml"))

    jvm = [
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "--add-opens", "java.base/java.util=ALL-UNNAMED",
        "--add-opens", "java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens", "java.base/java.text=ALL-UNNAMED",
        "--add-opens", "java.desktop/java.awt.font=ALL-UNNAMED",
    ]
    cmd = (
        [JAVA_EXE, "-Xmx4g"]
        + jvm
        + ["-cp", CLASSPATH,
           "repast.simphony.batch.BatchMain",
           "-params", "batch_params.xml",
           SCENARIO]
    )
    result = subprocess.run(cmd, cwd=run_dir, timeout=1800)
    if result.returncode != 0:
        return None
    csv_path = os.path.join(run_dir, "v5_custom_locations.csv")
    return (csv_path
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            else None)


def normalise_columns(df, eval_months=None):
    """Standardise column names and filter to eval_months."""
    if eval_months is None:
        eval_months = ALL_MONTHS
    col_map = {
        c: ('X'     if c.lower() == 'x'               else
            'Y'     if c.lower() == 'y'               else
            'month' if c.lower() in ('tick', 'month') else c)
        for c in df.columns
    }
    df = df.rename(columns=col_map)
    if not {'X', 'Y', 'month'}.issubset(df.columns):
        return None
    df = df[df['month'] != 0].copy()
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
    df = df.dropna(subset=['month'])
    df = df[df['month'].isin(eval_months)].copy()
    for col in ('CurrentSavi', 'CurrentRainfall'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df if not df.empty else None


# ============================================================
# VISUALIZATIONS
# ============================================================

def plot_spatial_overlay(abm_df, gps_df, strategy, month, out_dir):
    abm_m = abm_df[abm_df['month'] == month]
    gps_m = gps_df[gps_df['month'] == month]
    if len(abm_m) == 0 or len(gps_m) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(gps_m['X'], gps_m['Y'],
               c='#185FA5', alpha=0.6, s=12,
               label=f'GPS survey (n={len(gps_m)})', zorder=3)
    ax.scatter(abm_m['X'], abm_m['Y'],
               c='#E24B4A', alpha=0.3, s=8,
               label=f'ABM agents (n={len(abm_m)})', zorder=2)
    mname = MONTH_NAMES.get(month, f'Month {month}')
    ax.set_title(f"Spatial validation — {strategy}\n"
                 f"{mname} (out-of-sample)", fontsize=12)
    ax.set_xlabel("X (UTM metres)")
    ax.set_ylabel("Y (UTM metres)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"spatial_map_month_{month:02d}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_density_comparison(abm_df, engine, month, out_dir):
    abm_m = abm_df[abm_df['month'] == month]
    if len(abm_m) < 3 or month not in engine.gps_density:
        return
    a_pts  = abm_m[['X', 'Y']].values
    H_abm  = engine.build_abm_density(a_pts)
    ny     = len(engine.y_bins) - 1
    nx     = len(engine.x_bins) - 1
    H_gps  = engine.gps_density[month].reshape(ny, nx)
    mname  = MONTH_NAMES.get(month, f'Month {month}')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title in zip(
        axes,
        [H_abm, H_gps],
        [f'ABM density — {mname}', f'GPS density — {mname}'],
    ):
        im = ax.imshow(data, origin='lower', cmap='YlOrRd', aspect='auto')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Grid column (4 km cells)")
        ax.set_ylabel("Grid row (4 km cells)")
        plt.colorbar(im, ax=ax, label='Probability density')
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"density_comparison_month_{month:02d}.png"),
        dpi=150, bbox_inches='tight',
    )
    plt.close()


def plot_null_comparison(abm_scores, null_dists, month, out_dir):
    metrics_plot = [
        ('null_spearman', 'Weighted Spearman', 'higher'),
        ('null_bhat',     'Bhattacharyya',      'higher'),
    ]
    abm_key_map = {
        'null_spearman': 'density_spearman',
        'null_bhat':     'bhattacharyya',
    }
    mname = MONTH_NAMES.get(month, f'Month {month}')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"Null model comparison — {mname} "
                 f"(n={N_NULL} random runs)", fontsize=11)
    for ax, (null_key, label, direction) in zip(axes, metrics_plot):
        null_vals = null_dists.get(null_key, np.array([]))
        abm_val   = abm_scores.get(abm_key_map[null_key], np.nan)
        if len(null_vals) == 0 or np.isnan(abm_val):
            ax.set_visible(False)
            continue
        ax.hist(null_vals, bins=20, color='#AAAAAA',
                edgecolor='white', alpha=0.8, label='Null (random)')
        ax.axvline(abm_val, color='#C0392B', lw=2.0,
                   label=f'ABM = {abm_val:.3f}')
        if direction == 'higher':
            p_null = float(np.mean(null_vals >= abm_val))
        else:
            p_null = float(np.mean(null_vals <= abm_val))
        ax.set_title(f"{label}\np_null = {p_null:.3f}"
                     + (" ✓" if p_null < 0.05 else " ✗"), fontsize=9)
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"null_comparison_month_{month:02d}.png"),
        dpi=150, bbox_inches='tight',
    )
    plt.close()


def plot_metric_bootstrap(df_reps, metric_cols, out_dir):
    n_metrics = len(metric_cols)
    if n_metrics == 0:
        return
    fig, axes = plt.subplots(
        1, min(n_metrics, 5), figsize=(14, 4), squeeze=False
    )
    axes = axes.flatten()
    fig.suptitle("Replicate scores with 95% BCa bootstrap CIs (vF_revised)",
                 fontsize=11)
    rng = np.random.default_rng(BASE_SEED)
    for ax, col in zip(axes, metric_cols):
        vals = df_reps[col].dropna().values
        if len(vals) < 2:
            ax.set_visible(False)
            continue
        lo, hi = bca_ci(vals, rng=rng)
        mean   = np.mean(vals)
        ax.scatter(range(1, len(vals) + 1), vals,
                   color='#2166AC', s=50, zorder=4)
        ax.axhline(mean, color='#333', lw=1.5,
                   label=f'Mean={mean:.3f}')
        ax.fill_between([0.5, len(vals) + 0.5], lo, hi,
                        alpha=0.20, color='#2166AC',
                        label=f'95% BCa [{lo:.3f},{hi:.3f}]')
        ax.set_xticks(range(1, len(vals) + 1))
        ax.set_xticklabels([f'R{i}' for i in range(1, len(vals) + 1)],
                           fontsize=8)
        short = col.replace('density_', '').replace('_tracking', '')[:14]
        ax.set_title(short, fontsize=9)
        ax.legend(fontsize=6)
    for ax in axes[n_metrics:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bootstrap_ci_metrics.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# OVERALL 
# ============================================================

def compute_overall_verdict(summary_mean, null_p_values):
    """
    Classify overall validation outcome as PASS / WARN / FAIL.
    Based on classification of primary metrics and null p-values.
    """
    primary = ['density_correlation', 'savi_tracking',
               'rainfall_tracking', 'bhattacharyya']
    key_map = {
        'density_correlation': 'density_spearman',
        'savi_tracking':       'savi_tracking',
        'rainfall_tracking':   'rainfall_tracking',
        'bhattacharyya':       'bhattacharyya',
    }
    classifications = {
        col: classify_metric(key_map.get(col, col), summary_mean.get(col, np.nan))
        for col in primary
    }
    classes = list(classifications.values())
    p_null  = null_p_values.get('p_null_density', 1.0)

    if 'POOR' in classes:
        verdict = 'FAIL'
        reason  = (f"Primary metric(s) POOR: "
                   f"{[k for k,v in classifications.items() if v=='POOR']}")
    elif p_null >= 0.05:
        verdict = 'WARN'
        reason  = (f"Density not distinguishable from random "
                   f"(p_null = {p_null:.3f} ≥ 0.05)")
    elif 'WEAK' in classes:
        verdict = 'WARN'
        reason  = (f"Primary metric(s) WEAK: "
                   f"{[k for k,v in classifications.items() if v=='WEAK']}")
    else:
        verdict = 'PASS'
        reason  = "All primary metrics ACCEPTABLE+ and p_null < 0.05"

    return {
        'verdict':            verdict,
        'reason':             reason,
        'classifications':    classifications,
        'p_null_spearman':    p_null,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("PHASE 5 — OUT-OF-SAMPLE VALIDATION (vF_revised)")
    print("=" * 72)
    print(f"  Validation months : {VALID_MONTHS} (Nov, Dec — withheld from calibration)")
    print(f"  Fitness function  : density={W_DENSITY} | SAVI={W_SAVI} | rainfall={W_RAINFALL}")
    print(f"  Trajectory metric : REMOVED")
    print(f"  Grid resolution   : {COMPARISON_RES} m ({COMPARISON_RES//1000} km)")
    print(f"  Month weighting   : equal (Nov/Dec have similar fix counts)")
    print(f"  Replicates        : {REPLICATES}")
    print(f"  Null model runs   : {N_NULL} per month")
    print(f"  Bootstrap samples : {N_BOOTSTRAP}")
    print("=" * 72)

    # ---- Load Phase 4 summary ---------------------------------
    p4_path = os.path.join(BASE_DIR, "calibration_outputs_final",
                           "phase4_abc", "phase4_summary_final.json")
    if not os.path.exists(p4_path):
        raise FileNotFoundError(
            f"phase4_summary_final.json not found.\n"
            f"Run Phase 4 (vF_revised) before Phase 5."
        )
    with open(p4_path) as f:
        phase4 = json.load(f)

    strategy = phase4['winner_strategy']
    params   = phase4['posterior_mean_params']
    print(f"\n  Winner strategy      : {strategy}")
    print(f"  Posterior mean params: {params}")

    # ---- Load GPS validation data ----------------------------
    gps_path = os.path.join(BASE_DIR, "gps_data", "gps_data.csv")
    if not os.path.exists(gps_path):
        raise FileNotFoundError(f"GPS data not found: {gps_path}")
    gps_full = pd.read_csv(gps_path)
    required = {'X', 'Y', 'month', 'GPS_SAVI', 'GPS_Rainfall'}
    missing  = required - set(gps_full.columns)
    if missing:
        raise ValueError(f"GPS data missing columns: {missing}")

    gps_valid = gps_full[gps_full['month'].isin(VALID_MONTHS)].copy()
    print(f"\n  GPS validation rows : {len(gps_valid)} (months {VALID_MONTHS})")

    # ---- Build validation engine ------------------------------
    engine = ValidationEngine(gps_valid, eval_months=VALID_MONTHS)

    # ---- Load agent count ------------------------------------
    AGENT_COUNT, _ = load_agent_allocation(BASE_DIR)
    print(f"\n  Agent count: {AGENT_COUNT:,}")

    # ---- Pre-compute null model distributions ----------------
    print(f"\n  Pre-computing null distributions ({N_NULL} runs per month)...")
    null_dists = {}
    for m in VALID_MONTHS:
        if m in engine.gps_density:
            null_dists[m] = engine.compute_null_distribution(m, AGENT_COUNT)

    # ---- Run replicates --------------------------------------
    out_dir = os.path.join(BASE_DIR, "calibration_outputs_final",
                           "phase5_validation")
    os.makedirs(out_dir, exist_ok=True)

    replicate_results    = []
    replicate_details    = []
    density_surfs_bymon  = {m: [] for m in VALID_MONTHS}
    abm_pts_bymon        = {m: [] for m in VALID_MONTHS}

    for rep in range(1, REPLICATES + 1):
        print(f"\n  Running replicate {rep}/{REPLICATES} ...")
        rep_seed = BASE_SEED + rep
        run_dir_rep = os.path.join(BASE_DIR, "temp_runs",
                                   f"validation_rep_{rep}")
        os.makedirs(run_dir_rep, exist_ok=True)

        agent_csv = run_repast_simulation(
            params, strategy, run_dir_rep,
            seed=rep_seed, agent_count=AGENT_COUNT,
        )
        if not agent_csv:
            print(f"  !! Replicate {rep} failed — simulation produced no output.")
            shutil.rmtree(run_dir_rep, ignore_errors=True)
            continue

        rep_out = os.path.join(out_dir, f"replicate_{rep}")
        os.makedirs(rep_out, exist_ok=True)

        df_raw  = pd.read_csv(agent_csv)
        df_full = normalise_columns(df_raw.copy(), eval_months=ALL_MONTHS)
        if df_full is not None:
            df_full.to_csv(
                os.path.join(rep_out, "agent_locations_all_months.csv"),
                index=False,
            )

        df_val = normalise_columns(df_raw.copy(), eval_months=VALID_MONTHS)
        if df_val is None:
            print(f"  !! Replicate {rep}: no rows for months {VALID_MONTHS}.")
            shutil.rmtree(run_dir_rep, ignore_errors=True)
            continue
        df_val.to_csv(os.path.join(rep_out, "agent_locations.csv"),
                      index=False)

        summary, per_month, d_surfs, a_pts, _ = \
            engine.compute_all_metrics(df_val)
        if not summary:
            print(f"  !! Replicate {rep}: metrics could not be computed.")
            shutil.rmtree(run_dir_rep, ignore_errors=True)
            continue

        # Accumulate density surfaces and point clouds for ensemble
        for m in VALID_MONTHS:
            if m in d_surfs:
                density_surfs_bymon[m].append(d_surfs[m])
            if m in a_pts:
                abm_pts_bymon[m].append(a_pts[m])

        summary['replicate'] = rep
        summary['seed']      = rep_seed
        replicate_results.append(summary)
        replicate_details.append({'replicate': rep, 'per_month': per_month})

        # Composite POM fitness (distinct from the density-only correlation
        # printed below it) — recomputed here for the per-replicate log line.
        val_fitness = engine.compute_fitness(df_val)
        print(f"  Rep {rep} — POM fitness={val_fitness:.4f} | "
              f"density={summary['density_correlation']:.4f} | "
              f"savi={summary['savi_tracking']:.4f} | "
              f"rainfall={summary['rainfall_tracking']:.4f}")

        # Visualisations: spatial overlay and density comparison (rep 1 only)
        if rep == 1:
            for m in VALID_MONTHS:
                plot_spatial_overlay(df_val, gps_valid, strategy, m, out_dir)
                plot_density_comparison(df_val, engine, m, out_dir)
                if m in null_dists and m in per_month:
                    plot_null_comparison(per_month[m], null_dists[m], m, out_dir)

        shutil.rmtree(run_dir_rep, ignore_errors=True)

    if not replicate_results:
        raise RuntimeError("All replicates failed. Check Java environment.")

    # ---- Aggregate per-replicate results ----------------------
    df_reps = pd.DataFrame(replicate_results)
    df_reps.to_csv(os.path.join(out_dir, "validation_metrics.csv"),
                   index=False)

    # Bootstrap CIs on primary metrics
    primary_metric_cols = [
        'density_correlation', 'savi_tracking',
        'rainfall_tracking', 'bhattacharyya',
    ]
    plot_cols = [c for c in primary_metric_cols if c in df_reps.columns]
    plot_metric_bootstrap(df_reps, plot_cols, out_dir)

    # Summary statistics with BCa CIs
    summary_rows = []
    rng_boot     = np.random.default_rng(BASE_SEED)
    for col in df_reps.columns:
        if col in ('replicate', 'seed', 'n_months_used'):
            continue
        vals = df_reps[col].dropna().values
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        cv   = (std / abs(mean) * 100) if mean != 0 else 0.0
        lo, hi = bca_ci(vals, rng=rng_boot)
        cv_status = "PASS" if cv <= CV_WARN_THRESHOLD * 100 else "WARN"
        summary_rows.append({
            'Metric':    col,
            'Mean':      round(mean, 4),
            'Std':       round(std, 4),
            'CV_%':      round(cv, 2),
            'CI95_lo':   round(lo, 4),
            'CI95_hi':   round(hi, 4),
            'CV_status': cv_status,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "validation_summary.csv"),
                      index=False)

    # ---- Ensemble metrics ------------------------------------
    ens_surfs = {m: s for m, s in density_surfs_bymon.items() if s}
    ens_pts   = {m: p for m, p in abm_pts_bymon.items() if p}
    ens_summary, ens_per_month = engine.compute_ensemble_metrics(
        ens_surfs, abm_pts_by_month=ens_pts
    )
    if ens_summary:
        pd.DataFrame([
            {'Metric': k, 'Value': round(float(v), 6)}
            for k, v in ens_summary.items()
        ]).to_csv(os.path.join(out_dir, "ensemble_metrics.csv"), index=False)

    # ---- Null p-values ---------------------------------------
    null_p_values = {}
    check_pairs   = [
        ('density_correlation', 'null_spearman', 'higher'),
        ('bhattacharyya',       'null_bhat',     'higher'),
    ]
    for metric_key, null_key, direction in check_pairs:
        abm_mean = (df_reps[metric_key].mean()
                    if metric_key in df_reps.columns else np.nan)
        all_null = []
        for m in VALID_MONTHS:
            if m in null_dists and null_key in null_dists[m]:
                all_null.extend(null_dists[m][null_key])
        if len(all_null) > 0 and not np.isnan(abm_mean):
            all_null = np.array(all_null)
            p = (float(np.mean(all_null >= abm_mean)) if direction == 'higher'
                 else float(np.mean(all_null <= abm_mean)))
            short = metric_key.split('_')[0]
            null_p_values[f'p_null_{short}'] = p

    pd.DataFrame([null_p_values]).to_csv(
        os.path.join(out_dir, "null_pvalues.csv"), index=False
    )

    # ---- Overall verdict -------------------------------------
    summary_mean = df_reps[primary_metric_cols].mean().to_dict()
    verdict_dict = compute_overall_verdict(summary_mean, null_p_values)

    # ---- Print validation report -----------------------------
    print("\n" + "=" * 72)
    print("VALIDATION REPORT — Phase 5 (vF_revised)")
    print("=" * 72)
    print(f"  Strategy          : {strategy}")
    print(f"  Validation months : {VALID_MONTHS} (Nov, Dec — out-of-sample)")
    print(f"  Completed reps    : {len(replicate_results)} / {REPLICATES}")
    print(f"  Grid resolution   : {COMPARISON_RES} m | Trajectory: REMOVED")
    print()
    print("  Metric summary (mean ± std | 95% BCa CI):")
    print(df_summary.to_string(index=False))
    print()
    print("  Null model p-values (p < 0.05 = model beats random placement):")
    for k, v in null_p_values.items():
        sig = "✓" if v < 0.05 else "✗"
        print(f"    {k}: {v:.3f} {sig}")
    print()
    print("  Ensemble metrics (mean of replicate density surfaces):")
    for k, v in ens_summary.items():
        print(f"    {k}: {v:.4f}")
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  OVERALL VERDICT: {verdict_dict['verdict']:<31}║")
    print(f"  ║  {verdict_dict['reason'][:50]:<50}║")
    print("  ╚══════════════════════════════════════════════════╝")

    # ---- Save Phase 5 summary JSON ---------------------------
    phase5_summary = {
        'version':                  'final_spearman4km_no_trajectory',
        'strategy':                 strategy,
        'validation_months':        VALID_MONTHS,
        'n_replicates':             len(replicate_results),
        'agent_count':              AGENT_COUNT,
        'posterior_params':         params,
        'metrics_mean':             {r['Metric']: r['Mean']
                                     for _, r in df_summary.iterrows()},
        'metrics_std':              {r['Metric']: r['Std']
                                     for _, r in df_summary.iterrows()},
        'metrics_ci95':             {r['Metric']: [r['CI95_lo'], r['CI95_hi']]
                                     for _, r in df_summary.iterrows()},
        'ensemble_metrics':         ens_summary,
        'null_pvalues':             null_p_values,
        'overall_verdict':          verdict_dict,
        'fitness_weights': {
            'density_correlation': W_DENSITY,
            'savi_tracking':       W_SAVI,
            'rainfall_tracking':   W_RAINFALL,
        },
        'month_weighting':          'equal (Nov/Dec similar fix counts)',
        'gps_fix_counts':           GPS_FIX_COUNTS,
    }

    summary_path = os.path.join(out_dir, "phase5_summary_final.json")
    with open(summary_path, 'w') as f:
        json.dump(phase5_summary, f, indent=4, default=str)

    print(f"\n  Summary saved -> {summary_path}")
    print(f"  All outputs    -> {out_dir}")
    print("\n  Phase 5 (vF_revised) complete.")
