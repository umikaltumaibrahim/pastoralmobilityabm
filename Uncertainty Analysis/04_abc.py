"""
PHASE 4 — ABC UNCERTAINTY QUANTIFICATION 
======================================================
Characterizes parameter uncertainty around the Phase 2 best-fit point
estimate using Approximate Bayesian Computation (ABC).

Reads:
    calibration_outputs_final/phase3_summary_final.json  — winner strategy
    calibration_outputs_final/<winner>/best_results.json — best parameters
    gps_data/gps_data.csv                                — GPS calibration data


Fitness function used is identical to calibration phase

Agent count is read from districts_pop vector via agent_allocation.py.
"""

import numpy as np
import pandas as pd
import subprocess
import os
import json
import time
import shutil
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata, ks_2samp
import warnings

from agent_allocation import load_agent_allocation

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR  = os.path.abspath(os.getcwd())
JAVA_EXE  = os.path.join(BASE_DIR, "jre", "bin", "java.exe")
CLASSPATH = (f"{os.path.join(BASE_DIR, 'output', 'complete_model.jar')};"
             f"{os.path.join(BASE_DIR, 'headless_lib', '*')}")
SCENARIO  = os.path.join(BASE_DIR, "Geography.rs")

# ABC settings 
ABC_ITERATIONS        = 300    # total parameter sets evaluated
PARALLEL_WORKERS      = 4      # parallel Java subprocesses
EPSILON_MULTIPLIER    = 1.05   # ε = (1 - F_best) * 1.05  (5% tolerance)
MIN_POSTERIOR_SAMPLES = 10     # fallback if fewer than this are accepted
PERTURB_STD           = 0.10   # Gaussian sigma as fraction of prior range

# Model settings 
# Calibration months: same as calibration
CALIB_MONTHS = [1, 4, 7, 8, 9, 10]

# GPS fix counts: same dict as calibration
GPS_FIX_COUNTS = {
    1: 270, 2: 15, 4: 179, 7: 272,
    8: 63, 9: 1046, 10: 138, 11: 835, 12: 779,
}
MIN_GPS_POINTS   = 50
SMOOTH_SIGMA     = 3
WEIGHT_THRESHOLD = 0.02
COMPARISON_RES   = 4000    

#Fixed parameters
FIXED_PARAMS = {
    "visitedWeight": 0.05,
    "randomWeight":  0.10,
}

#Calibrated parameter bounds that are same as in calibration
PARAM_BOUNDS = {
    "rainThreshold": (1.0, 100.0),
    "saviThreshold": (0.0, 0.6),
    "pMove":         (0.6, 1.0),
}

# POM fitness weights 
W_DENSITY  = 0.60
W_SAVI     = 0.20
W_RAINFALL = 0.20
assert abs(W_DENSITY + W_SAVI + W_RAINFALL - 1.0) < 1e-9


def get_gps_count(month):
    return GPS_FIX_COUNTS.get(month, MIN_GPS_POINTS)


# METRICS ENGINE
# Same as that in calibration


class MetricsEngine:

    def __init__(self, gps_df, eval_months=None):
        if eval_months is None:
            eval_months = CALIB_MONTHS
        self.eval_months = eval_months

        agg_dict = {'GPS_SAVI': 'mean', 'GPS_Rainfall': 'mean'}
        gps_df = (
            gps_df
            .groupby(['X', 'Y', 'month'], as_index=False)
            .agg({
                **agg_dict,
                **{c: 'first' for c in gps_df.columns
                   if c not in ['X', 'Y', 'month', 'GPS_SAVI', 'GPS_Rainfall']},
            })
        )
        counts = (gps_df.groupby(['X', 'Y', 'month'])
                        .size().reset_index(name='weight'))
        gps_df = gps_df.merge(counts, on=['X', 'Y', 'month'])

        xmin, xmax = gps_df['X'].min(), gps_df['X'].max()
        ymin, ymax = gps_df['Y'].min(), gps_df['Y'].max()
        self.x_bins = np.arange(xmin, xmax + COMPARISON_RES, COMPARISON_RES)
        self.y_bins = np.arange(ymin, ymax + COMPARISON_RES, COMPARISON_RES)

        self.gps_density  = {}
        self.weights      = {}
        self.gps_savi     = {}
        self.gps_rainfall = {}

        for m in self.eval_months:
            g = gps_df[gps_df['month'] == m].copy()
            if len(g) < MIN_GPS_POINTS:
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
            self.gps_density[m]  = H.flatten()
            self.weights[m]      = W.flatten()
            self.gps_savi[m]     = g['GPS_SAVI'].dropna().values.astype(float)
            self.gps_rainfall[m] = g['GPS_Rainfall'].dropna().values.astype(float)

    def weighted_spearman(self, a_flat, m):
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

    def _ks_score(self, abm_df, m, abm_col, gps_dict):
        abm_m = abm_df[abm_df['month'] == m].copy()
        if abm_col not in abm_m.columns:
            return 0.5
        a = abm_m[abm_col].dropna().values.astype(float)
        if len(a) < 3:
            return 0.5
        g = gps_dict.get(m, np.array([]))
        if len(g) < 3:
            return 0.5
        if len(np.unique(a)) < 2 or len(np.unique(g)) < 2:
            return 0.5
        ks, _ = ks_2samp(a, g)
        return float(1.0 - ks)

    def compute_fitness(self, abm_df):
        """
        F(θ) = 0.60 * density + 0.20 * savi + 0.20 * rainfall
        Returns scalar float in [0, 1].
        """
        monthly_dens = []
        monthly_savi = []
        monthly_rain = []
        monthly_wt   = []

        for m in self.eval_months:
            if m not in self.gps_density:
                continue
            abm_m = abm_df[abm_df['month'] == m]
            if len(abm_m) < 3:
                continue
            pts = abm_m[['X', 'Y']].values
            H, _, _ = np.histogram2d(
                pts[:, 0], pts[:, 1],
                bins=[self.x_bins, self.y_bins],
            )
            H = gaussian_filter(H.astype(float), sigma=SMOOTH_SIGMA)
            H = H / (H.sum() + 1e-12)

            monthly_dens.append(self.weighted_spearman(H.flatten(), m))
            monthly_savi.append(self._ks_score(abm_df, m, 'CurrentSavi',     self.gps_savi))
            monthly_rain.append(self._ks_score(abm_df, m, 'CurrentRainfall', self.gps_rainfall))
            monthly_wt.append(np.sqrt(get_gps_count(m)))

        if not monthly_dens:
            return 0.0

        wt = np.array(monthly_wt) / np.sum(monthly_wt)
        return float(
            W_DENSITY  * np.average(monthly_dens, weights=wt) +
            W_SAVI     * np.average(monthly_savi, weights=wt) +
            W_RAINFALL * np.average(monthly_rain, weights=wt)
        )


# ============================================================
# REPAST RUNNER
# ============================================================

def run_repast(params, strat, run_dir, agent_count):
    """Write XML and launch Repast headless. Returns CSV path or None."""
    os.makedirs(run_dir, exist_ok=True)
    tree = ET.parse(os.path.join(BASE_DIR, "batch", "batch_params.xml"))
    root = tree.getroot()
    for p in root.findall('parameter'):
        name = p.get('name')
        if name == 'movementStrategy':
            p.set('value', strat)
        elif name == 'numAgents':
            p.set('value', str(agent_count))
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
        [JAVA_EXE, "-Xmx2g"]
        + jvm
        + ["-cp", CLASSPATH,
           "repast.simphony.batch.BatchMain",
           "-params", "batch_params.xml",
           SCENARIO]
    )
    subprocess.run(
        cmd, cwd=run_dir, timeout=1200,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    csv = os.path.join(run_dir, "v5_custom_locations.csv")
    return csv if (os.path.exists(csv) and os.path.getsize(csv) > 0) else None


# ============================================================
# COLUMN NORMALISER
# ============================================================

def normalise(df, eval_months=None):
    """Standardise ABM CSV column names and filter to eval_months."""
    if eval_months is None:
        eval_months = CALIB_MONTHS
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
# ABC WORKER (runs in a separate process via multiprocessing)
# ============================================================

def abc_worker(task):
    """
    Single ABC iteration:
        1. Load GPS data and build MetricsEngine independently.
           (Each worker is a separate process. Re-loading GPS avoids
            shared-memory complications under Python's spawn context.)
        2. Merge perturbed calibrated params with fixed params.
        3. Write XML, launch Repast, read output CSV.
        4. Compute fitness → distance = 1 - fitness.
        5. Return (params_dict, distance). distance=1.0 on failure.
    """
    params_dict, strat, gps_path, agent_count = task

    gps_df = pd.read_csv(gps_path)
    engine = MetricsEngine(gps_df, eval_months=CALIB_MONTHS)

    run_id  = f"abc_{os.getpid()}_{int(time.time() * 1000) % 100000}"
    run_dir = os.path.join(BASE_DIR, "temp_runs", run_id)

    try:
        all_params = {**params_dict, **FIXED_PARAMS}
        csv        = run_repast(all_params, strat, run_dir, agent_count)
        if not csv:
            return params_dict, 1.0

        df = normalise(pd.read_csv(csv), eval_months=CALIB_MONTHS)
        if df is None:
            return params_dict, 1.0

        fitness  = engine.compute_fitness(df)
        distance = float(1.0 - fitness)
        print(f"  OK | dist={distance:.4f} | "
              f"rain={params_dict['rainThreshold']:.2f} | "
              f"savi={params_dict['saviThreshold']:.3f} | "
              f"pMove={params_dict['pMove']:.3f}")
        return params_dict, distance

    except Exception as e:
        print(f"  FAIL | {e}")
        return params_dict, 1.0
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)



# POSTERIOR SUMMARY


def summarise_posterior(accepted_df):
    """
    Per-parameter descriptive statistics over accepted samples.
    The posterior mean is used as the Phase 5/6 point estimate.
    """
    rows = []
    for param in PARAM_BOUNDS:
        vals = accepted_df[param].values
        rows.append({
            'Parameter': param,
            'Mean':      float(np.mean(vals)),
            'Std':       float(np.std(vals)),
            'Median':    float(np.median(vals)),
            'P5':        float(np.percentile(vals,  5)),
            'P95':       float(np.percentile(vals, 95)),
            'N_samples': len(vals),
        })
    return pd.DataFrame(rows)



# VISUALIZATIONS


def plot_distance_distribution(df_all, epsilon, out_dir):
    n_acc = (df_all['Distance'] <= epsilon).sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_all['Distance'], bins=30,
            color='#B5D4F4', edgecolor='white', label='All samples')
    ax.axvline(epsilon, color='#E24B4A', linestyle='--', linewidth=1.5,
               label=f'ε = {epsilon:.4f}')
    ax.set_xlabel("ABC distance  (1 − POM fitness)")
    ax.set_ylabel("Count")
    ax.set_title(f"ABC distance distribution — "
                 f"{n_acc}/{len(df_all)} accepted "
                 f"({100 * n_acc / len(df_all):.1f}%)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(out_dir, "distance_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_posterior_scatter(accepted_df, out_dir):
    params = list(PARAM_BOUNDS.keys())
    n      = len(params)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    for i, p_y in enumerate(params):
        for j, p_x in enumerate(params):
            ax = axes[i][j]
            if i == j:
                ax.hist(accepted_df[p_x], bins=15,
                        color='#378ADD', edgecolor='white')
                ax.set_xlabel(p_x, fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
            else:
                ax.scatter(accepted_df[p_x], accepted_df[p_y],
                           alpha=0.5, s=20, color='#378ADD')
                ax.set_xlabel(p_x, fontsize=9)
                ax.set_ylabel(p_y, fontsize=9)
    fig.suptitle("Posterior parameter distribution (accepted ABC samples)",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "posterior_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")



# MAIN


if __name__ == "__main__":

    print("=" * 70)
    print("PHASE 4 — ABC UNCERTAINTY QUANTIFICATION (vF_revised)")
    print("=" * 70)
    print(f"  Fitness function  : density={W_DENSITY} | SAVI={W_SAVI} | rainfall={W_RAINFALL}")
    print(f"  Trajectory metric : REMOVED")
    print(f"  Grid resolution   : {COMPARISON_RES} m ({COMPARISON_RES//1000} km)")
    print(f"  ABC iterations    : {ABC_ITERATIONS}")
    print(f"  Parallel workers  : {PARALLEL_WORKERS}")
    print(f"  Perturbation std  : {PERTURB_STD} × parameter range")
    print(f"  Epsilon multiplier: {EPSILON_MULTIPLIER}")
    print(f"  Min posterior size: {MIN_POSTERIOR_SAMPLES} (fallback)")
    print("=" * 70)

    os.makedirs(os.path.join(BASE_DIR, "temp_runs"), exist_ok=True)
    out_dir = os.path.join(BASE_DIR, "calibration_outputs_final", "phase4_abc")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load Phase 3 winner ----------------------------------
    p3_path = os.path.join(BASE_DIR, "calibration_outputs_final",
                           "phase3_summary_final.json")
    if not os.path.exists(p3_path):
        raise FileNotFoundError(
            f"phase3_summary_final.json not found.\n"
            f"Run Phase 3 (vF_revised) before Phase 4."
        )
    with open(p3_path) as f:
        phase3 = json.load(f)
    winner = phase3['winner']
    print(f"\n  Phase 3 winner   : {winner}")
    print(f"  P(Rank 1)        : {phase3['confidence']:.3f}")
    print(f"  Interpretation   : {phase3['interpretation']}")

    # ---- Load Phase 2 best parameters for winner -------------
    best_path = os.path.join(BASE_DIR, "calibration_outputs_final",
                             winner, "best_results.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"best_results.json not found for '{winner}'.\n"
            f"Run Phase 2 (vF_revised) for this strategy first."
        )
    with open(best_path) as f:
        best = json.load(f)

    best_calibrated = {k: best['best_params'][k] for k in PARAM_BOUNDS}
    best_fitness    = best['best_fitness']
    epsilon         = (1.0 - best_fitness) * EPSILON_MULTIPLIER

    print(f"\n  Phase 2 best fitness : {best_fitness:.4f}")
    print(f"  ABC epsilon          : {epsilon:.4f}")
    print(f"  Best calibrated params:")
    for k, v in best_calibrated.items():
        print(f"    {k:<18}: {v:.6f}")

    # ---- Load agent count ------------------------------------
    global AGENT_COUNT
    AGENT_COUNT, _ = load_agent_allocation(BASE_DIR)
    print(f"\n  Agent count: {AGENT_COUNT:,}")

    # ---- GPS file path ---------------------------------------
    gps_file = os.path.join(BASE_DIR, "gps_data", "gps_data.csv")
    if not os.path.exists(gps_file):
        raise FileNotFoundError(f"GPS data not found: {gps_file}")

    # ---- Build perturbation tasks ---------------------------
    # Seeded RNG ensures reproducible perturbations.
    # Each task is self-contained (GPS path + agent count) so
    # multiprocessing workers need no shared state.
    rng   = np.random.default_rng(seed=42)
    tasks = []
    for _ in range(ABC_ITERATIONS):
        perturbed = {}
        for param, val in best_calibrated.items():
            lo, hi = PARAM_BOUNDS[param]
            sigma  = (hi - lo) * PERTURB_STD
            perturbed[param] = float(np.clip(rng.normal(val, sigma), lo, hi))
        tasks.append((perturbed, winner, gps_file, AGENT_COUNT))

    # ---- Run ABC iterations ----------------------------------
    print(f"\n  Running {ABC_ITERATIONS} iterations "
          f"({PARALLEL_WORKERS} parallel workers)...")
    print(f"  Perturbation sigma:")
    for param in PARAM_BOUNDS:
        lo, hi = PARAM_BOUNDS[param]
        print(f"    {param:<18}: ±{(hi - lo) * PERTURB_STD:.4f}")

    with Pool(PARALLEL_WORKERS) as pool:
        raw_results = pool.map(abc_worker, tasks)

    # ---- Compile results -------------------------------------
    rows   = [{**p, 'Distance': d} for p, d in raw_results]
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(out_dir, "all_samples.csv"), index=False)
    print(f"\n  All samples : {len(df_all)} rows saved")
    print(f"  Distance range  : [{df_all['Distance'].min():.4f}, "
          f"{df_all['Distance'].max():.4f}]")

    # ---- Apply epsilon threshold -----------------------------
    accepted     = df_all[df_all['Distance'] <= epsilon].copy()
    fallback_used = False

    if len(accepted) < MIN_POSTERIOR_SAMPLES:
        print(f"\n  WARNING: Only {len(accepted)} samples accepted. "
              f"Retaining top {MIN_POSTERIOR_SAMPLES} by distance.")
        accepted      = df_all.nsmallest(MIN_POSTERIOR_SAMPLES, 'Distance').copy()
        fallback_used = True

    acc_rate = float(len(accepted) / len(df_all))
    print(f"  Accepted     : {len(accepted)} / {len(df_all)} ({acc_rate:.1%})")
    accepted.to_csv(os.path.join(out_dir, "posterior_samples.csv"), index=False)

    # ---- Posterior summary -----------------------------------
    df_summary = summarise_posterior(accepted)
    df_summary.to_csv(os.path.join(out_dir, "posterior_summary.csv"),
                      index=False)
    print(f"\n  Posterior parameter summary:")
    print(df_summary.to_string(index=False))

    # ---- Visualisations -------------------------------------
    plot_distance_distribution(df_all, epsilon, out_dir)
    plot_posterior_scatter(accepted, out_dir)

    # ---- Build validation summary JSON ---------------------
    # posterior_mean_params: 3 calibrated posterior means + 2 fixed.
      posterior_mean = {
        param: float(df_summary.loc[
            df_summary['Parameter'] == param, 'Mean'].values[0])
        for param in PARAM_BOUNDS
    }
    posterior_std = {
        param: float(df_summary.loc[
            df_summary['Parameter'] == param, 'Std'].values[0])
        for param in PARAM_BOUNDS
    }
    posterior_ci90 = {
        param: [
            float(df_summary.loc[df_summary['Parameter'] == param, 'P5'].values[0]),
            float(df_summary.loc[df_summary['Parameter'] == param, 'P95'].values[0]),
        ]
        for param in PARAM_BOUNDS
    }

    phase4_summary = {
        'version':               'final_spearman4km_no_trajectory',
        'winner_strategy':       winner,
        'best_fitness':          best_fitness,
        'epsilon':               epsilon,
        'epsilon_multiplier':    EPSILON_MULTIPLIER,
        'fallback_used':         fallback_used,
        'n_iterations':          ABC_ITERATIONS,
        'n_accepted':            len(accepted),
        'acceptance_rate':       acc_rate,
        'posterior_mean_params': {**posterior_mean, **FIXED_PARAMS},
        'posterior_std_params':  posterior_std,
        'posterior_ci90':        posterior_ci90,
        'posterior_summary':     df_summary.to_dict('records'),
        'fitness_weights': {
            'density_correlation': W_DENSITY,
            'savi_tracking':       W_SAVI,
            'rainfall_tracking':   W_RAINFALL,
        },
        'calib_months':          CALIB_MONTHS,
        'gps_fix_counts':        GPS_FIX_COUNTS,
        'agent_count':           AGENT_COUNT,
    }

    summary_path = os.path.join(out_dir, "phase4_summary_final.json")
    with open(summary_path, 'w') as f:
        json.dump(phase4_summary, f, indent=4)

    print(f"\n  Phase 4 summary saved -> {summary_path}")
    print(f"  (Phases 5 and 6 read phase4_summary_final.json)")

    print("\n" + "=" * 70)
    print("PHASE 4 (vF_revised) COMPLETE")
    print(f"  Posterior mean params (used in Phases 5 and 6):")
    for k, v in posterior_mean.items():
        std = posterior_std[k]
        ci  = posterior_ci90[k]
        print(f"    {k:<18}: {v:.4f}  "
              f"(std={std:.4f},  90% CI [{ci[0]:.4f}, {ci[1]:.4f}])")
    for k, v in FIXED_PARAMS.items():
        print(f"    {k:<18}: {v}  (fixed)")
    print("=" * 70)
