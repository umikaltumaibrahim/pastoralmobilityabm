"""
PHASE 2 — BAYESIAN OPTIMISATION CALIBRATION (vF_revised)
=========================================================
Calibrates three behavioural decision parameters for four competing
pastoral mobility strategies using Bayesian optimisation (TPE, Optuna).

Calibrated parameters:
    rainThreshold   — minimum rainfall improvement (mm) to trigger movement
    saviThreshold   — minimum SAVI improvement to trigger movement
    pMove           — probability of executing a confirmed move decision

Fixed parameters (written to XML every run; never optimized):
    visitedWeight = 0.05   — mild revisitation preference in tie-breaking
    randomWeight  = 0.10   — stochastic noise coefficient in tie-breaking

POM fitness function (3 metrics):
    F(θ) = 0.60 * density_correlation
         + 0.20 * savi_tracking
         + 0.20 * rainfall_tracking

Spatial configuration:
    COMPARISON_RES = 4000 m  — 4 km grid, matches model SEARCH_STEP_METERS
    SMOOTH_SIGMA   = 3 cells — Gaussian smoothing radius of 12 km at 4 km

Parameter bounds (revised):
    rainThreshold : [0, 100] mm   — expanded to accommodate Gu/Deyr rainfall
    saviThreshold : [0, 0.6]      — expanded upper bound for wet-season SAVI

Calibration months: [1, 4, 7, 8, 9, 10]  (Jan, Apr, Jul, Aug, Sep, Oct)
Validation months:  [11, 12]              (Nov, Dec — withheld for validation)

Agent count: read dynamically from districts_pop vector file via
    agent_allocation.py

"""

import optuna
import numpy as np
import pandas as pd
import subprocess
import os
import json
import shutil
import xml.etree.ElementTree as ET
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata, ks_2samp
import warnings

from agent_allocation import load_agent_allocation

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR  = os.path.abspath(os.getcwd())
JAVA_EXE  = os.path.join(BASE_DIR, "jre", "bin", "java.exe")
JAR_PATH  = os.path.join(BASE_DIR, "output", "complete_model.jar")
LIB_PATH  = os.path.join(BASE_DIR, "headless_lib", "*")
CLASSPATH = f"{JAR_PATH};{LIB_PATH}"
SCENARIO  = os.path.join(BASE_DIR, "Geography.rs")
OUT_DIR   = os.path.join(BASE_DIR, "calibration_outputs_final")
TEMP_DIR  = os.path.join(BASE_DIR, "temp_runs_final")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ---- Execution -----------------------------------------------
PARALLEL_WORKERS = 8
N_TRIALS         = 200 # Optuna trials per strategy; increase to 300 if
                           # density correlation ceiling is observed
AGENT_COUNT      = None   # Populated at runtime from agent_allocation.py

# ---- Month configuration ------------------------------------
# Calibration: Jan(1), Apr(4), Jul(7), Aug(8), Sep(9), Oct(10)
# Withheld for Phase 5 validation: Nov(11), Dec(12)
CALIB_MONTHS = [1, 4, 7, 8, 9, 10]
VALID_MONTHS = [11, 12]

# GPS fix counts per month — used for sqrt-weighting.
# Source: verified against gps_data.csv row counts.
GPS_FIX_COUNTS = {
    1: 270, 2: 15, 4: 179, 7: 272,
    8: 63, 9: 1046, 10: 138, 11: 835, 12: 779,
}

# ---- Spatial grid -------------------------------------------
# 4 km grid: matches model SEARCH_STEP_METERS for spatial consistency.
# Gaussian smoothing sigma = 3 cells = 12 km effective radius at 4 km.
MIN_GPS_POINTS   = 50      # months below this are excluded from fitness
SMOOTH_SIGMA     = 3       # Gaussian smoothing cells
WEIGHT_THRESHOLD = 0.02    # relative density threshold for spatial mask
COMPARISON_RES   = 4000    # metres — 4 km grid

# ---- Fixed parameters ----------------------------------------
# Both must be written to batch_params.xml on every run.
# NomadicHousehold.java reads all five parameters via params.getDouble();
# absence of either causes a ParameterNotFoundException crash.
FIXED_PARAMS = {
    "visitedWeight": 0.05,
    "randomWeight":  0.10,
}

# ---- Calibrated parameter search bounds ----------------------
# rainThreshold: expanded to [0, 100] to accommodate Gu/Deyr rainfall
# saviThreshold: expanded to [0, 0.6] to capture full seasonal SAVI range
PARAM_BOUNDS = {
    "rainThreshold": (0.0, 100.0),
    "saviThreshold": (0.0, 0.6),
    "pMove":         (0.6, 1.0),
}

# ---- POM fitness weights (no trajectory) --------------------
# Weights must sum exactly to 1.0.
# Density (0.60): primary spatial pattern test — highest weight because
#   spatial distribution agreement is the most comprehensive single measure.
# SAVI tracking (0.20): direct test of the primary decision hypothesis.
# Rainfall tracking (0.20): secondary environmental check.
W_DENSITY  = 0.60
W_SAVI     = 0.20
W_RAINFALL = 0.20
assert abs(W_DENSITY + W_SAVI + W_RAINFALL - 1.0) < 1e-9, \
    "Fitness weights must sum to 1.0"


def get_gps_count(month):
    """Return GPS fix count for a given month. Defaults to MIN_GPS_POINTS."""
    return GPS_FIX_COUNTS.get(month, MIN_GPS_POINTS)


# ============================================================
# METRICS ENGINE
# ============================================================

class MetricsEngine:
    """
    Pre-computes GPS reference structures once at initialisation.
    compute_fitness() is called once per Optuna trial.

    Three metrics, no trajectory:
        Metric 1 — Spatial density correlation (weighted Spearman, 4 km grid)
        Metric 2 — SAVI tracking fidelity (two-sample KS, direct GPS_SAVI)
        Metric 3 — Rainfall tracking fidelity (two-sample KS, GPS_Rainfall)

    Monthly scores are averaged with sqrt(n_gps) weights so that
    data-rich months (Sep: 1046 fixes) proportionally outweigh
    sparse months (Aug: 63 fixes) without completely suppressing them.
    """

    def __init__(self, gps_df, eval_months=None):
        if eval_months is None:
            eval_months = CALIB_MONTHS
        self.eval_months = eval_months

        # Collapse exact (X, Y, month) duplicates; average environmental values.
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
        # Attach duplicate counts as location weights
        counts = (gps_df.groupby(['X', 'Y', 'month'])
                        .size().reset_index(name='weight'))
        gps_df = gps_df.merge(counts, on=['X', 'Y', 'month'])

        # Build 4 km comparison grid from GPS coordinate extent
        xmin, xmax = gps_df['X'].min(), gps_df['X'].max()
        ymin, ymax = gps_df['Y'].min(), gps_df['Y'].max()
        self.x_bins = np.arange(xmin, xmax + COMPARISON_RES, COMPARISON_RES)
        self.y_bins = np.arange(ymin, ymax + COMPARISON_RES, COMPARISON_RES)

        # Per-month reference structures
        self.gps_density  = {}   # sum-to-1 GPS density surface (flat array)
        self.weights      = {}   # spatial focus mask (active cells only)
        self.gps_savi     = {}   # GPS_SAVI values for KS comparison
        self.gps_rainfall = {}   # GPS_Rainfall values for KS comparison

        for m in self.eval_months:
            g = gps_df[gps_df['month'] == m].copy()
            if len(g) < MIN_GPS_POINTS:
                continue

            pts = g[['X', 'Y']].values
            w   = g['weight'].values

            # Build Gaussian-smoothed sum-to-1 histogram density
            H, _, _ = np.histogram2d(
                pts[:, 0], pts[:, 1],
                bins=[self.x_bins, self.y_bins],
                weights=w,
            )
            H = gaussian_filter(H.astype(float), sigma=SMOOTH_SIGMA)
            H = H / (H.sum() + 1e-12)   # sum-to-1 normalisation

            # Spatial focus mask: suppress cells below 2% of distribution peak
            W = H.copy()
            W[W < WEIGHT_THRESHOLD * H.max()] = 0.0

            self.gps_density[m]  = H.flatten()
            self.weights[m]      = W.flatten()
            self.gps_savi[m]     = g['GPS_SAVI'].dropna().values.astype(float)
            self.gps_rainfall[m] = g['GPS_Rainfall'].dropna().values.astype(float)

    # ----------------------------------------------------------
    # Metric 1: Weighted Spearman rank correlation (Density)
    # ----------------------------------------------------------

    def weighted_spearman(self, a_flat, m):
        """
        Weighted Spearman correlation between ABM and GPS density surfaces,
        restricted to GPS-active cells (spatial focus mask W > 0).

        Spearman is preferred over Pearson because pastoral density
        distributions are heavily right-skewed. The correlation is rescaled
        from [-1, 1] to [0, 1] for use in the fitness sum.
        """
        mask = self.weights[m] > 0
        if mask.sum() < 10:
            return 0.5   # neutral — insufficient active cells
        a = a_flat[mask]
        b = self.gps_density[m][mask]
        w = self.weights[m][mask]

        ra, rb = rankdata(a), rankdata(b)
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
    # Metric 2: SAVI tracking fidelity (KS statistic)
    # ----------------------------------------------------------

    def compute_savi_tracking(self, abm_df, m):
        """
        Two-sample KS statistic comparing SAVI at ABM agent locations
        vs GPS_SAVI at survey locations. Score = 1 - KS_distance.
        Uses GPS_SAVI directly (no nearest-agent approximation).
        """
        abm_m = abm_df[abm_df['month'] == m].copy()
        if 'CurrentSavi' not in abm_m.columns:
            return 0.5
        a_savi = abm_m['CurrentSavi'].dropna().values.astype(float)
        if len(a_savi) < 3:
            return 0.5
        g_savi = self.gps_savi.get(m, np.array([]))
        if len(g_savi) < 3:
            return 0.5
        if len(np.unique(a_savi)) < 2 or len(np.unique(g_savi)) < 2:
            return 0.5
        ks_stat, _ = ks_2samp(a_savi, g_savi)
        return float(1.0 - ks_stat)

    # ----------------------------------------------------------
    # Metric 3: Rainfall tracking fidelity (KS statistic)
    # ----------------------------------------------------------

    def compute_rainfall_tracking(self, abm_df, m):
        """
        Two-sample KS statistic comparing rainfall at ABM agent locations
        vs GPS_Rainfall at survey locations. Score = 1 - KS_distance.
        """
        abm_m = abm_df[abm_df['month'] == m].copy()
        if 'CurrentRainfall' not in abm_m.columns:
            return 0.5
        a_rain = abm_m['CurrentRainfall'].dropna().values.astype(float)
        if len(a_rain) < 3:
            return 0.5
        g_rain = self.gps_rainfall.get(m, np.array([]))
        if len(g_rain) < 3:
            return 0.5
        if len(np.unique(a_rain)) < 2 or len(np.unique(g_rain)) < 2:
            return 0.5
        ks_stat, _ = ks_2samp(a_rain, g_rain)
        return float(1.0 - ks_stat)

    # ----------------------------------------------------------
    # Composite POM fitness (3 metrics)
    # ----------------------------------------------------------

    def compute_fitness(self, abm_df, return_raw=False):
        """
        F(θ) = 0.60 * density_avg + 0.20 * savi_avg + 0.20 * rain_avg

        Monthly scores averaged with sqrt(n_gps) weights.
        Returns scalar float in [0, 1]. If return_raw=True, also returns
        a dict of raw metric scores for logging.
        """
        monthly_density = []
        monthly_savi    = []
        monthly_rain    = []
        monthly_weights = []
        months_used     = []

        for m in self.eval_months:
            if m not in self.gps_density:
                continue
            abm_m = abm_df[abm_df['month'] == m]
            if len(abm_m) < 3:
                continue

            a_pts = abm_m[['X', 'Y']].values

            # Build 4 km density surface for this month
            H, _, _ = np.histogram2d(
                a_pts[:, 0], a_pts[:, 1],
                bins=[self.x_bins, self.y_bins],
            )
            H = gaussian_filter(H.astype(float), sigma=SMOOTH_SIGMA)
            H = H / (H.sum() + 1e-12)

            monthly_density.append(self.weighted_spearman(H.flatten(), m))
            monthly_savi.append(self.compute_savi_tracking(abm_df, m))
            monthly_rain.append(self.compute_rainfall_tracking(abm_df, m))
            monthly_weights.append(np.sqrt(get_gps_count(m)))
            months_used.append(m)

        if not months_used:
            return (0.0, {}) if return_raw else 0.0

        # sqrt(n)-weighted averages across qualifying months
        wt = np.array(monthly_weights)
        wt = wt / wt.sum()
        density_avg = float(np.average(monthly_density, weights=wt))
        savi_avg    = float(np.average(monthly_savi,    weights=wt))
        rain_avg    = float(np.average(monthly_rain,    weights=wt))

        fitness = W_DENSITY * density_avg + W_SAVI * savi_avg + W_RAINFALL * rain_avg

        raw_metrics = {
            'density_correlation': density_avg,
            'savi_tracking':       savi_avg,
            'rainfall_tracking':   rain_avg,
            'n_months_used':       len(months_used),
        }
        if return_raw:
            return float(fitness), raw_metrics
        return float(fitness)


# ============================================================
# REPAST SIMULATION RUNNER
# ============================================================

def run_repast_simulation(params, strategy_name, run_dir, agent_count=None):
    """
    Write batch_params.xml with all 5 parameters and launch Repast headless.
    Returns path to v5_custom_locations.csv, or None on failure.

    All five parameters (3 calibrated + 2 fixed) must be present in the XML.
    NomadicHousehold.java reads all via params.getDouble(); absence crashes.
    """
    if agent_count is None:
        agent_count = AGENT_COUNT
    if agent_count is None:
        raise RuntimeError("AGENT_COUNT not set — call load_agent_allocation first.")

    os.makedirs(run_dir, exist_ok=True)
    tree = ET.parse(os.path.join(BASE_DIR, "batch", "batch_params.xml"))
    root = tree.getroot()

    for p in root.findall('parameter'):
        name = p.get('name')
        if name == 'movementStrategy':
            p.set('value', strategy_name)
        elif name == 'numAgents':
            p.set('value', str(agent_count))
        elif name in params:
            p.set('value', str(round(float(params[name]), 6)))

    tree.write(os.path.join(run_dir, "batch_params.xml"))

    jvm_opens = [
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "--add-opens", "java.base/java.util=ALL-UNNAMED",
        "--add-opens", "java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens", "java.base/java.text=ALL-UNNAMED",
        "--add-opens", "java.desktop/java.awt.font=ALL-UNNAMED",
    ]
    cmd = (
        [JAVA_EXE, "-Xmx4g"]
        + jvm_opens
        + ["-cp", CLASSPATH,
           "repast.simphony.batch.BatchMain",
           "-params", "batch_params.xml",
           SCENARIO]
    )
    result = subprocess.run(cmd, cwd=run_dir, timeout=1800)
    if result.returncode != 0:
        print(f"  !! Java crash (return code {result.returncode}): {run_dir}")
        return None

    agent_csv = os.path.join(run_dir, "v5_custom_locations.csv")
    return (agent_csv
            if os.path.exists(agent_csv) and os.path.getsize(agent_csv) > 0
            else None)


# ============================================================
# COLUMN NORMALISER
# ============================================================

def normalise_columns(df, eval_months=None):
    """
    Standardise ABM CSV column names and filter to eval_months.
    Preserves CurrentSavi and CurrentRainfall as numeric for KS metrics.
    Excludes tick 0 (initialisation step, not a valid monthly position).
    """
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
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial, strategy_name, engine):
    """
    Single Optuna trial: sample 3 calibrated parameters, merge with
    2 fixed parameters, run ABM, compute 3-metric POM fitness.
    Returns scalar fitness in [0, 1].
    """
    calibrated = {
        name: trial.suggest_float(name, lo, hi)
        for name, (lo, hi) in PARAM_BOUNDS.items()
    }
    all_params = {**calibrated, **FIXED_PARAMS}

    run_dir = os.path.join(TEMP_DIR, f"trial_{trial.number}_{os.getpid()}")
    try:
        agent_csv = run_repast_simulation(
            all_params, strategy_name, run_dir, agent_count=AGENT_COUNT
        )
        if not agent_csv:
            return 0.0

        df = normalise_columns(pd.read_csv(agent_csv))
        if df is None:
            return 0.0

        fitness, raw_metrics = engine.compute_fitness(df, return_raw=True)

        # Log every trial to CSV for post-hoc convergence diagnostics
        log_file = os.path.join(OUT_DIR, strategy_name, "evaluation_log.csv")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_row = {
            'trial':    trial.number,
            'strategy': strategy_name,
            **all_params,
            **raw_metrics,
            'fitness':  fitness,
        }
        pd.DataFrame([log_row]).to_csv(
            log_file, mode='a',
            header=not os.path.exists(log_file),
            index=False,
        )
        return fitness

    except Exception as e:
        print(f"  !! Trial {trial.number} error: {e}")
        return 0.0
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ============================================================
# SAVE BEST TRIAL RESULTS
# ============================================================

def save_best_trial_results(study, strategy_name, engine):
    """
    Re-runs best parameter set, saves agent locations, computes
    final metrics, writes best_results.json and best_results.csv.
    """
    print(f"\n  Saving best results for {strategy_name} ...")
    best_calibrated = study.best_trial.params
    best_all_params = {**best_calibrated, **FIXED_PARAMS}

    strat_dir = os.path.join(OUT_DIR, strategy_name)
    os.makedirs(strat_dir, exist_ok=True)

    run_dir   = os.path.join(TEMP_DIR, f"best_run_{strategy_name}")
    agent_csv = run_repast_simulation(
        best_all_params, strategy_name, run_dir, agent_count=AGENT_COUNT
    )

    best_metrics = {}
    if agent_csv and os.path.exists(agent_csv):
        dest = os.path.join(strat_dir, "best_agent_locations.csv")
        shutil.copy2(agent_csv, dest)
        df = normalise_columns(pd.read_csv(agent_csv))
        if df is not None:
            _, best_metrics = engine.compute_fitness(df, return_raw=True)
    else:
        print(f"    WARNING: Best-run simulation failed — metrics unavailable.")

    results = {
        'version':              'final_spearman4km_no_trajectory',
        'strategy':             strategy_name,
        'best_fitness':         study.best_value,
        'best_params':          best_all_params,
        'calibrated_params':    best_calibrated,
        'fixed_params':         FIXED_PARAMS,
        'best_metrics':         best_metrics,
        'fitness_weights': {
            'density_correlation': W_DENSITY,
            'savi_tracking':       W_SAVI,
            'rainfall_tracking':   W_RAINFALL,
        },
        'comparison_grid_res_m': COMPARISON_RES,
        'parameter_bounds':      PARAM_BOUNDS,
        'calib_months':          CALIB_MONTHS,
        'valid_months':          VALID_MONTHS,
        'gps_fix_counts':        GPS_FIX_COUNTS,
        'n_trials':              len(study.trials),
        'agent_count':           AGENT_COUNT,
    }

    json_path = os.path.join(strat_dir, "best_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"    Saved -> {json_path}")

    pd.DataFrame([{
        'strategy':     strategy_name,
        'best_fitness': study.best_value,
        **best_all_params,
        **best_metrics,
    }]).to_csv(os.path.join(strat_dir, "best_results.csv"), index=False)

    shutil.rmtree(run_dir, ignore_errors=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PHASE 2 — BAYESIAN OPTIMISATION CALIBRATION (vF_revised)")
    print("=" * 70)
    print(f"  Grid resolution   : {COMPARISON_RES} m ({COMPARISON_RES//1000} km)")
    print(f"  Fitness function  : density={W_DENSITY}  SAVI={W_SAVI}  rainfall={W_RAINFALL}")
    print(f"  Trajectory metric : REMOVED")
    print(f"  Calibration months: {CALIB_MONTHS}")
    print(f"  Validation months : {VALID_MONTHS} (withheld for Phase 5)")
    print(f"  Optuna trials     : {N_TRIALS} per strategy")
    print(f"  Parallel workers  : {PARALLEL_WORKERS}")
    print()
    print(f"  Parameter bounds:")
    for name, (lo, hi) in PARAM_BOUNDS.items():
        print(f"    {name:<18}: [{lo}, {hi}]")
    print()
    print(f"  Fixed parameters:")
    for name, val in FIXED_PARAMS.items():
        print(f"    {name:<18}: {val}")
    print("=" * 70)

    # ---- Load GPS calibration data ----------------------------
    gps_path = os.path.join(BASE_DIR, "gps_data", "gps_data.csv")
    gps_data = pd.read_csv(gps_path)
    required_cols = {'X', 'Y', 'month', 'GPS_SAVI', 'GPS_Rainfall'}
    missing_cols  = required_cols - set(gps_data.columns)
    if missing_cols:
        raise ValueError(
            f"GPS data missing required columns: {missing_cols}\n"
            f"Expected: {required_cols}\n"
            f"Found: {list(gps_data.columns)}"
        )

    # ---- Load agent count from shared vector file -------------
    AGENT_COUNT, _ = load_agent_allocation(BASE_DIR)
    import sys
    sys.modules[__name__].AGENT_COUNT = AGENT_COUNT
    print(f"\n  Agent count loaded: {AGENT_COUNT:,}\n")

    # ---- Build GPS reference engine (calibration months) ------
    engine = MetricsEngine(gps_data, eval_months=CALIB_MONTHS)

    strategies = [
        "RainfallPriority_30km",
        "PasturePriority_30km", 
        "RainfallPriority_60km",
        "PasturePriority_60km",
    ]

    for strat in strategies:
        print(f"\n{'=' * 60}")
        print(f"  Strategy: {strat}")
        print("=" * 60)

        db_path = os.path.join(OUT_DIR, f"{strat}_final.db")
        study   = optuna.create_study(
            direction      = "maximize",
            storage        = f"sqlite:///{db_path}",
            study_name     = f"{strat}_final",
            load_if_exists = True,
        )
        completed = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        remaining = max(0, N_TRIALS - completed)

        if completed > 0:
            print(f"  Resuming: {completed} completed, {remaining} remaining.")
        if remaining > 0:
            study.optimize(
                lambda t: objective(t, strat, engine),
                n_trials = remaining,
                n_jobs   = PARALLEL_WORKERS,
            )

        print(f"\n  Best fitness : {study.best_value:.4f}")
        print("  Best calibrated params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k:<18}: {v:.6f}")
        save_best_trial_results(study, strat, engine)

    print("\n" + "=" * 70)
    print("PHASE 2 (vF_revised) COMPLETE")
    print(f"  Outputs: {OUT_DIR}")
    print("=" * 70)
