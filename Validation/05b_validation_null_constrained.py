"""
OUT-OF-SAMPLE TEMPORAL VALIDATION WITH NULL MODEL (LANDSCAPE‑CONSTRAINED)
===========================================
Computes null model p‑values using a landscape‑constrained null:
random points placed only on cells that satisfy Rangeland=1, Slope=1,
and non‑sedentary area.

Reads:
    - Existing ABM agent locations from Phase 5 replicates
    - GPS reference data (months 11, 12)
    - Raster files from C:\RepastData\
      - rangelands_resampled_utm.tif
      - slopepercent_reclass_utm.tif
      - nonsedentaryareas_utm.shp

Outputs (saved in separate folder):
    - null_pvalues_constrained.csv
    - null_pvalues_comparison.csv
    - phase5_summary_constrained_null.json
    - null_comparison_month_*_constrained.png
"""

import numpy as np
import pandas as pd
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata, gaussian_kde
import warnings
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling

warnings.filterwarnings('ignore')


# CONFIGURATION


BASE_DIR = os.path.abspath(os.getcwd())
OUT_DIR_ORIG = os.path.join(BASE_DIR, "calibration_outputs_final", "phase5_validation")
OUT_DIR_NEW = os.path.join(BASE_DIR, "calibration_outputs_final", "phase5_validation_null_constrained")
os.makedirs(OUT_DIR_NEW, exist_ok=True)

VALID_MONTHS = [11, 12]
MONTH_NAMES = {11: 'Nov', 12: 'Dec'}

# Spatial settings (must match Phase 2)
COMPARISON_RES = 4000   # 4 km
SMOOTH_SIGMA = 3
WEIGHT_THRESHOLD = 0.02
MIN_GPS_POINTS = 50

# ---- File paths (C:\RepastData) ----
RANGELAND_RASTER = r"C:\RepastData\rangelands_resampled_utm.tif"
SLOPE_RASTER = r"C:\RepastData\slopepercent_reclass_utm.tif"
NONSEDENTARY_SHP = r"C:\RepastData\nonsedentaryareas_utm.shp"

GPS_PATH = os.path.join(BASE_DIR, "gps_data", "gps_data.csv")
PHASE4_JSON = os.path.join(BASE_DIR, "calibration_outputs_final", "phase4_abc", "phase4_summary_final.json")

# Null model settings
N_NULL = 99
NULL_MODEL_SEED = 9999

# Agent count 
AGENT_COUNT = None



# UTILITIES


def get_gps_count(m):
    """Return GPS fix count for validation months."""
    return 835 if m == 11 else 779


def build_valid_mask():
    """
    Build a binary mask of cells that are suitable for agent placement.
    Conditions: Rangeland == 1, Slope == 1, and inside non‑sedentary polygons.
    Returns: list of (x, y) coordinates of valid cell centres.
    """
    print("Building landscape‑constrained valid mask...")
    
    # 1. Open rangeland raster (reference grid)
    with rasterio.open(RANGELAND_RASTER) as src:
        rangeland = src.read(1)
        ref_transform = src.transform
        ref_crs = src.crs
        ref_width = src.width
        ref_height = src.height
    
    # 2. Open slope raster and reproject to reference grid if needed
    with rasterio.open(SLOPE_RASTER) as src_slope:
        # Check if already aligned (same shape and transform)
        if (src_slope.width == ref_width and src_slope.height == ref_height and
                src_slope.transform == ref_transform and src_slope.crs == ref_crs):
            slope = src_slope.read(1)
        else:
            # Reproject slope to reference grid (nearest neighbour for categorical)
            slope = np.empty((ref_height, ref_width), dtype=np.float32)
            reproject(
                source=src_slope.read(1),
                destination=slope,
                src_transform=src_slope.transform,
                src_crs=src_slope.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest
            )
    
    # Combine rangeland and slope masks
    valid = (rangeland == 1) & (slope == 1)
    
    # 3. Rasterise non‑sedentary polygons onto the same grid
    gdf = gpd.read_file(NONSEDENTARY_SHP)
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)
    
    # Burn polygons into a mask
    shapes = ((geom, 1) for geom in gdf.geometry)
    burned = features.rasterize(
        shapes,
        out_shape=(ref_height, ref_width),
        transform=ref_transform,
        fill=0,
        dtype='uint8'
    )
    nonsed_mask = burned == 1
    
    # Final valid mask
    valid = valid & nonsed_mask
    
    # Extract coordinates of valid cell centres
    valid_coords = []
    for row in range(ref_height):
        for col in range(ref_width):
            if valid[row, col]:
                x = ref_transform[2] + col * ref_transform[0] + ref_transform[0] / 2
                y = ref_transform[5] + row * ref_transform[4] + ref_transform[4] / 2
                valid_coords.append((x, y))
    print(f"  Found {len(valid_coords)} valid cells")
    return valid_coords


def weighted_spearman(a_flat, gps_density, weights):
    """Weighted Spearman rank correlation (rescaled to [0,1])."""
    mask = weights > 0
    if mask.sum() < 10:
        return 0.5
    a = a_flat[mask]
    b = gps_density[mask]
    w = weights[mask]
    ra, rb = rankdata(a), rankdata(b)
    sw = w.sum()
    ma = np.sum(w * ra) / sw
    mb = np.sum(w * rb) / sw
    cov = np.sum(w * (ra - ma) * (rb - mb))
    sa = np.sqrt(np.sum(w * (ra - ma) ** 2))
    sb = np.sqrt(np.sum(w * (rb - mb) ** 2))
    if sa == 0 or sb == 0:
        return 0.5
    return (cov / (sa * sb) + 1.0) / 2.0


def build_abm_density(a_pts, x_bins, y_bins):
    """Gaussian‑smoothed sum‑to‑1 density surface on 4 km grid."""
    H, _, _ = np.histogram2d(a_pts[:, 0], a_pts[:, 1], bins=[x_bins, y_bins])
    H = gaussian_filter(H.astype(float), sigma=SMOOTH_SIGMA)
    if H.sum() > 0:
        H = H / H.sum()
    return H


def bhattacharyya(a_pts, gps_pts):
    """Bhattacharyya coefficient from KDE (higher = better)."""
    if len(a_pts) < 5 or len(gps_pts) < 5:
        return np.nan
    try:
        kde_abm = gaussian_kde(a_pts.T, bw_method='scott')
        kde_gps = gaussian_kde(gps_pts.T, bw_method='scott')
        x_lo = min(a_pts[:, 0].min(), gps_pts[:, 0].min())
        x_hi = max(a_pts[:, 0].max(), gps_pts[:, 0].max())
        y_lo = min(a_pts[:, 1].min(), gps_pts[:, 1].min())
        y_hi = max(a_pts[:, 1].max(), gps_pts[:, 1].max())
        xi = np.linspace(x_lo, x_hi, 50)
        yi = np.linspace(y_lo, y_hi, 50)
        xx, yy = np.meshgrid(xi, yi)
        grid = np.vstack([xx.ravel(), yy.ravel()])
        p_abm = kde_abm(grid)
        p_gps = kde_gps(grid)
        p_abm = p_abm / (p_abm.sum() + 1e-12)
        p_gps = p_gps / (p_gps.sum() + 1e-12)
        return np.sum(np.sqrt(p_abm * p_gps))
    except Exception:
        return np.nan



# MAIN


def main():
    global AGENT_COUNT
    print("=" * 72)
    print("Phase 5 – Landscape‑Constrained Null Model")
    print("=" * 72)

    # ---- Load agent count from Phase 4 ---------------------------------
    if not os.path.exists(PHASE4_JSON):
        raise FileNotFoundError(f"Phase 4 summary not found: {PHASE4_JSON}")
    with open(PHASE4_JSON) as f:
        phase4 = json.load(f)
    AGENT_COUNT = phase4.get('agent_count', 1154)
    print(f"  Agent count: {AGENT_COUNT}")

    # ---- Load GPS reference data ---------------------------------------
    gps_df = pd.read_csv(GPS_PATH)
    gps_df = gps_df[gps_df['month'].isin(VALID_MONTHS)].copy()
    if gps_df.empty:
        raise ValueError("No GPS data for months 11 and 12.")

    # Determine grid from GPS data (same as original Phase 5)
    xmin, xmax = gps_df['X'].min(), gps_df['X'].max()
    ymin, ymax = gps_df['Y'].min(), gps_df['Y'].max()
    x_bins = np.arange(xmin, xmax + COMPARISON_RES, COMPARISON_RES)
    y_bins = np.arange(ymin, ymax + COMPARISON_RES, COMPARISON_RES)

    gps_data = {}
    for m in VALID_MONTHS:
        g = gps_df[gps_df['month'] == m].copy()
        if len(g) < MIN_GPS_POINTS:
            continue
        pts = g[['X', 'Y']].values
        w = np.ones(len(pts))
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_bins, y_bins], weights=w)
        H = gaussian_filter(H.astype(float), sigma=SMOOTH_SIGMA)
        H = H / (H.sum() + 1e-12)
        W = H.copy()
        W[W < WEIGHT_THRESHOLD * H.max()] = 0.0
        gps_data[m] = {
            'density': H.flatten(),
            'weights': W.flatten(),
            'pts': pts,
            'x_bins': x_bins,
            'y_bins': y_bins
        }
        print(f"  GPS month {m}: {len(pts)} points")

    # ---- Build landscape‑constrained valid mask ------------------------
    valid_coords = build_valid_mask()
    if len(valid_coords) == 0:
        raise RuntimeError("No valid cells found. Check raster files.")
    print(f"  Valid cells: {len(valid_coords)}")

    # ---- Load existing ABM replicates from Phase 5 --------------------
    replicates = []
    for rep in range(1, 6):
        fname = os.path.join(OUT_DIR_ORIG, f"replicate_{rep}", "agent_locations.csv")
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            df = df[df['month'].isin(VALID_MONTHS)]
            replicates.append(df)
        else:
            print(f"  Warning: replicate {rep} not found.")
    if not replicates:
        raise FileNotFoundError("No replicate agent location files. Run original Phase 5 first.")

    # Aggregate agent points per month (across replicates)
    abm_pts_by_month = {m: [] for m in VALID_MONTHS}
    for df in replicates:
        for m in VALID_MONTHS:
            df_m = df[df['month'] == m]
            if len(df_m) > 0:
                abm_pts_by_month[m].append(df_m[['X', 'Y']].values)

    # ---- Compute model scores (average across replicates per month) ----
    model_scores = {m: {} for m in VALID_MONTHS}
    for m in VALID_MONTHS:
        if m not in gps_data:
            continue
        g = gps_data[m]
        all_pts = np.vstack(abm_pts_by_month[m]) if abm_pts_by_month[m] else np.empty((0, 2))
        if len(all_pts) == 0:
            continue
        H_flat = build_abm_density(all_pts, x_bins, y_bins).flatten()
        model_scores[m]['density_spearman'] = weighted_spearman(H_flat, g['density'], g['weights'])
        model_scores[m]['bhattacharyya'] = bhattacharyya(all_pts, g['pts'])

    # ---- Run landscape‑constrained null model -------------------------
    rng = np.random.default_rng(NULL_MODEL_SEED)
    n_valid = len(valid_coords)
    null_results = {m: {metric: [] for metric in ['density_spearman', 'bhattacharyya']}
                    for m in VALID_MONTHS}

    print(f"\n  Running {N_NULL} null iterations (landscape‑constrained)...")
    for i in range(N_NULL):
        idx = rng.choice(n_valid, size=AGENT_COUNT, replace=True)
        rand_pts = np.array([valid_coords[idx[j]] for j in range(AGENT_COUNT)])
        for m in VALID_MONTHS:
            if m not in gps_data:
                continue
            g = gps_data[m]
            H_null = build_abm_density(rand_pts, x_bins, y_bins).flatten()
            null_results[m]['density_spearman'].append(weighted_spearman(H_null, g['density'], g['weights']))
            null_results[m]['bhattacharyya'].append(bhattacharyya(rand_pts, g['pts']))
        if (i+1) % 20 == 0:
            print(f"    Iteration {i+1}/{N_NULL} done")

    # ---- Compute overall p‑values (combined across months) ------------
    overall_p = {}
    for metric in ['density_spearman', 'bhattacharyya']:
        all_null = []
        all_model = []
        for m in VALID_MONTHS:
            if m not in null_results or m not in model_scores:
                continue
            all_null.extend(null_results[m][metric])
            all_model.append(model_scores[m][metric])
        if len(all_null) == 0:
            overall_p[metric] = np.nan
            continue
        all_null = np.array(all_null)
        model_avg = np.mean(all_model)
        p = float(np.mean(all_null >= model_avg))   # both metrics: higher is better
        overall_p[metric] = p

    print("\n  Overall p‑values (landscape‑constrained null):")
    for k, v in overall_p.items():
        print(f"    {k}: {v:.4f}")

    # ---- Save results ------------------------------------------------
    df_p = pd.DataFrame([overall_p]).T
    df_p.columns = ['p_value_constrained']
    df_p['significant'] = df_p['p_value_constrained'] < 0.05
    df_p.to_csv(os.path.join(OUT_DIR_NEW, "null_pvalues_constrained.csv"))
    print(f"\n  Saved: null_pvalues_constrained.csv")

    # Load original p‑values (uniform bounding box null) if available
    orig_summary = os.path.join(OUT_DIR_ORIG, "phase5_summary_final.json")
    orig_p = {}
    if os.path.exists(orig_summary):
        with open(orig_summary) as f:
            orig = json.load(f)
            orig_p = orig.get('null_pvalues', {})
    # Comparison table
    comp_rows = []
    metric_map = {
        'density_spearman': 'p_null_density',
        'bhattacharyya': 'p_null_bhattacharyya'
    }
    for metric, key in metric_map.items():
        comp_rows.append({
            'Metric': metric,
            'p_value_original_uniform': orig_p.get(key, np.nan),
            'p_value_constrained': overall_p.get(metric, np.nan),
            'significant_constrained': overall_p.get(metric, 1.0) < 0.05
        })
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(os.path.join(OUT_DIR_NEW, "null_pvalues_comparison.csv"), index=False)
    print("  Saved: null_pvalues_comparison.csv")

    # ---- Update Phase 5 summary JSON with new null p‑values ----------
    if os.path.exists(orig_summary):
        with open(orig_summary) as f:
            summary = json.load(f)
        new_null = {}
        for metric, key in metric_map.items():
            new_null[key] = overall_p.get(metric, np.nan)
        summary['null_pvalues_constrained'] = new_null
        summary['null_model_type'] = 'landscape_constrained (rangeland+slope+non-sedentary)'
        summary['n_null'] = N_NULL
        if new_null.get('p_null_density', 1.0) < 0.05:
            summary['overall_verdict_constrained'] = 'PASS'
        else:
            summary['overall_verdict_constrained'] = 'WARN'
        new_summary_path = os.path.join(OUT_DIR_NEW, "phase5_summary_constrained_null.json")
        with open(new_summary_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        print(f"  Saved updated summary: {new_summary_path}")

    # ---- Generate plots per month ------------------------------------
    for m in VALID_MONTHS:
        if m not in null_results or m not in model_scores:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        metrics_plot = [
            ('density_spearman', 'Weighted Spearman', 'higher'),
            ('bhattacharyya', 'Bhattacharyya', 'higher')
        ]
        for ax, (metric, label, direction) in zip(axes, metrics_plot):
            null_vals = np.array(null_results[m][metric])
            null_vals = null_vals[~np.isnan(null_vals)]
            model_val = model_scores[m][metric]
            ax.hist(null_vals, bins=20, alpha=0.7, color='#AAAAAA', edgecolor='white')
            ax.axvline(model_val, color='#C0392B', lw=2, label=f'ABM = {model_val:.3f}')
            if direction == 'higher':
                p = float(np.mean(null_vals >= model_val))
            else:
                p = float(np.mean(null_vals <= model_val))
            ax.set_title(f"{label}\np = {p:.3f}" + (" ✓" if p<0.05 else " ✗"), fontsize=9)
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            ax.legend(fontsize=7)
        plt.suptitle(f"Null model (landscape‑constrained) – {MONTH_NAMES[m]}")
        plt.tight_layout()
        out_plot = os.path.join(OUT_DIR_NEW, f"null_comparison_month_{m}_constrained.png")
        plt.savefig(out_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {out_plot}")

    print("\n" + "=" * 72)
    print(f"All outputs saved in: {OUT_DIR_NEW}")
    print("=" * 72)


if __name__ == "__main__":
    main()