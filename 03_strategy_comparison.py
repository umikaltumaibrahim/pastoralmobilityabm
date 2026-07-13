"""
CROSS-STRATEGY COMPARISON & MONTE CARLO 
===============================================================
Reads best_results.json from calibration of all four strategies.
Computes fitness ranking and Monte Carlo robustness analysis.

Fitness function used is identical to calibration phase
  
Monte Carlo robustness:
    10,000 perturbation iterations with Gaussian noise sigma = NOISE_SCALE.
    NOISE_SCALE is set to ~half the smallest inter-strategy fitness gap.
  
Outputs:
    fitness_ranking_final.csv
    monte_carlo_results_final.csv
    phase3_summary_final.json  — read by abc phase
    fitness_ranking_final.png
    mc_rank_probabilities_final.png
    metric_radar_final.png
"""

import numpy as np
import pandas as pd
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# CONFIGURATION

BASE_DIR = os.path.abspath(os.getcwd())
OUT_DIR  = os.path.join(BASE_DIR, "calibration_outputs_final")

STRATEGIES = [
    "RainfallPriority_30km",
    "PasturePriority_30km",
    "RainfallPriority_60km",
    "PasturePriority_60km",
]

# Monte Carlo settings
MC_ITER     = 10_000
# NOISE_SCALE: set to ~half the smallest observed inter-strategy gap.
# Probes whether the ranking is stable at the scale of the closest
# competition. 
NOISE_SCALE = 0.005  # Gaussian noise std-dev added to fitness scores per MC iteration

# Fitness weights match calibration 
W_DENSITY  = 0.60
W_SAVI     = 0.20
W_RAINFALL = 0.20

# Metric labels and weights for visualizations.
METRIC_LABELS = {
    'density_correlation': 'Density correlation (Spearman 4 km)',
    'savi_tracking':       'SAVI tracking (KS)',
    'rainfall_tracking':   'Rainfall tracking (KS)',
}
METRIC_WEIGHTS = {
    'density_correlation': W_DENSITY,
    'savi_tracking':       W_SAVI,
    'rainfall_tracking':   W_RAINFALL,
}

# Monte Carlo rank-1 probability classification thresholds
RANK1_ROBUST     = 0.80
RANK1_STABLE     = 0.50
RANK1_COMPETITIVE = 0.20



# LOAD PHASE 2 RESULTS

def load_results():
    """
    Load best_results.json for each strategy from Phase 2 output.
    Returns list of dicts with keys: strategy, fitness, params, metrics, version.

    Backwards compatibility: if 'resource_tracking' is found instead of
    'savi_tracking' (older pipeline versions), it is remapped with a warning.
    """
    data = []
    for strat in STRATEGIES:
        path = os.path.join(OUT_DIR, strat, "best_results.json")
        if not os.path.exists(path):
            print(f"  WARNING: Missing results for {strat} — skipping.")
            continue

        with open(path) as f:
            d = json.load(f)

        metrics = d.get('best_metrics', {})

        # Backwards compatibility remap
        if 'resource_tracking' in metrics and 'savi_tracking' not in metrics:
            print(f"  NOTE ({strat}): remapping 'resource_tracking' -> 'savi_tracking'.")
            metrics['savi_tracking'] = metrics.pop('resource_tracking')

        version = d.get('version', 'unknown')
        if version != 'final_spearman4km_no_trajectory':
            print(f"  NOTE ({strat}): version = '{version}'. "
                  f"Expected 'final_spearman4km_no_trajectory'.")

        data.append({
            'strategy': strat,
            'fitness':  d['best_fitness'],
            'params':   d.get('best_params', {}),
            'metrics':  metrics,
            'version':  version,
        })
    return data



# DETERMINISTIC RANKING


def build_ranking_table(strategy_data):
    """
    Sort strategies by composite fitness descending.
    Returns DataFrame with fitness and all three POM metric scores.
    """
    rows = []
    for d in strategy_data:
        row = {'strategy': d['strategy'], 'fitness': d['fitness']}
        for key in METRIC_LABELS:
            row[key] = d['metrics'].get(key, float('nan'))
        rows.append(row)

    df = (pd.DataFrame(rows)
            .sort_values(by='fitness', ascending=False)
            .reset_index(drop=True))
    df.insert(0, 'Rank', range(1, len(df) + 1))
    return df



# MONTE CARLO ROBUSTNESS ANALYSIS


def run_monte_carlo(strategy_data):
    """
    Adds Gaussian noise to fitness scores across MC_ITER iterations.
    Records rank distribution for each strategy.

    Returns DataFrame with P(Rank k) for each strategy, sorted by P(Rank 1).
    """
    strategies = [d['strategy'] for d in strategy_data]
    fitness    = np.array([d['fitness'] for d in strategy_data])
    n          = len(strategies)

    rank_counts = np.zeros((n, n), dtype=np.int64)
    rng         = np.random.default_rng(seed=42)   # fixed seed for reproducibility

    for _ in range(MC_ITER):
        noise     = rng.normal(0.0, NOISE_SCALE, size=n)
        perturbed = fitness + noise
        ranks     = np.argsort(-perturbed)   # descending — rank 0 = best
        for rank_pos, strat_idx in enumerate(ranks):
            rank_counts[strat_idx, rank_pos] += 1

    results = []
    for i, strat in enumerate(strategies):
        probs     = rank_counts[i] / MC_ITER
        mean_rank = float(np.sum(probs * np.arange(1, n + 1)))
        rank_std  = float(np.sqrt(
            np.sum(probs * (np.arange(1, n + 1) - mean_rank) ** 2)
        ))

        if probs[0] > RANK1_ROBUST:
            interpretation = "Robust winner"
        elif probs[0] > RANK1_STABLE:
            interpretation = "Moderately stable"
        elif probs[0] > RANK1_COMPETITIVE:
            interpretation = "Competitive"
        else:
            interpretation = "Weak"

        row = {
            'Strategy':       strat,
            'Fitness':        round(float(fitness[i]), 6),
            'Mean_Rank':      round(mean_rank, 3),
            'Rank_Std':       round(rank_std, 3),
            'Interpretation': interpretation,
        }
        for r in range(n):
            row[f'Prob_Rank_{r + 1}'] = round(float(probs[r]), 4)
        results.append(row)

    return (pd.DataFrame(results)
              .sort_values(by='Prob_Rank_1', ascending=False)
              .reset_index(drop=True))



# VISUALIZATIONS


def plot_fitness_ranking(df_rank):
    """
    Two-panel: composite fitness horizontal bar chart (left) and
    stacked weighted metric decomposition (right).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c'][:len(df_rank)]

    # Left: composite fitness
    ax   = axes[0]
    bars = ax.barh(df_rank['strategy'], df_rank['fitness'],
                   color=colours, edgecolor='white')
    ax.set_xlabel("POM composite fitness (3 metrics)")
    ax.set_title("Deterministic fitness ranking\n"
                 "(Spearman 4 km | no trajectory)")
    ax.set_xlim(0, min(1.0, df_rank['fitness'].max() * 1.25))
    for bar, val in zip(bars, df_rank['fitness']):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=9)

    # Right: metric decomposition
    ax2       = axes[1]
    bottom    = np.zeros(len(df_rank))
    m_colours = ['#1a9641', '#fdae61', '#d7191c']

    for (metric, weight), colour in zip(METRIC_WEIGHTS.items(), m_colours):
        if metric not in df_rank.columns:
            continue
        vals         = weight * df_rank[metric].fillna(0).values
        ax2.barh(df_rank['strategy'], vals, left=bottom,
                 color=colour, edgecolor='white',
                 label=f"{METRIC_LABELS[metric]} (w={weight:.2f})")
        bottom += vals

    ax2.set_xlabel("Weighted metric contribution")
    ax2.set_title("POM fitness decomposition (3 metrics)")
    ax2.set_xlim(0, min(1.0, bottom.max() * 1.25))
    ax2.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fitness_ranking_final.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_monte_carlo(df_mc):
    """
    Two-panel: P(Rank 1) horizontal bar chart (left) and
    full rank probability distribution stacked bar (right).
    """
    n         = len(df_mc)
    rank_cols = [c for c in df_mc.columns if c.startswith('Prob_Rank_')]

    # Convert percentage strings if present; handle numeric directly
    def parse_prob(x):
        if isinstance(x, str):
            return float(x.replace('%', '')) / 100
        return float(x)

    # Fix for pandas 2.1+ where applymap is removed. Use apply + map.
    prob_matrix = df_mc[rank_cols].apply(lambda col: col.map(parse_prob)).values
    p_rank1     = prob_matrix[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P(Rank 1)
    ax      = axes[0]
    colours = ['#1a9641' if p > RANK1_ROBUST  else
               '#a6d96a' if p > RANK1_STABLE  else
               '#fdae61' if p > RANK1_COMPETITIVE else
               '#d7191c' for p in p_rank1]
    bars = ax.barh(df_mc['Strategy'], p_rank1,
                   color=colours, edgecolor='white')
    ax.axvline(RANK1_STABLE, linestyle='--', color='grey',
               linewidth=0.8, label=f'P = {RANK1_STABLE}')
    ax.axvline(RANK1_ROBUST, linestyle='--', color='black',
               linewidth=0.8, label=f'P = {RANK1_ROBUST}')
    ax.set_xlabel("Probability of ranking 1st")
    ax.set_title("Monte Carlo: P(Rank 1)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, p_rank1):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va='center', fontsize=9)

    # Right: full rank distribution
    ax2          = axes[1]
    rank_colours = ['#1a9641', '#a6d96a', '#fdae61', '#d7191c'][:n]
    left         = np.zeros(n)
    for r in range(n):
        ax2.barh(df_mc['Strategy'], prob_matrix[:, r], left=left,
                 color=rank_colours[r], edgecolor='white',
                 label=f"Rank {r + 1}")
        left += prob_matrix[:, r]
    ax2.set_xlabel("Probability")
    ax2.set_title("Monte Carlo: full rank distribution")
    ax2.set_xlim(0, 1)
    ax2.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "mc_rank_probabilities_final.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_metric_radar(df_rank):
    """
    Radar (spider) chart showing the three POM metric scores per strategy.
    Trajectory metric is absent — do not add to this plot.
    """
    metrics  = list(METRIC_LABELS.keys())
    labels   = [METRIC_LABELS[m] for m in metrics]
    n_m      = len(metrics)
    angles   = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
    angles  += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colours = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c']

    for i, (_, row) in enumerate(df_rank.iterrows()):
        values  = [
            float(row[m]) if m in row and not pd.isna(row[m]) else 0.0
            for m in metrics
        ]
        values += values[:1]
        ax.plot(angles, values, color=colours[i % len(colours)],
                linewidth=2, label=row['strategy'])
        ax.fill(angles, values, color=colours[i % len(colours)], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("POM metric profile by strategy\n(3 metrics | 4 km grid)",
                 pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "metric_radar_final.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")



# MAIN


def main():
    print("=" * 70)
    print("PHASE 3 — CROSS-STRATEGY COMPARISON (vF_revised)")
    print("=" * 70)
    print(f"  Fitness weights  : density={W_DENSITY} | SAVI={W_SAVI} | rainfall={W_RAINFALL}")
    print(f"  Trajectory metric: REMOVED")
    print(f"  Grid resolution  : 4 km")
    print(f"  MC iterations    : {MC_ITER:,}")
    print(f"  MC noise scale   : {NOISE_SCALE}")
    print("=" * 70)

    # ---- Load Phase 2 results ---------------------------------
    strategy_data = load_results()
    if len(strategy_data) < 2:
        print("\n  ERROR: Fewer than 2 strategies loaded. "
              "Run Phase 2 (vF_revised) for all strategies first.")
        return

    print(f"\n  Loaded {len(strategy_data)} strategies:")
    for d in strategy_data:
        print(f"    {d['strategy']:<30} fitness = {d['fitness']:.4f}")

    # ---- Deterministic ranking --------------------------------
    print("\n" + "-" * 70)
    print("  DETERMINISTIC FITNESS RANKING")
    print("-" * 70)
    df_rank     = build_ranking_table(strategy_data)
    display_cols = ['Rank', 'strategy', 'fitness'] + list(METRIC_LABELS.keys())
    print(df_rank[[c for c in display_cols if c in df_rank.columns]]
          .to_string(index=False))
    df_rank.to_csv(os.path.join(OUT_DIR, "fitness_ranking_final.csv"), index=False)
    print(f"\n  Saved: fitness_ranking_final.csv")

    # ---- Monte Carlo robustness -------------------------------
    print("\n" + "-" * 70)
    print("  MONTE CARLO ROBUSTNESS")
    print("-" * 70)
    df_mc = run_monte_carlo(strategy_data)
    mc_display = (
        ['Strategy', 'Fitness', 'Mean_Rank', 'Rank_Std', 'Interpretation']
        + [f'Prob_Rank_{r + 1}' for r in range(len(strategy_data))]
    )
    print(df_mc[[c for c in mc_display if c in df_mc.columns]]
          .to_string(index=False))
    df_mc.to_csv(os.path.join(OUT_DIR, "monte_carlo_results_final.csv"),
                 index=False)
    print(f"\n  Saved: monte_carlo_results_final.csv")

    # ---- Visualisations ---------------------------------------
    print("\n" + "-" * 70)
    print("  VISUALISATIONS")
    print("-" * 70)
    plot_fitness_ranking(df_rank)
    plot_monte_carlo(df_mc)
    plot_metric_radar(df_rank)

    # ---- Winner summary ---------------------------------------
    winner     = df_mc.iloc[0]['Strategy']
    confidence = float(df_mc.iloc[0]['Prob_Rank_1'])
    mean_rank  = float(df_mc.iloc[0]['Mean_Rank'])
    interp     = df_mc.iloc[0]['Interpretation']

    winner_row     = df_rank[df_rank['strategy'] == winner].iloc[0]
    winner_fitness = float(winner_row['fitness'])
    winner_metrics = {
        k: (float(winner_row[k]) if k in winner_row and not pd.isna(winner_row[k]) else None)
        for k in METRIC_LABELS
    }

    print(f"\n  WINNER : {winner}")
    print(f"  Fitness: {winner_fitness:.4f}")
    print(f"  P(Rank 1): {confidence:.3f}  ({interp})")
    print(f"  Mean MC rank: {mean_rank:.3f}")

    # ---- Noise scale diagnostic -------------------------------
    fitness_vals = np.array([d['fitness'] for d in strategy_data])
    sorted_vals  = np.sort(fitness_vals)[::-1]
    gaps         = np.diff(sorted_vals) * -1   # positive gap values
    min_gap      = float(np.min(gaps)) if len(gaps) > 0 else 0.0
    print(f"\n  Fitness gap diagnostic:")
    print(f"    Smallest inter-strategy gap : {min_gap:.4f}")
    print(f"    Current NOISE_SCALE         : {NOISE_SCALE:.4f}")
    if NOISE_SCALE > min_gap:
        print(f"    WARN: NOISE_SCALE > gap. "
              f"Recommended: ~{min_gap / 2:.4f}")
    elif NOISE_SCALE < min_gap / 5:
        print(f"    WARN: NOISE_SCALE << gap. "
              f"Recommended: ~{min_gap / 2:.4f}")
    else:
        print(f"    OK: NOISE_SCALE calibrated appropriately.")

    # ---- Save phase3_summary_final.json -----------------------
    summary = {
        'version':          'final_spearman4km_no_trajectory',
        'winner':           winner,
        'confidence':       confidence,
        'mean_rank':        mean_rank,
        'interpretation':   interp,
        'winner_fitness':   winner_fitness,
        'winner_metrics':   winner_metrics,
        'noise_scale_used': NOISE_SCALE,
        'fitness_weights':  METRIC_WEIGHTS,
        'results':          df_mc.to_dict('records'),
        'fitness_ranking':  df_rank[['Rank', 'strategy', 'fitness']].to_dict('records'),
    }

    summary_path = os.path.join(OUT_DIR, "phase3_summary_final.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\n  Phase 3 summary saved -> {summary_path}")
    print("  (Phase 4 reads phase3_summary_final.json)")
    print("\n  Phase 3 (vF_revised) complete.")


if __name__ == "__main__":
    main()