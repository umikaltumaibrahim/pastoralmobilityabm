"""
agent_allocation.py
====================
Shared utility module imported by all pipeline phases

Reads the `districts_pop` vector file (Shapefile or GeoPackage) and returns:
    - Per-district agent counts from the `agent_numb` attribute field.
    - The total agent count (sum of all `agent_numb` values).
    - A structured DataFrame of district records for reporting.

Expected vector file
--------------------
File     : C:\RepastData\districts_pop.*
           (.shp, .gpkg, or any format readable by geopandas/fiona)
Field    : agent_numb  — integer, >= 0 per district polygon
         
Returned values
---------------
load_agent_allocation(base_dir)
    Returns: (total_agents: int, district_df: pd.DataFrame)
    district_df columns: district_id, district_name (if available),
                         agent_numb, geometry_area_km2 (if available)

get_total_agents(base_dir)
    Convenience wrapper — returns only the int total.
"""

import os
import pandas as pd

# ---- Vector file location (Hardcoded to match Java's C:\RepastData path) ----
# Accepted formats: .shp, .gpkg, .geojson, .json
VECTOR_CANDIDATES = [
    r"C:\RepastData\districts_pop.gpkg",
    r"C:\RepastData\districts_pop.shp",
    r"C:\RepastData\districts_pop.geojson",
    r"C:\RepastData\districts_pop.json",
]

# Name of the per-district agent count field
AGENT_NUMB_FIELD = "agent_numb"
# Fallback in case it's named 'agent_num'
AGENT_NUMB_FALLBACK = "agent_num"

# Optional district name field (used in reporting if present)
DISTRICT_NAME_CANDIDATES = ["district_name", "name", "Name", "DISTRICT",
                             "ADM2_EN", "ADM1_EN"]


# ============================================================
# PRIVATE: RESOLVE VECTOR PATH
# ============================================================

def _resolve_vector_path(base_dir):
    """
    Search VECTOR_CANDIDATES (hardcoded to C:\RepastData).
    Returns the first existing path, or raises FileNotFoundError.
    """
    tried = []
    for full in VECTOR_CANDIDATES:
        tried.append(full)
        if os.path.exists(full):
            return full

    raise FileNotFoundError(
        f"districts_pop vector file not found. Searched:\n"
        + "\n".join(f"  {p}" for p in tried)
        + "\n\nPlace the file at one of the above paths and ensure "
          f"it contains the '{AGENT_NUMB_FIELD}' attribute field."
    )


# ============================================================
# PRIVATE: VALIDATE agent_numb FIELD
# ============================================================

def _validate_agent_numb(gdf, path):
    """
    Validate the agent_numb column in the loaded GeoDataFrame.
    Raises ValueError with a clear message on any violation.
    """
    # Check for the primary field or fallback field
    actual_field = AGENT_NUMB_FIELD
    if AGENT_NUMB_FIELD not in gdf.columns:
        if AGENT_NUMB_FALLBACK in gdf.columns:
            actual_field = AGENT_NUMB_FALLBACK
        else:
            available = [c for c in gdf.columns if c != 'geometry']
            raise ValueError(
                f"Field '{AGENT_NUMB_FIELD}' not found in {path}.\n"
                f"Available fields: {available}\n"
                f"Rename the agent count field to '{AGENT_NUMB_FIELD}'."
            )

    col = gdf[actual_field]

    # Check for nulls
    n_null = col.isna().sum()
    if n_null > 0:
        null_idx = gdf.index[col.isna()].tolist()
        raise ValueError(
            f"'{actual_field}' has {n_null} null value(s) "
            f"at row index(es): {null_idx}. "
            f"All districts must have a valid integer agent count."
        )

    # Coerce to int and check for non-integer values
    try:
        counts = col.astype(int)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"'{actual_field}' could not be coerced to integer: {e}. "
            f"All values must be whole numbers."
        ) from e

    # Check for negatives (0 is now permitted and will be skipped downstream)
    n_bad = int((counts < 0).sum())
    if n_bad > 0:
        bad_idx = gdf.index[counts < 0].tolist()
        raise ValueError(
            f"'{actual_field}' has {n_bad} negative value(s) "
            f"at row index(es): {bad_idx}. "
            f"All district agent counts must be >= 0."
        )

    return counts, actual_field


# ============================================================
# PUBLIC: LOAD AGENT ALLOCATION
# ============================================================

def load_agent_allocation(base_dir):
    """
    Load per-district agent allocation from the districts_pop vector file.
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "geopandas is required to read the districts_pop vector file.\n"
            "Install with:  pip install geopandas\n"
            f"Original error: {e}"
        ) from e

    # ---- Locate the file -----------------------------------------
    vector_path = _resolve_vector_path(base_dir)
    print(f"  [agent_allocation] Reading: {vector_path}")

    # ---- Load with geopandas ------------------------------------
    gdf = gpd.read_file(vector_path)
    print(f"  [agent_allocation] Loaded {len(gdf)} district polygons")

    if len(gdf) == 0:
        raise ValueError(
            f"The vector file {vector_path} contains no features. "
            f"Ensure it has at least one district polygon with '{AGENT_NUMB_FIELD}'."
        )

    # ---- Validate agent_numb ------------------------------------
    counts, actual_field = _validate_agent_numb(gdf, vector_path)
    total_agents = int(counts.sum())

    # ---- Resolve optional district name field -------------------
    name_field = None
    for candidate in DISTRICT_NAME_CANDIDATES:
        if candidate in gdf.columns:
            name_field = candidate
            break

    # ---- Compute area if CRS is projected (metric units) --------
    area_km2 = None
    try:
        if gdf.crs is not None and gdf.crs.is_projected:
            area_km2 = gdf.geometry.area / 1e6   # m² → km²
    except Exception:
        pass

    # ---- Build output DataFrame ---------------------------------
    rows = []
    for i, row in gdf.iterrows():
        n = int(row[actual_field])
        
        # Match Java's behavior: safely skip any districts with 0 agents
        if n <= 0:
            continue
            
        name = (str(row[name_field])
                if name_field and not pd.isna(row[name_field])
                else f"District_{i}")
        area = (float(area_km2.loc[i])
                if area_km2 is not None else float('nan'))
        rows.append({
            'district_id':   i,
            'district_name': name,
            'agent_numb':    n,
            'area_km2':      area,
            'pct_total':     round(100.0 * n / total_agents, 2)
                             if total_agents > 0 else 0.0,
        })

    district_df = pd.DataFrame(rows)

    # ---- Summary report -----------------------------------------
    print(f"  [agent_allocation] Total agents      : {total_agents:,}")
    print(f"  [agent_allocation] Active Districts  : {len(district_df)} (Skipped {(len(gdf) - len(district_df))} empty)")
    print(f"  [agent_allocation] Min per district  : "
          f"{district_df['agent_numb'].min():,}")
    print(f"  [agent_allocation] Max per district  : "
          f"{district_df['agent_numb'].max():,}")
    print(f"  [agent_allocation] Mean per district : "
          f"{district_df['agent_numb'].mean():.1f}")

    return total_agents, district_df


# ============================================================
# PUBLIC: GET TOTAL AGENTS (convenience wrapper)
# ============================================================

def get_total_agents(base_dir):
    """
    Convenience function — returns only the integer total agent count.
    Use this wherever a scalar AGENT_COUNT value is needed.
    """
    total, _ = load_agent_allocation(base_dir)
    return total


# ============================================================
# STANDALONE TEST (run this file directly to validate the setup)
# ============================================================

if __name__ == "__main__":
    import sys

    base = os.path.abspath(
        sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    )
    print(f"\nTesting agent_allocation.py")
    print(f"  BASE_DIR: {base}")
    print()

    try:
        total, df = load_agent_allocation(base)
        print()
        print(f"  District allocation table:")
        print(df.to_string(index=False))
        print()
        print(f"  TOTAL AGENTS (replaces hardcoded AGENT_COUNT): {total:,}")
        print()
        print("  ✓  agent_allocation.py test PASSED")
    except Exception as e:
        print(f"\n  ✗  agent_allocation.py test FAILED: {e}")
        sys.exit(1)