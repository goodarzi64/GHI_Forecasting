# src/data_pipeline.py
"""
Data + preprocessing pipeline for CYL GHI project.

Design goals:
- No Colab-specific code here (no drive.mount, no !pip)
- Paths are passed in from notebooks/scripts
- Preprocessed artifacts saved/loaded as compressed .npz files
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# -----------------------------
# Static geo features
# -----------------------------
def build_static_geo_features(
    stations_csv_path: str,
    geo_feature_cols=("height", "Slope_DEM2_U1", "Aspect_DEM2_1", "rastercalc"),
) -> Tuple[pd.DataFrame, List[str], MinMaxScaler]:
    """
    Reads stations.csv and builds normalized static terrain features.
    Converts aspect(deg) -> sin/cos and drops raw aspect.
    Returns:
      vars_geo: (N, G) dataframe (scaled 0..1)
      station_files: list of station_code
      geo_scaler: fitted MinMaxScaler
    """
    df_geo = pd.read_csv(stations_csv_path)
    station_files = df_geo["station_code"].tolist()

    vars_geo = df_geo[list(geo_feature_cols)].rename(
        columns={
            "Slope_DEM2_U1": "slope",
            "Aspect_DEM2_1": "aspect",
            "rastercalc": "twi",
        }
    )

    # aspect -> sin/cos, drop raw aspect
    aspect_rad = np.deg2rad(vars_geo["aspect"].to_numpy())
    vars_geo["aspect_cos"] = np.cos(aspect_rad)
    vars_geo["aspect_sin"] = np.sin(aspect_rad)
    vars_geo = vars_geo.drop(columns=["aspect"])

    # normalize
    geo_scaler = MinMaxScaler()
    vars_geo_scaled = geo_scaler.fit_transform(vars_geo.to_numpy())
    vars_geo = pd.DataFrame(vars_geo_scaled, columns=vars_geo.columns)

    return vars_geo, station_files, geo_scaler


# -----------------------------
# Preprocessed artifacts I/O
# -----------------------------
@dataclass(frozen=True)
class PreprocessArtifacts:
    temporal_node_tensor: np.ndarray  # (T, N, F)
    temporal_target_tensor: np.ndarray  # (T, N)
    df_cols: np.ndarray  # (F,)
    masks: Dict[str, np.ndarray]  # boolean masks keyed by name


def _artifact_paths(data_dir: str, suffix: str = "2") -> Dict[str, str]:
    return {
        "node_tensor": os.path.join(data_dir, f"node_tensor{suffix}.npz"),
        "target_tensor": os.path.join(data_dir, f"target_tensor{suffix}.npz"),
        "columns": os.path.join(data_dir, f"columns{suffix}.npz"),
        "masks": os.path.join(data_dir, f"masks{suffix}.npz"),
    }


def load_preprocessed(data_dir: str, suffix: str = "2") -> PreprocessArtifacts:
    """
    Loads node/target/columns/masks from compressed npz files.
    Expects files created by save_preprocessed().
    """
    paths = _artifact_paths(data_dir, suffix=suffix)

    node_npz = np.load(paths["node_tensor"], allow_pickle=True)
    tgt_npz = np.load(paths["target_tensor"], allow_pickle=True)
    col_npz = np.load(paths["columns"], allow_pickle=True)
    masks_npz = np.load(paths["masks"], allow_pickle=True)

    temporal_node_tensor = node_npz["data"]
    temporal_target_tensor = tgt_npz["data"]
    df_cols = col_npz["data"]

    masks = {k: masks_npz[k] for k in masks_npz.files}
    return PreprocessArtifacts(temporal_node_tensor, temporal_target_tensor, df_cols, masks)


def save_preprocessed(
    data_dir: str,
    temporal_node_tensor: np.ndarray,
    temporal_target_tensor: np.ndarray,
    df_cols: np.ndarray,
    masks: Dict[str, np.ndarray],
    suffix: str = "2",
) -> None:
    """Saves compressed npz artifacts (node/target/columns/masks)."""
    os.makedirs(data_dir, exist_ok=True)
    paths = _artifact_paths(data_dir, suffix=suffix)

    np.savez_compressed(paths["node_tensor"], data=temporal_node_tensor)
    np.savez_compressed(paths["target_tensor"], data=temporal_target_tensor)
    np.savez_compressed(paths["columns"], data=df_cols)
    np.savez_compressed(paths["masks"], **masks)


# -----------------------------
# Build from scratch
# -----------------------------
def build_from_scratch(
    station_files: List[str],
    data_dir: str,
    prep_suffix: str = "_prep.csv",
    timestamp_col: int = 0,
    target_col: str = "GHI",
    exclude_from_norm: Optional[List[str]] = None,
) -> PreprocessArtifacts:
    """
    Builds temporal tensors and masks from per-station *_prep.csv files.

    Returns PreprocessArtifacts:
      temporal_node_tensor: (T, N, F)
      temporal_target_tensor: (T, N)
      df_cols: feature names (F,)
      masks: dict of boolean masks

    Notes:
    - Expects each station file to be in `data_dir/{station_code}{prep_suffix}`
    - Applies time encodings + trig features for wind_dir and sun_azim
    - Normalizes selected features only (MinMaxScaler across all time*nodes)
    - Appends meteo_cat categorical label as last feature
    """
    if exclude_from_norm is None:
        exclude_from_norm = ["NDVI", "wind_dir"]  # plus cloud_* one-hot handled below

    df_list, target_list = [], []
    last_df_cols = None

    for station in station_files:
        path = os.path.join(data_dir, f"{station}{prep_suffix}")
        df = pd.read_csv(path)

        ts = pd.to_datetime(df.iloc[:, timestamp_col])
        doy = ts.dt.dayofyear.to_numpy()
        tod = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy()

        # add cyclic time + trig transforms
        df = df.assign(
            doy_sin=np.sin(2 * np.pi * doy / 365.0),
            doy_cos=np.cos(2 * np.pi * doy / 365.0),
            tod_sin=np.sin(2 * np.pi * tod / 86400.0),
            tod_cos=np.cos(2 * np.pi * tod / 86400.0),
            wind_dir_sin=lambda x: np.sin(np.radians(x["wind_dir"])),
            wind_dir_cos=lambda x: np.cos(np.radians(x["wind_dir"])),
            sun_azim_sin=lambda x: np.sin(np.radians(x["sun_azim"])),
            sun_azim_cos=lambda x: np.cos(np.radians(x["sun_azim"])),
        )

        # drop raw sun_azim, keep wind_dir at end
        if "sun_azim" in df.columns:
            df = df.drop(columns=["sun_azim"])
        if "wind_dir" in df.columns:
            cols = [c for c in df.columns if c != "wind_dir"] + ["wind_dir"]
            df = df[cols]

        # features / target
        features = df.iloc[:, 1:]  # drop timestamp
        target = df[target_col].to_numpy()

        df_list.append(features)
        target_list.append(target)
        last_df_cols = features.columns

    if last_df_cols is None:
        raise ValueError("No station data loaded. Check station_files and data_dir.")

    df_cols = np.array(last_df_cols, dtype=object)
    temporal_node_tensor = np.stack([d.to_numpy() for d in df_list], axis=1)  # (T, N, F)
    temporal_target_tensor = np.stack(target_list, axis=1)  # (T, N)

    # -----------------------------
    # Normalize selected features
    # -----------------------------
    cloud_cols = [c for c in df_cols if str(c).startswith("cloud_")]
    non_norm_features = set(exclude_from_norm) | set(cloud_cols) | {"meteo_cat"}
    norm_mask = np.array([c not in non_norm_features for c in df_cols], dtype=bool)
    notnorm_mask = ~norm_mask

    norm_node = temporal_node_tensor[:, :, norm_mask]
    notnorm_node = temporal_node_tensor[:, :, notnorm_mask]

    T, N, F_norm = norm_node.shape
    scaler = MinMaxScaler()
    norm_2d = norm_node.reshape(-1, F_norm)
    norm_2d_scaled = scaler.fit_transform(norm_2d)
    norm_node_scaled = norm_2d_scaled.reshape(T, N, F_norm)

    temporal_node_tensor_scaled = np.zeros_like(temporal_node_tensor, dtype=np.float32)
    temporal_node_tensor_scaled[:, :, norm_mask] = norm_node_scaled
    temporal_node_tensor_scaled[:, :, notnorm_mask] = notnorm_node.astype(np.float32)
    temporal_node_tensor = temporal_node_tensor_scaled

    # -----------------------------
    # Meteorological group label (meteo_cat)
    # -----------------------------
    col_idx = {str(col): i for i, col in enumerate(df_cols)}
    cloud_idxs = [col_idx[c] for c in df_cols if str(c).startswith("cloud_")]
    cloud_pc_idx = col_idx.get("cloud_Probably_Clear", None)

    precip_idx = col_idx.get("precipitation", None)
    aod_idx = col_idx.get("AOD", None)
    sun_elev_idx = col_idx.get("sun_elev", None)

    if precip_idx is None or aod_idx is None or sun_elev_idx is None:
        raise KeyError("Expected columns missing: precipitation, AOD, sun_elev")

    group_labels = np.zeros((T, N), dtype=np.int8)
    for t in range(T):
        x_t = temporal_node_tensor[t]  # (N, F)
        for n in range(N):
            x = x_t[n]
            is_rainy = x[precip_idx] > 0.1
            is_dusty = x[aod_idx] > 0.2
            is_low_sun = x[sun_elev_idx] < 10
            is_cloudy = any(x[i] > 0 for i in cloud_idxs if i != cloud_pc_idx)
            is_clear = (all(x[i] == 0 for i in cloud_idxs) or
                        (cloud_pc_idx is not None and x[cloud_pc_idx] > 0.5))

            if is_rainy:
                group = 2
            elif is_dusty:
                group = 3
            elif is_low_sun:
                group = 4
            elif is_cloudy:
                group = 1
            elif is_clear:
                group = 0
            else:
                group = 1

            group_labels[t, n] = group

    temporal_node_tensor = np.concatenate(
        [temporal_node_tensor, group_labels[..., None].astype(np.float32)],
        axis=-1,
    )
    df_cols = np.append(df_cols, "meteo_cat")

    # -----------------------------
    # Masks (same intent as your notebook)
    # -----------------------------
    F = len(df_cols)
    mask_gate = np.zeros(F, dtype=bool)
    mask_wind = np.zeros(F, dtype=bool)
    mask_embed = np.zeros(F, dtype=bool)
    mask_forecast = np.zeros(F, dtype=bool)
    mask_cloud = np.zeros(F, dtype=bool)

    col_idx = {str(col): i for i, col in enumerate(df_cols)}

    for key in [
        "GHI", "humidity", "precipitation", "air_temp", "sun_elev", "AOD",
        "C_GHI", "Dew_Point", "S_Albedo", "Pressure", "sun_azim_sin",
        "sun_azim_cos", "cloud_Clear", "cloud_Probably_Clear", "cloud_Water",
        "cloud_Super-Cooled_Water", "cloud_Mixed", "cloud_Opaque_Ice",
        "cloud_Cirrus", "cloud_Overlapping", "cloud_Overshooting",
        "cloud_Overshooting", "NDVI", "toa", "wind_sp",
    ]:
        if key in col_idx:
            mask_gate[col_idx[key]] = True

    for key in ["wind_dir", "wind_sp"]:
        if key in col_idx:
            mask_wind[col_idx[key]] = True

    for key in [
        "GHI", "humidity", "precipitation", "air_temp", "sun_elev", "AOD",
        "C_GHI", "Dew_Point", "S_Albedo", "Pressure", "sun_azim_sin",
        "sun_azim_cos", "cloud_Clear", "cloud_Probably_Clear", "cloud_Water",
        "cloud_Super-Cooled_Water", "cloud_Mixed", "cloud_Opaque_Ice",
        "cloud_Cirrus", "cloud_Overlapping", "cloud_Overshooting",
    ]:
        if key in col_idx:
            mask_embed[col_idx[key]] = True

    for key in [
        "cloud_Clear", "cloud_Probably_Clear", "cloud_Water",
        "cloud_Super-Cooled_Water", "cloud_Mixed", "cloud_Opaque_Ice",
        "cloud_Cirrus", "cloud_Overlapping", "cloud_Overshooting",
    ]:
        if key in col_idx:
            mask_cloud[col_idx[key]] = True

    for key in list(col_idx.keys()):
        if key not in ["wind_dir", "toa", "NDVI"]:
            mask_forecast[col_idx[key]] = True

    masks = {
        "mask_gate": mask_gate,
        "mask_wind": mask_wind,
        "mask_embed": mask_embed,
        "mask_forecast": mask_forecast,
        "mask_cloud": mask_cloud,
    }

    return PreprocessArtifacts(temporal_node_tensor, temporal_target_tensor, df_cols, masks)


# -----------------------------
# Convenience helpers
# -----------------------------
def get_wind_positions(df_cols: np.ndarray, masks: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """
    Returns positions (within the wind-mask-selected feature list) of:
      - wind_dir
      - wind_sp
    This matches your notebook logic.
    """
    wind_cols = df_cols[masks["mask_wind"]].tolist()
    if "wind_dir" not in wind_cols or "wind_sp" not in wind_cols:
        raise KeyError("wind_dir and/or wind_sp not found inside mask_wind selection.")
    return wind_cols.index("wind_dir"), wind_cols.index("wind_sp")
