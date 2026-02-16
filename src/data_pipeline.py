# src/data_pipeline.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class GeoArtifacts:
    vars_geo: pd.DataFrame              # (N, G) scaled
    coords: pd.DataFrame                # (N, 2) raw latitude/longitude
    station_files: List[str]            # station_code list
    geo_scaler: MinMaxScaler            # fitted scaler


@dataclass
class TemporalArtifacts:
    temporal_node_tensor: np.ndarray    # (T, N, F) float32
    temporal_target_tensor: np.ndarray  # (T, N) float32
    df_cols: np.ndarray                 # (F,) dtype object/str
    masks: Dict[str, np.ndarray]        # boolean masks


# -----------------------------
# Geo features
# -----------------------------
def load_geo_and_build_static_features(
    geo_dir: str,
    stations_filename: str = "stations.csv",
) -> GeoArtifacts:
    stations_csv = os.path.join(geo_dir, stations_filename)
    df_geo = pd.read_csv(stations_csv)
    station_files = df_geo["station_code"].tolist()
    coords = df_geo[["latitude", "longitude"]].copy()

    vars_geo = df_geo[["height", "Slope_DEM2_U1", "Aspect_DEM2_1", "rastercalc"]].rename(
        columns={"Slope_DEM2_U1": "slope", "Aspect_DEM2_1": "aspect", "rastercalc": "twi"}
    )

    aspect_rad = np.deg2rad(vars_geo["aspect"].to_numpy())
    vars_geo["aspect_cos"] = np.cos(aspect_rad)
    vars_geo["aspect_sin"] = np.sin(aspect_rad)
    vars_geo = vars_geo.drop(columns=["aspect"])

    geo_scaler = MinMaxScaler()
    vars_geo_scaled = geo_scaler.fit_transform(vars_geo.to_numpy())
    vars_geo = pd.DataFrame(vars_geo_scaled, columns=vars_geo.columns)

    return GeoArtifacts(
        vars_geo=vars_geo,
        coords=coords,
        station_files=station_files,
        geo_scaler=geo_scaler,
    )


# -----------------------------
# NPZ I/O
# -----------------------------
def _npz_paths(artifacts_dir: str, suffix: str = "2") -> Dict[str, str]:
    return {
        "node_tensor": os.path.join(artifacts_dir, f"node_tensor{suffix}.npz"),
        "target_tensor": os.path.join(artifacts_dir, f"target_tensor{suffix}.npz"),
        "columns": os.path.join(artifacts_dir, f"columns{suffix}.npz"),
        "masks": os.path.join(artifacts_dir, f"masks{suffix}.npz"),
    }


def load_temporal_artifacts(artifacts_dir: str, suffix: str = "2") -> TemporalArtifacts:
    paths = _npz_paths(artifacts_dir, suffix)

    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")

    node = np.load(paths["node_tensor"], allow_pickle=True)["data"]
    tgt = np.load(paths["target_tensor"], allow_pickle=True)["data"]
    cols = np.load(paths["columns"], allow_pickle=True)["data"]
    masks_file = np.load(paths["masks"], allow_pickle=True)
    masks = {k: masks_file[k] for k in masks_file.files}

    return TemporalArtifacts(
        temporal_node_tensor=node,
        temporal_target_tensor=tgt,
        df_cols=cols,
        masks=masks,
    )


def save_temporal_artifacts(
    artifacts_dir: str,
    artifacts: TemporalArtifacts,
    suffix: str = "2",
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    paths = _npz_paths(artifacts_dir, suffix)

    np.savez_compressed(paths["node_tensor"], data=artifacts.temporal_node_tensor)
    np.savez_compressed(paths["target_tensor"], data=artifacts.temporal_target_tensor)
    np.savez_compressed(paths["columns"], data=artifacts.df_cols)
    np.savez_compressed(paths["masks"], **artifacts.masks)


# -----------------------------
# Build tensors from *_prep.csv
# -----------------------------
def build_temporal_from_prep_csv(
    station_files: List[str],
    raw_prep_dir: str,
    *,
    prep_suffix: str = "_prep.csv",
    target_col: str = "GHI",
    keep_doy_tod: bool = True,
    exclude_from_norm: Optional[List[str]] = None,
) -> TemporalArtifacts:
    """
    Builds temporal tensors (T,N,F) and masks from per-station *_prep.csv.
    - Preserves doy/tod (sin/cos) by default
    - Omits meteorological category (meteo_cat)
    """
    if exclude_from_norm is None:
        exclude_from_norm = ["NDVI", "wind_dir"]  # cloud_* handled below

    df_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    df_cols_ref: Optional[List[str]] = None

    for station in station_files:
        path = os.path.join(raw_prep_dir, f"{station}{prep_suffix}")
        df = pd.read_csv(path)

        ts = pd.to_datetime(df.iloc[:, 0])
        if keep_doy_tod:
            doy = ts.dt.dayofyear.to_numpy()
            tod = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy()
        else:
            doy = tod = None  # unused

        assign_kwargs = dict(
            wind_dir_sin=lambda x: np.sin(np.radians(x["wind_dir"])),
            wind_dir_cos=lambda x: np.cos(np.radians(x["wind_dir"])),
            sun_azim_sin=lambda x: np.sin(np.radians(x["sun_azim"])),
            sun_azim_cos=lambda x: np.cos(np.radians(x["sun_azim"])),
        )
        if keep_doy_tod:
            assign_kwargs.update(
                doy_sin=np.sin(2 * np.pi * doy / 365.0),
                doy_cos=np.cos(2 * np.pi * doy / 365.0),
                tod_sin=np.sin(2 * np.pi * tod / 86400.0),
                tod_cos=np.cos(2 * np.pi * tod / 86400.0),
            )

        df = df.assign(**assign_kwargs)

        # Drop raw sun_azim, keep wind_dir last
        if "sun_azim" in df.columns:
            df = df.drop(columns=["sun_azim"])
        cols = [c for c in df.columns if c != "wind_dir"] + ["wind_dir"]
        df = df[cols]

        features = df.iloc[:, 1:]  # drop timestamp
        target = df[target_col].to_numpy(dtype=np.float32)

        if df_cols_ref is None:
            df_cols_ref = features.columns.to_list()
        elif features.columns.to_list() != df_cols_ref:
            raise ValueError(f"Column mismatch at station {station}. Check *_prep.csv consistency.")

        df_list.append(features.to_numpy(dtype=np.float32))
        target_list.append(target)

    if df_cols_ref is None:
        raise ValueError("No stations loaded.")

    df_cols = np.array(df_cols_ref, dtype=object)
    temporal_node_tensor = np.stack(df_list, axis=1).astype(np.float32)        # (T,N,F)
    temporal_target_tensor = np.stack(target_list, axis=1).astype(np.float32) # (T,N)

    # Normalize selected features
    cloud_cols = [c for c in df_cols if str(c).startswith("cloud_")]
    non_norm = set(exclude_from_norm) | set(cloud_cols)

    norm_mask = np.array([c not in non_norm for c in df_cols], dtype=bool)
    notnorm_mask = ~norm_mask

    norm_node = temporal_node_tensor[:, :, norm_mask]
    notnorm_node = temporal_node_tensor[:, :, notnorm_mask]

    T, N, F_norm = norm_node.shape
    scaler = MinMaxScaler()
    norm_2d = norm_node.reshape(-1, F_norm)
    norm_2d_scaled = scaler.fit_transform(norm_2d)
    norm_node_scaled = norm_2d_scaled.reshape(T, N, F_norm).astype(np.float32)

    out = np.zeros_like(temporal_node_tensor, dtype=np.float32)
    out[:, :, norm_mask] = norm_node_scaled
    out[:, :, notnorm_mask] = notnorm_node.astype(np.float32)
    temporal_node_tensor = out

    masks = build_feature_masks(df_cols)

    return TemporalArtifacts(
        temporal_node_tensor=temporal_node_tensor,
        temporal_target_tensor=temporal_target_tensor,
        df_cols=df_cols,
        masks=masks,
    )


# -----------------------------
# Masks + helpers
# -----------------------------
def build_feature_masks(df_cols: np.ndarray) -> Dict[str, np.ndarray]:
    F = len(df_cols)
    mask_gate = np.zeros(F, dtype=bool)
    mask_wind = np.zeros(F, dtype=bool)
    mask_embed = np.zeros(F, dtype=bool)
    mask_forecast = np.zeros(F, dtype=bool)
    mask_cloud = np.zeros(F, dtype=bool)

    col_idx = {col: i for i, col in enumerate(df_cols)}

    for key in [
        "GHI","humidity","precipitation","air_temp","sun_elev","AOD",
        "C_GHI","Dew_Point","S_Albedo","Pressure","sun_azim_sin","sun_azim_cos",
        "cloud_Clear","cloud_Probably_Clear","cloud_Water","cloud_Super-Cooled_Water",
        "cloud_Mixed","cloud_Opaque_Ice","cloud_Cirrus","cloud_Overlapping","cloud_Overshooting",
        "NDVI","toa","wind_sp"
    ]:
        if key in col_idx:
            mask_gate[col_idx[key]] = True

    for key in ["wind_dir", "wind_sp"]:
        if key in col_idx:
            mask_wind[col_idx[key]] = True

    for key in [
        "GHI","humidity","precipitation","air_temp","sun_elev","AOD",
        "C_GHI","Dew_Point","S_Albedo","Pressure","sun_azim_sin","sun_azim_cos",
        "cloud_Clear","cloud_Probably_Clear","cloud_Water","cloud_Super-Cooled_Water",
        "cloud_Mixed","cloud_Opaque_Ice","cloud_Cirrus","cloud_Overlapping","cloud_Overshooting"
    ]:
        if key in col_idx:
            mask_embed[col_idx[key]] = True

    for key in [
        "cloud_Clear","cloud_Probably_Clear","cloud_Water","cloud_Super-Cooled_Water",
        "cloud_Mixed","cloud_Opaque_Ice","cloud_Cirrus","cloud_Overlapping","cloud_Overshooting"
    ]:
        if key in col_idx:
            mask_cloud[col_idx[key]] = True

    for key in list(col_idx.keys()):
        if key not in ["wind_dir", "toa", "NDVI"]:
            mask_forecast[col_idx[key]] = True

    return {
        "mask_gate": mask_gate,
        "mask_wind": mask_wind,
        "mask_embed": mask_embed,
        "mask_forecast": mask_forecast,
        "mask_cloud": mask_cloud,
    }


def get_wind_positions(df_cols: np.ndarray, masks: Dict[str, np.ndarray]) -> Tuple[int, int]:
    wind_cols = df_cols[masks["mask_wind"]].tolist()
    if "wind_dir" not in wind_cols or "wind_sp" not in wind_cols:
        raise KeyError("wind_dir and/or wind_sp not found in wind-mask-selected columns.")
    return wind_cols.index("wind_dir"), wind_cols.index("wind_sp")


class GraphSequenceDataset(Dataset):
    def __init__(
        self,
        node_tensor: np.ndarray,
        target_tensor: np.ndarray,
        masks: Dict[str, np.ndarray],
        lags: int = 6,
        horizon: int = 1,
    ) -> None:
        """
        Args:
            node_tensor: [T, N, F] numpy array
            target_tensor: [T, N] numpy array
            masks: dictionary with feature masks
            lags: input sequence length
            horizon: forecast horizon
        """
        self.node_tensor = node_tensor
        self.target_tensor = target_tensor
        self.masks = masks
        self.lags = lags
        self.horizon = horizon

        T = node_tensor.shape[0]
        self.length = T - lags - horizon + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        # Forecast features (drop meteo_cat)
        x_seq_full = self.node_tensor[idx : idx + self.lags, :, self.masks["mask_forecast"]]
        x_seq = torch.tensor(x_seq_full[..., :-1], dtype=torch.float32)

        # Embedor features
        e_seq = torch.tensor(
            self.node_tensor[idx : idx + self.lags, :, self.masks["mask_embed"]],
            dtype=torch.float32,
        )

        # Wind features
        w_seq = torch.tensor(
            self.node_tensor[idx : idx + self.lags, :, self.masks["mask_wind"]],
            dtype=torch.float32,
        )

        # Gate features
        g_seq = torch.tensor(
            self.node_tensor[idx : idx + self.lags, :, self.masks["mask_gate"]],
            dtype=torch.float32,
        )

        # Targets
        y_seq = torch.tensor(
            self.target_tensor[idx + self.lags : idx + self.lags + self.horizon],
            dtype=torch.float32,
        )

        return x_seq, e_seq, w_seq, g_seq, y_seq
