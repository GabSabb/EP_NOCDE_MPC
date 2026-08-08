import os

# Keep JAX's default allocator for substantially lower allocation overhead.
# If GPU memory growth becomes a problem, the slower "platform" allocator can
# be restored manually.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import concurrent.futures
import csv
import io
import multiprocessing as mp
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as tu
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from trajax import optimizers

warnings.filterwarnings("ignore")
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
print(jax.devices())

class Func(eqx.Module):
    mmlp_NODE: eqx.Module
    mmlp_NCDE: eqx.Module
    z_mlp: eqx.Module  # per-zone MLPs, each outputs dz_i (kz,)

    control_interp: diffrax.LinearInterpolation = eqx.field(static=True)
    data_size: int
    hidden_size_CDE: int = 1

    # --- cached buffers / constants to avoid allocations in RHS ---
    n_zones: int = eqx.field(static=True)
    aux_dim: int = eqx.field(static=True)
    kz: int = eqx.field(static=True)          # per-zone latent dim
    latent_dim: int = eqx.field(static=True)  # total latent dim = n_zones * kz

    ones_z: jax.Array
    node_in_raw_buf: jax.Array
    z_in_raw_buf: jax.Array
    ncde_in_raw_buf: jax.Array
    dX_buf: jax.Array

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        hidden_sizes,
        data_size: int,
        masks: list,
        control_interp,
        num_zones: int,
        mmlp_flag: bool = False,
        activation: Callable = None,
        final_activation: Callable = None,
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        key: jax.random.PRNGKey,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if activation is None:
            activation = jax.nn.soft_sign

        self.data_size = data_size
        self.control_interp = control_interp

        self.n_zones = num_zones
        self.aux_dim = 2

        self.kz = 2  # per-zone latent dim
        self.latent_dim = self.n_zones * self.kz
        node_in_dim_raw = self.n_zones + self.kz + 1 + self.aux_dim  # [T_all, z_i, P_i, aux]
        z_in_dim_raw    = self.n_zones + 1 + self.kz + self.aux_dim  # [T_all, P_i, z_i, aux]
        ncde_in_dim_raw = 7 + self.aux_dim

        key_node, key_ncde, key_z = jr.split(key, 3)
        keys_node = jr.split(key_node, self.n_zones)
        keys_ncde = jr.split(key_ncde, self.n_zones)
        keys_z    = jr.split(key_z, self.n_zones)

        def make_node_mlp(k):
            return eqx.nn.MLP(
                in_size=node_in_dim_raw,
                out_size=1,
                width_size=48,
                depth=2,
                activation=activation,
                key=k,
            )

        def make_ncde_mlp(k):
            return eqx.nn.MLP(
                in_size=ncde_in_dim_raw,
                out_size=3,
                width_size=48,
                depth=2,
                activation=activation,
                key=k,
            )

        def make_z_mlp(k):
            return eqx.nn.MLP(
                in_size=z_in_dim_raw,
                out_size=self.kz,
                width_size=48,
                depth=2,
                activation=activation,
                key=k,
            )

        self.mmlp_NODE = eqx.filter_vmap(make_node_mlp)(keys_node)
        self.mmlp_NCDE = eqx.filter_vmap(make_ncde_mlp)(keys_ncde)
        self.z_mlp     = eqx.filter_vmap(make_z_mlp)(keys_z)

        # ---- cached constants/buffers ----
        self.ones_z = jnp.ones((self.n_zones,), dtype=jnp.float32)

        self.node_in_raw_buf = jnp.zeros((self.n_zones, node_in_dim_raw), dtype=jnp.float32)
        self.z_in_raw_buf    = jnp.zeros((self.n_zones, z_in_dim_raw),    dtype=jnp.float32)
        self.ncde_in_raw_buf = jnp.zeros((self.n_zones, ncde_in_dim_raw), dtype=jnp.float32)
        self.dX_buf          = jnp.zeros((self.n_zones, 3),               dtype=jnp.float32)

    def __call__(self, t, y, args):
        ts, interpolator_us, interpolator, us = args

        n   = self.n_zones
        ts0 = ts[0]
        kz  = self.kz
        k_total = self.latent_dim

        # Evaluate interpolators once
        aux = jnp.ravel(interpolator.evaluate(t))      # (aux_dim,)
        u   = jnp.ravel(interpolator_us.evaluate(t))   # (n,)

        # Slice y: [T(0:n), P(n:2n), z(2n:2n+k_total)]
        y_base = y[:n]                      # (n,)
        y_p    = y[n:2 * n]                 # (n,)
        z      = y[2 * n:2 * n + k_total]   # (n*kz,)
        z_mat  = z.reshape(n, kz)           # (n, kz)

        # 1) NODE: dT = f_i(T_all, z_i, P_i, aux)
        node_in_raw = self.node_in_raw_buf.astype(y.dtype)
        node_in_raw = node_in_raw.at[:, :n].set(y_base[None, :])  # T_all
        node_in_raw = node_in_raw.at[:, n:n + kz].set(z_mat)      # z_i
        node_in_raw = node_in_raw.at[:, n + kz].set(y_p)          # P_i
        node_in_raw = node_in_raw.at[:, n + kz + 1:n + kz + 1 + self.aux_dim].set(aux[None, :])  # aux

        node_out = eqx.filter_vmap(lambda m, x: m(x))(self.mmlp_NODE, node_in_raw)
        node_out = jnp.ravel(node_out)  # (n,)

        # 1b) Latent dynamics :
        z_in_raw = self.z_in_raw_buf.astype(y.dtype)
        z_in_raw = z_in_raw.at[:, :n].set(y_base[None, :])                    # T_all
        z_in_raw = z_in_raw.at[:, n].set(y_p)                                 # P_i
        z_in_raw = z_in_raw.at[:, n + 1:n + 1 + kz].set(z_mat)                # z_i
        z_in_raw = z_in_raw.at[:, n + 1 + kz:n + 1 + kz + self.aux_dim].set(aux[None, :])  # aux

        dz_mat = eqx.filter_vmap(lambda m, x: m(x))(self.z_mlp, z_in_raw)      # (n, kz)
        dz     = dz_mat.reshape(-1)                                            # (n*kz,)

        # 2) Control derivatives
        d_ctrl = self.control_interp.derivative(t)  # (n, 2)
        dt    = d_ctrl[:, 0]
        dTSet = d_ctrl[:, 1]

        d_aux  = interpolator.derivative(t)         # (aux_dim,)
        dT_ext = d_aux[0]                           # scalar (assuming aux[0] is T_ext)

        # 3) NCDE: dP from learned vector field * dX
        ncde_in_raw = self.ncde_in_raw_buf.astype(y.dtype)
        ncde_in_raw = ncde_in_raw.at[:, 0].set((t - ts0) * self.ones_z.astype(y.dtype))
        ncde_in_raw = ncde_in_raw.at[:, 1].set(y_base)
        ncde_in_raw = ncde_in_raw.at[:, 2].set(u[:n])
        ncde_in_raw = ncde_in_raw.at[:, 3].set(u[:n] - y_base)
        ncde_in_raw = ncde_in_raw.at[:, 4].set(node_out)
        ncde_in_raw = ncde_in_raw.at[:, 5].set(dTSet)
        ncde_in_raw = ncde_in_raw.at[:, 6].set(y_p)
        ncde_in_raw = ncde_in_raw.at[:, 7:7 + self.aux_dim].set(aux[None, :])

        ncde_vec = eqx.filter_vmap(lambda m, x: m(x))(self.mmlp_NCDE, ncde_in_raw)  # (n, 3)

        dX = self.dX_buf.astype(y.dtype)
        dX = dX.at[:, 0].set(dt)
        dX = dX.at[:, 1].set(dTSet)
        dX = dX.at[:, 2].set(dT_ext * self.ones_z.astype(y.dtype))

        ncde_out = jnp.einsum("bi,bi->b", ncde_vec, dX)  # (n,)

        # 4) Smooth saturation on dP + return dy
        dT = node_out
        dP = ncde_out

        P = y_p
        x = jnp.clip((1.0 - P) / 0.02, 0.0, 1.0)
        up_room = x * x * (3.0 - 2.0 * x)
        x = jnp.clip((P - 0.0) / 0.02, 0.0, 1.0)
        down_room = x * x * (3.0 - 2.0 * x)
        scale = jnp.where(dP >= 0.0, up_room, down_room)
        dP = dP * scale

        dy = jnp.concatenate([dT, dP, dz], axis=0)  # length 2n + n*kz
        return dy



class NeuralODE(eqx.Module):
    func: Func
    z_encoder: eqx.Module  # y0 -> z0

    def __init__(
        self,
        in_size,
        out_size,
        hidden_sizes,
        data_size,
        masks,
        control_interp,
        mmlp_flag,
        num_zones,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)

        key_func, key_enc = jr.split(key, 2)

        self.func = Func(
            in_size,
            out_size,
            hidden_sizes,
            data_size,
            masks,
            control_interp,
            num_zones,
            mmlp_flag,
            key=key_func,
        )

        k_total = self.func.latent_dim
        y0_dim = 2 * num_zones

        if k_total > 0:
            self.z_encoder = eqx.nn.MLP(
                in_size=y0_dim,
                out_size=k_total,
                width_size=48,
                depth=2,
                activation=jax.nn.soft_sign,
                final_activation=jax.nn.identity,
                key=key_enc,
            )
        else:
            self.z_encoder = eqx.nn.Identity()

    def __call__(self, ts, y0, args, z0=None, return_same_length=True):
        interp_us  = diffrax.LinearInterpolation(ts, args)
        interp_aux = diffrax.LinearInterpolation(ts, args[:, -2:])

        n = self.func.n_zones
        k_total = self.func.latent_dim

        if k_total > 0:
            if z0 is None:
                z0 = self.z_encoder(y0).astype(y0.dtype)
            y0_main = jnp.concatenate([y0, z0], axis=0)   # [T, P, z]
        else:
            y0_main = y0

        # Augment with cumulative integral of P:
        # I_P'(t) = P(t)
        I0 = jnp.zeros((n,), dtype=y0_main.dtype)
        y0_aug = jnp.concatenate([y0_main, I0], axis=0)  # [T, P, z, I_P]

        args_ode = (ts, interp_us, interp_aux, args)

        def vf(t, y_aug, ode_args):
            y_main = y_aug[: 2 * n + k_total]             # [T, P, z]
            dy_main = self.func(t, y_main, ode_args)      # [dT, dP, dz]
            P_state = y_main[n:2 * n]                     # current power state
            dI = P_state                                  # integrate power
            return jnp.concatenate([dy_main, dI], axis=0)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vf),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0_aug,
            args=args_ode,
            max_steps=100000,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )

        ys = solution.ys                                # [len(ts), 2n + k_total + n]
        y_main = ys[:, : 2 * n + k_total]              # [T, P, z]
        I_hist = ys[:, 2 * n + k_total:]               # [I_P]

        T_pred = y_main[:, :n]                         # pointwise T at ts
        P_point = y_main[:, n:2 * n]                   # pointwise P at ts

        dt_intervals = (ts[1:] - ts[:-1])[:, None]     # [len(ts)-1, 1]
        P_avg_interval = (I_hist[1:] - I_hist[:-1]) / dt_intervals
        # shape: [len(ts)-1, n]
        # This is the average predicted power over each interval [ts[k], ts[k+1]]

        if return_same_length:
            # Align target at t_{k+1} with average over [t_k, t_{k+1}]
            # Keep first sample as pointwise initial value
            P_for_loss = jnp.concatenate([P_point[:1], P_avg_interval], axis=0)
        else:
            # Use only interval-averaged values
            P_for_loss = P_avg_interval

        return jnp.concatenate([T_pred, P_for_loss], axis=1)


def split_sequence_data(data, timestamps, split_ratio):
    # 0) Sort by time
    timestamps_series = pd.to_datetime(timestamps)
    sort_idx = np.argsort(timestamps_series.values)
    timestamps_sorted = timestamps_series.values[sort_idx]
    data_sorted = data[sort_idx]

    # 1) Group into unique days
    dates = pd.to_datetime(timestamps_sorted).normalize().to_numpy()
    unique_dates, start_indices, counts = np.unique(
        dates, return_index=True, return_counts=True
    )

    total_days = max(len(unique_dates) - 1, 0)

    # 2) Normalize split_ratio
    ratios = np.asarray(split_ratio, dtype=float)
    if ratios.sum() <= 0:
        raise ValueError("split_ratio must sum to a positive number")

    ratios = ratios / ratios.sum() 

    if len(ratios) == 2:
        r_train, r_val = ratios
        r_test = 0.0
    elif len(ratios) == 3:
        r_train, r_val, r_test = ratios
    else:
        raise ValueError("split_ratio must have length 2 or 3")

    # 3) Convert ratios -> day counts
    train_days = int(np.floor(r_train * total_days))
    val_days   = int(np.floor(r_val   * total_days))
    test_days  = total_days - train_days - val_days  # remainder

    if test_days < 0:
        test_days = 0
        overflow = (train_days + val_days) - total_days
        if overflow > 0:
            take = min(overflow, val_days)
            val_days -= take
            overflow -= take
        if overflow > 0:
            train_days = max(train_days - overflow, 0)

    # 4) Helper to slice by day range
    def get_slice(start_day, num_days):
        if num_days <= 0 or start_day >= len(start_indices):
            return slice(0, 0)
        start_idx = start_indices[start_day]
        end_day = start_day + num_days
        end_idx = start_indices[end_day] if end_day < len(start_indices) else len(data_sorted)
        return slice(start_idx, end_idx)

    train_slice = get_slice(0, train_days)
    val_slice   = get_slice(train_days, val_days)
    test_slice  = get_slice(train_days + val_days, test_days)

    return (
        data_sorted[train_slice], data_sorted[val_slice], data_sorted[test_slice],
        timestamps_sorted[train_slice], timestamps_sorted[val_slice], timestamps_sorted[test_slice]
    )


def create_sliding_windows(data, timestamps, window_size=24, stride=1, jax_flag=True):
    # Ensure timestamps are NumPy datetime64 array
    timestamps = np.asarray(timestamps)
    if not np.issubdtype(timestamps.dtype, np.datetime64):
        timestamps = pd.to_datetime(timestamps).to_numpy()

    timestamps_hours = timestamps.astype('datetime64[h]')
    min_time = timestamps_hours[0]
    max_time = timestamps_hours[-1]

    windowed_data = []
    windowed_timestamps = []

    delta = np.timedelta64(window_size, 'h')
    stride_delta = np.timedelta64(stride, 'h')
    current_start = min_time

    while current_start + delta <= max_time:
        current_end = current_start + delta
        mask = (timestamps_hours >= current_start) & (timestamps_hours < current_end)
        window = data[mask]
        if len(window) >= window_size:
            if jax_flag:
                windowed_data.append(jnp.array(window))  # JAX-compatible array
            else:
                windowed_data.append(np.array(window))
            windowed_timestamps.append(timestamps[mask])
        current_start += stride_delta

    return windowed_data, windowed_timestamps

def build_presence_masks(list1, list2):
    return [
        jnp.isin(arr1, arr2, assume_unique=True).astype(jnp.int32)
        for arr1, arr2 in zip(list1, list2)
    ]

def selective_anchor_scaler(scaler, features_to_anchor):
    """
    Anchors selected features at 0 and leaves others unchanged.

    Args:
        scaler: fitted MinMaxScaler
        features_to_anchor: list or array of feature indices to anchor at 0
    """
    # Clone the original min
    new_data_min = scaler.data_min_.copy()
    
    # Set specified features' min to 0
    new_data_min[features_to_anchor] = 0.0
    
    # Update scaler internals
    scaler.data_min_ = new_data_min
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    
    feature_range = scaler.feature_range
    scaler.scale_ = (feature_range[1] - feature_range[0]) / scaler.data_range_
    scaler.min_ = feature_range[0] - scaler.data_min_ * scaler.scale_
    
    return scaler

def backward_fill_initial_nans(arr):
    # Loop over each column
    for col in range(arr.shape[1]):
        # Get the column as a 1D array
        column_data = arr[:, col]
        
        # Find the first index where the column is not NaN
        first_non_nan_idx = jnp.where(~jnp.isnan(column_data))[0]
        
        # If there's at least one non-NaN value, perform backward fill
        if first_non_nan_idx.size > 0:
            first_non_nan_idx = first_non_nan_idx[0]  # Take the first valid index
            
            # Replace NaNs before the first non-NaN index with that first value
            # column_data[:first_non_nan_idx] = column_data[first_non_nan_idx]
            fill_value = column_data[first_non_nan_idx]
            column_data = column_data.at[:first_non_nan_idx].set(fill_value)
        arr = arr.at[:,col].set(column_data)

    
    return arr


def get_data(time_horizon = 6, split_ratio = [(19/21)*100, (2/21)*100], stride = 1, num_zones = 10, first_zone_index = 1, house_idx = None):

    df = pd.read_csv(SIMULATED_DATA_CSV_PATH)


    # Convert to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Set as index
    df = df.set_index("time")

    df = df.sort_index()


    t0 = df.index[0]

    new_times = [
        t0 - pd.Timedelta(minutes=30),  # 18:00
        t0 - pd.Timedelta(minutes=15),  # 18:15
    ]

    df_pre = pd.DataFrame(
        index=pd.DatetimeIndex(new_times, name=df.index.name),
        columns=df.columns,
        dtype=float,
    )

    df = pd.concat([df_pre, df])
    df = df.sort_index()

    df = df.bfill()

    power_cols = [c for c in df.columns if c.startswith("power_")]
    df[power_cols] = df[power_cols].abs()


    zones_to_remove = ["salle_bain", "heat_pump"]

    cols_to_drop = [
        c for c in df.columns
        if any(zone in c for zone in zones_to_remove)
    ]

    df = df.drop(columns=cols_to_drop)


    zone_names = [
        "garage",
        "sous_sol_1",
        "sous_sol_2",
        "cuisine",
        "salle_manger",
        "salle_eau",
        "chambre_1",
        "chambre_2",
        "chambre_3",
        "salon",
    ]

    rename_map = {}

    for i, zone in enumerate(zone_names):
        rename_map[f"temperature_{zone}"] = f"ambient_temperature_average_Z{i}"
        rename_map[f"power_{zone}"]       = f"energy_consumed_calculated_wh_Z{i}"
        rename_map[f"setpoint_{zone}"]    = f"ambient_setpoint_average_Z{i}"

    df = df.rename(columns=rename_map)

    df = df.drop(columns=df.filter(regex=r'^demand').columns)


    # Build one control tensor: (T, num_zones, 2)  -> [time, setpoint] per zone
    control_stack = []
    for index_zone in range(num_zones):
        df_data = df[[f'ambient_temperature_average_Z{index_zone}',
                    f'ambient_setpoint_average_Z{index_zone}',
                    f'energy_consumed_calculated_wh_Z{index_zone}']].copy()

        df_data["time_"] = df_data.index
        t0 = df_data["time_"].iloc[0]
        df_data["time_diff_seconds"] = (df_data["time_"] - t0).dt.total_seconds()
        df_data["time_diff_hours"] = df_data["time_diff_seconds"] / 3600

        _ts = jnp.asarray(df_data["time_diff_hours"].values)

        Xs = jnp.stack(
            [
                _ts,
                jnp.asarray(df_data[f"ambient_setpoint_average_Z{index_zone}"].values),
            ],
            axis=-1,   # (T, 2)
        )

        Xs = backward_fill_initial_nans(Xs)

        col_min = jnp.min(Xs, axis=0)
        col_max = jnp.max(Xs, axis=0)

        Xs = (Xs - col_min) / (col_max - col_min + 1e-6)
        Xs = Xs.at[:, 0].set(Xs[:, 0] * (col_max[0] - col_min[0] + 1e-6) + col_min[0])

        control_stack.append(Xs)  # list of (T,2)

    control_stack = jnp.stack(control_stack, axis=1)  # (T, num_zones, 2)

    # ONE interpolator for all zones
    control_interp = diffrax.LinearInterpolation(_ts, control_stack)



    data_size = 2


    temp_names = []

    for i in range(0, num_zones):
        if i < 10:
            temp_names.append(f'ambient_temperature_average_Z{i}')
        else:
            temp_names.append(f'ambient_temperature_average_Z{i}')

    for i in range(0, num_zones):
        if i < 10:
            temp_names.append(f'energy_consumed_calculated_wh_Z{i}')
        else:
            temp_names.append(f'energy_consumed_calculated_wh_Z{i}')


    df_data = df.copy()

    # 1) Build x (targets) and mask
    x_df = df_data[temp_names].copy()  # columns: [T0..Tn-1, P0..Pn-1]

    # mask: 1 if observed, 0 if missing
    x_mask_df = (~x_df.isna()).astype(np.float32)

    # Fill NaNs ONLY so downstream code (scalers, windows) can run
    x_df_filled = x_df.fillna(0.0)


    u_df = df_data.loc[:, ~df_data.columns.isin(temp_names)].copy()

    # Keep everything as DataFrames
    df_data = pd.concat([x_df_filled, u_df], axis=1)

    # Convert to numpy arrays
    x_array = df_data[temp_names].to_numpy()
    u_array = u_df.to_numpy()
    x_mask  = x_mask_df.to_numpy()

    timestamps = df_data.index.to_numpy()


    (x_train, x_val, x_test, _t_train, _t_val, _t_test) = split_sequence_data(x_array, timestamps, split_ratio)
    (mask_train, mask_val, mask_test, _, _, _) = split_sequence_data(x_mask, timestamps, split_ratio)
    (u_train, u_val, u_test, _, _, _) = split_sequence_data(u_array, timestamps, split_ratio)



    u_scaler = MinMaxScaler()
    u_scaler.fit(u_train)


    u_train = u_scaler.transform(u_train)
    u_val = u_scaler.transform(u_val)

    
    x_scaler = MinMaxScaler()
    x_scaler.fit(x_train)

    x_train = x_scaler.transform(x_train)
    x_val = x_scaler.transform(x_val)


    
    x_train, t_train = create_sliding_windows(x_train, _t_train, time_horizon, stride)
    x_val, t_val = create_sliding_windows(x_val, _t_val, time_horizon, stride)

    mask_train, _ = create_sliding_windows(mask_train, _t_train, time_horizon, stride)
    mask_val, _   = create_sliding_windows(mask_val,   _t_val,   time_horizon, stride)

    
    initial_time = timestamps[0]


    t_train_float = [
    jnp.asarray((ts_batch - initial_time) / np.timedelta64(1, 'h'), dtype=jnp.float32)
    for ts_batch in t_train
]


    t_val_float = [
    jnp.asarray((ts_batch - initial_time) / np.timedelta64(1, 'h'), dtype=jnp.float32)
    for ts_batch in t_val
]

    u_train, _ = create_sliding_windows(u_train, _t_train, time_horizon, stride)
    u_val, _ = create_sliding_windows(u_val, _t_val, time_horizon, stride)

    if len(u_test>0):
        u_test = u_scaler.transform(u_test)
        x_test = x_scaler.transform(x_test)
        x_test, t_test = create_sliding_windows(x_test, _t_test, time_horizon, stride)
        t_test_float = [
        jnp.asarray((ts_batch - initial_time) / np.timedelta64(1, 'h'), dtype=jnp.float32)
        for ts_batch in t_test]
        u_test, _ = create_sliding_windows(u_test, _t_test, time_horizon, stride)
        return (t_train_float, t_val_float, t_test_float), (t_train, t_val, t_test), \
        (x_train, x_val, x_test), (u_train, u_val, u_test), \
        (x_scaler, u_scaler), (mask_train, mask_val, mask_test), \
        control_interp, data_size


    return (t_train_float, t_val_float), (t_train, t_val), \
       (x_train, x_val), (u_train, u_val), \
       (x_scaler, u_scaler), (mask_train, mask_val), \
       control_interp, data_size




# =============================================================================
# EnergyPlus plant + NODE/NCDE MPC controller
# =============================================================================
#
# EnergyPlus is used as the "virtual real residence" / plant.
# NODE/NCDE model is  used inside the MPC optimizer.
#
# At each 15-minute control boundary:
#   1) read current zone temperatures from EnergyPlus,
#   2) solve the MPC horizon using historical Open-Meteo weather,
#   3) apply only the first optimized setpoint vector to EnergyPlus,
#   4) continue the EnergyPlus simulation until the next control boundary.
#
# =============================================================================

# ---- Trajax compatibility for newer JAX versions ----
if not hasattr(jax, "tree_map"):
    jax.tree_map = tu.tree_map

# 0) USER SETTINGS

# Repository-relative defaults make a clone runnable without editing local paths.
# Every path can still be overridden with the corresponding environment variable.
REPOSITORY_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPOSITORY_ROOT / "data"
ENERGYPLUS_ASSET_DIR = REPOSITORY_ROOT / "energyplus"
MODEL_DIR = REPOSITORY_ROOT / "models"

# IDF Path
IDF_PATH = os.environ.get(
    "MEEB_IDF_PATH",
    str(ENERGYPLUS_ASSET_DIR / "MEEB2013_0deg.idf"),
)

# EnergyPlus still needs an EPW file to start a weather-file RunPeriod.
# If None, the code tries to find the first .epw file in the same folder as the IDF.
WEATHER_EPW_PATH = os.environ.get(
    "MEEB_EPW_PATH",
    str(ENERGYPLUS_ASSET_DIR / "Shawinigan_OpenMeteo_2025-12-01_to_2026-04-01.epw"),
)

# DR-event spreadsheet exported as CSV. The script reads dateDebut/dateFin and runs each unique DR day.
DR_EVENTS_CSV_PATH = os.environ.get(
    "MEEB_DR_EVENTS_PATH",
    str(DATA_DIR / "evenements-pointe.csv"),
)

# Initial zone temperatures generated by the no-MPC one-day preconditioning script.
# The script reads one row per DR day and uses the 00:00 DR-day temperatures as
# the first MPC prediction state. The EnergyPlus plant itself reproduces the
# same preconditioning day and must match the CSV within the configured tolerance.
INITIAL_ZONE_TEMPERATURES_CSV_PATH = os.environ.get(
    "MEEB_INITIAL_TEMPERATURES_PATH",
    str(DATA_DIR / "initial_zone_temperatures_0000_all_dr_days_wide.csv"),
)

# The trained model's scaling metadata is reconstructed from this dataset.
SIMULATED_DATA_CSV_PATH = os.environ.get(
    "MEEB_TRAINING_DATA_PATH",
    str(DATA_DIR / "simulated_data.csv"),
)
USE_CSV_INITIAL_ZONE_TEMPERATURES = True

# Match the no-MPC script's physical EnergyPlus initialization. The MPC plant
# runs the same rule-based schedule for one full day before the DR day begins.
# At 00:00, the preconditioned EnergyPlus temperatures are compared with the
# no-MPC CSV. A mismatch larger than the tolerance aborts the run instead of
# silently comparing cases with different initial conditions.
PRECONDITIONING_DAYS = 1
REQUIRE_CSV_INITIAL_TEMPERATURES = True
REQUIRE_PRECONDITIONED_EP_CSV_MATCH = True
INITIAL_TEMPERATURE_MATCH_TOLERANCE_C = 0.25

# A direct zone-air-temperature actuator is not generally available in
# EnergyPlus. Matching is therefore achieved through identical preconditioning,
# while the CSV values initialize the first predictive MPC state.
TRY_APPLY_INITIAL_ZONE_TEMPERATURES_TO_ENERGYPLUS = False

# Parallel execution is optional. Keep False unless you have enough RAM/CPU and
# have confirmed that your EnergyPlus + JAX setup is stable in separate spawned
# processes. Each worker loads EnergyPlus, the trained model, and its own output
# directory, so start with 2 workers.
RUN_DAYS_IN_PARALLEL = False
MAX_PARALLEL_WORKERS = 1

# Run each DR day in a fresh Python interpreter. This is the safest mode for
# repeated EnergyPlus Runtime API + JAX/XLA simulations because all native state,
# callbacks, compiled functions, and device memory are released when the child
# process exits.
RUN_DAYS_IN_SUBPROCESSES = True
STOP_BATCH_ON_FIRST_FAILURE = True
SUBPROCESS_TIMEOUT_SECONDS = None  # set to an integer if you want a hard timeout per day

# EnergyPlus installation folder for the Runtime API.
#
# In WSL/Linux, this MUST point to the Linux EnergyPlus install, not the
# Windows folder under /mnt/c.
#
# The code first respects an existing ENERGYPLUS_INSTALL_DIR environment variable.
# If it is not set, it falls back to ~/EnergyPlus-26-1-0 in WSL/Linux.
def default_energyplus_install_dir():
    """
    Return the EnergyPlus install folder to use for the Runtime API.

    In WSL/Linux, never use the Windows install under /mnt/c because it contains
    Windows DLLs, while the Linux Python process needs libenergyplusapi.so.
    """
    env_value = os.environ.get("ENERGYPLUS_INSTALL_DIR", "").strip()

    if os.name == "nt":
        return os.path.expanduser(env_value) if env_value else r"C:\EnergyPlusV26-1-0"

    if env_value:
        env_expanded = os.path.abspath(os.path.expanduser(env_value))
        if "/mnt/c/EnergyPlus" not in env_expanded and (Path(env_expanded) / "libenergyplusapi.so").exists():
            return env_expanded
        print(
            "WARNING: ignoring ENERGYPLUS_INSTALL_DIR because it is not a Linux EnergyPlus API install:\n"
            f"  {env_expanded}\n"
            "Using ~/EnergyPlus-26-1-0 instead."
        )

    linux_candidates = [
        os.path.expanduser("~/EnergyPlus-26-1-0"),
        "/usr/local/EnergyPlus-26-1-0",
    ]
    for candidate in linux_candidates:
        if (Path(candidate) / "libenergyplusapi.so").exists():
            return candidate

    return os.path.expanduser("~/EnergyPlus-26-1-0")


ENERGYPLUS_INSTALL_DIR = default_energyplus_install_dir()

# The IDF has Site:Location = Shawinigan_Qc_Ca, 46.56, -72.77,
# and an existing 2026 RunPeriod. The helper below rewrites the working copy
# to the period you choose here.
SIM_START_DT = pd.Timestamp("2026-02-01 00:00:00", tz="America/Toronto")
SIM_DAYS = 1
SIM_END_DT = SIM_START_DT + pd.Timedelta(days=SIM_DAYS)

# Location for Open-Meteo historical weather and model auxiliary inputs.
# These values are taken from the IDF Site:Location object.
LAT = 46.56
LON = -72.77
OPEN_METEO_TIMEZONE = "America/Toronto"

# If True, EnergyPlus outdoor dry bulb and solar are overridden using the same
# Open-Meteo historical data used by the MPC. If the weather actuators are not
# available in your EnergyPlus build/model, the simulation continues using EPW
# weather, while the MPC still uses Open-Meteo.
OVERRIDE_EP_WEATHER_WITH_OPEN_METEO = False

# Output/logging folder. Historical bias logs can be supplied separately through
# MEEB_LEGACY_BIAS_LOG_DIR; by default previous runs under outputs are reused.
LOG_DIR = os.environ.get(
    "MEEB_OUTPUT_DIR",
    str(REPOSITORY_ROOT / "outputs"),
)
LEGACY_BIAS_LOG_DIR = os.environ.get(
    "MEEB_LEGACY_BIAS_LOG_DIR",
    LOG_DIR,
)

# Trained model path.
MODEL_PATH = os.environ.get(
    "MEEB_MODEL_PATH",
    str(MODEL_DIR / "NOCDE_MEEB_Sim.eqx"),
)

# MPC/model dimensions.
# IMPORTANT: trained MPC model has 10 zones. The IDF has
# 12 thermostat/baseboard zones. Therefore this script controls the 10 zones
# below and holds the remaining IDF-only zones at DEFAULT_UNMANAGED_SETPOINT_C.
NUM_ZONES = 10
# A two-hour controllable preheat window plus a four-hour DR event fits exactly
# in this six-hour horizon, so every applied preheat decision sees event end.
HORIZON_HOURS = 6
STEP_MINUTES = 15
TOTAL_MPC_RUNS = int((SIM_END_DT - SIM_START_DT).total_seconds() / (STEP_MINUTES * 60))

CONTROL_STEP = pd.Timedelta(minutes=STEP_MINUTES)
PRECONDITIONING_STEPS = int(
    PRECONDITIONING_DAYS * 24 * 60 / STEP_MINUTES
)

def control_grid_dt(step_idx: int) -> pd.Timestamp:
    """
    Exact MPC/control-grid timestamp.
    step_idx = 0 -> SIM_START_DT
    step_idx = 1 -> SIM_START_DT + 15 min
    ...
    """
    return (
        pd.Timestamp(SIM_START_DT)
        .tz_convert(OPEN_METEO_TIMEZONE)
        + step_idx * CONTROL_STEP
    )

# MPC model zones in the same order as trained model state.
# These names are controller-facing names, EP_ZONE_NAMES maps them to the
# exact EnergyPlus zone names in the IDF.
ZONE_ORDER = [
    "Garage",
    "Base 1",
    "Base 2",
    "Kitchen",
    "Dining",
    "Bath",
    "Bed 1",
    "Bed 2",
    "Bed 3",
    "Living",
]

# Exact EnergyPlus zone names from the IDF.
# NOTE: the controller-facing "Bath" state is now mapped to Bathroom2;
# Bathroom1 is intentionally left unmanaged below.
EP_ZONE_NAMES = {
    "Garage": "Garage",
    "Base 1": "Basement1",
    "Base 2": "Basement2",
    "Kitchen": "Kitchen",
    "Dining": "Dining room",
    "Bath": "Bathroom2",
    "Bed 1": "Bedroom1",
    "Bed 2": "Bedroom2",
    "Bed 3": "Bedroom3",
    "Living": "Living room",
}

# Thermostat heating setpoint schedules found in the IDF.
# IMPORTANT: this version preserves the original Schedule:File objects.
# The Runtime API tries to actuate each schedule value directly.
EP_SETPOINT_SCHEDULE_MAP = {
    "Garage": "SP_GA",
    "Base 1": "SP_BS1",
    "Base 2": "SP_BS2",
    "Kitchen": "SP_KIT",
    "Dining": "SP_DR",
    "Bath": "SP_BA2",
    "Bed 1": "SP_BE1",
    "Bed 2": "SP_BE2",
    "Bed 3": "SP_BE3",
    "Living": "SP_LR",
}

# IDF thermostat zones that are not part of the 10-zone trained MPC model.
# They are kept controlled by EnergyPlus but held at a fixed setpoint.
DEFAULT_UNMANAGED_SETPOINT_C = 18.0
UNMANAGED_SETPOINT_SCHEDULE_MAP = {
    "Hall1": "SP_HALL1",
    "Bathroom1": "SP_BA1",
}

# All IDF setpoint schedule names the MPC will try to actuate.
IDF_SETPOINT_SCHEDULES_TO_ACTUATE = sorted(
    set(EP_SETPOINT_SCHEDULE_MAP.values()) | set(UNMANAGED_SETPOINT_SCHEDULE_MAP.values())
)

# Candidate schedule component types for schedule-value actuation.
SCHEDULE_COMPONENT_TYPES_TO_TRY = [
    "Schedule:Constant",
    "Schedule:Compact",
    "Schedule:File",
    "Schedule:Year",
    "Schedule:Week:Daily",
    "Schedule:Day:Hourly",
    "Schedule:Day:Interval",
    "Schedule:Day:List",
]

# Candidate EnergyPlus temperature and heating-output variables. The script will
# request all candidates and use the first handle available for each zone.
ZONE_TEMPERATURE_VARIABLE_CANDIDATES = [
    "Zone Mean Air Temperature",
    "Zone Air Temperature",
]

ZONE_HEATING_RATE_VARIABLE_CANDIDATES = [
    "Zone Air System Sensible Heating Rate",
    "Baseboard Electricity Rate",
    "Zone Baseboard Electricity Rate",
]

# Candidate EnergyPlus variables for exact per-zone / per-equipment interval
# heating energy. When these are unavailable, the script falls back to estimating
# zone energy from the reported heating rate:
#   E_zone[kWh] ~= rate_zone[W] * timestep_hours / 1000
ZONE_HEATING_ENERGY_VARIABLE_CANDIDATES = [
    "Baseboard Electricity Energy",
    "Zone Baseboard Electricity Energy",
    "Zone Air System Sensible Heating Energy",
]

# EnergyPlus meter used to update the billing state from the simulated plant.
# If Heating:Electricity does not exist in the IDF, the code falls back to
# Electricity:Facility only for logging.
HEATING_METER_CANDIDATES = [
    "Heating:Electricity",
    "Baseboard:Electricity",
    "Electricity:HVAC",
    "Electricity:Facility",
]

# Candidate Runtime API actuators for forcing the zone-air-temperature initial
# condition. Not every EnergyPlus model/version exposes these actuators. If none
# are available, the CSV temperatures are still used as the first MPC state, but
# the EnergyPlus plant itself will start from its normal warmup state.
ZONE_INITIAL_TEMPERATURE_ACTUATOR_CANDIDATES = [
    ("Zone Air Temperature", "Zone Mean Air Temperature"),
    ("Zone Air Temperature", "Zone Air Temperature"),
    ("Zone Air Heat Balance", "Zone Air Temperature"),
    ("Zone Air Heat Balance", "Zone Mean Air Temperature"),
]

INITIAL_TEMPERATURES_BY_DAY = None
CURRENT_INITIAL_ZONE_TEMPERATURES_DEGC = None

# Initial setpoints before the first MPC solve and during EnergyPlus warmup.
# This is used only to initialize the EnergyPlus plant and the MPC previous-setpoint state.
DEFAULT_INITIAL_SETPOINT_C = 18.0

# If the EnergyPlus model cannot provide instantaneous per-zone heating powers,
# the NODE/NCDE power states are initialized with this scaled value.
DEFAULT_INITIAL_POWER_SCALED = 0.0

# =============================================================================
# 1) Tariff / controller constants
# =============================================================================
FLEX_ACCESS_D_PER_DAY = jnp.float32(0.46154)
FLEX_OFF1_D_PER_KWH = jnp.float32(0.08699)
FLEX_OFF2_D_PER_KWH = jnp.float32(0.08699)
FLEX_PEAK_D_PER_KWH = jnp.float32(0.45088)

BILL_DAYS = jnp.float32(1.0)
OFFPEAK_BLOCK_CAP_KWH = jnp.float32(40.0) * BILL_DAYS
EOFF_CUM0_KWH = jnp.float32(0.0)

POWER_STATE_IS_KW = False

# Full-rollout logging settings.
# The normal mpc_prediction_run_*.csv files contain the model-predicted rollout.
# After EnergyPlus finishes, the script also creates
# mpc_prediction_vs_measurement_run_*.csv files where every rollout interval is
# matched to the corresponding measured EnergyPlus plant interval when available.
LOG_FULL_ROLLOUT_POWER_ENERGY_MEASUREMENTS = True
LOG_APPLIED_INTERVAL_POWER_ENERGY_IN_PLANT_LOG = True

N_POINTS = int(HORIZON_HOURS * 60 / STEP_MINUTES) + 1
T_steps = N_POINTS - 1

REOPT_MAXITER = 256

# The MPC solver is rebuilt at every 15-minute control step. Periodically
# clearing JAX compilation caches prevents long-run host/GPU memory growth.
# A value of 0 disables cleanup.
MPC_CACHE_CLEAR_INTERVAL_RUNS = 12
PRINT_MPC_MEMORY_USAGE = True
MIN_SETPOINT_C = 14.0
MAX_SETPOINT_C = 28.0
STEP_C = 0.5

# Fixed schedule outside the optimized preheat-search window.
T_LOW_NORMAL = jnp.float32(21.0)
T_LOW_DR = jnp.float32(18.0)
NIGHT_SETBACK_C = jnp.float32(18.0)
DR_EVENT_SETPOINT_C = jnp.float32(18.0)
GARAGE_LOWER_BOUND_C = jnp.float32(18.0)
GARAGE_PREFERRED_C = GARAGE_LOWER_BOUND_C

# Preheating parameterization. The optimizer may raise each non-garage zone
# above its baseline at any 15-minute interval in the two hours before DR.
# Consequently, both magnitude and duration are decision variables. The
# 24 degC / two-hour values only define the first iLQR candidate trajectory.
PREHEAT_SEARCH_WINDOW_HOURS = 2.0
RULE_BASED_PREHEAT_HOURS = 2.0
PREHEAT_MAX_SETPOINT_C = jnp.float32(24.0)
RULE_BASED_PREHEAT_SETPOINT_C = jnp.float32(24.0)
RULE_PREHEAT_OFF_RAW = jnp.float32(-3.5)
PREHEAT_FRACTION_EPS = jnp.float32(1.0e-4)

# Multi-start optimization and rule-based economic safeguard. The moderate raw
# value produces approximately 23 degC from a 21 degC daytime baseline.
ENABLE_MULTISTART = True
MULTISTART_MAXITER = 48
MULTISTART_ONLY_WHEN_PREHEAT_VISIBLE = True
MODERATE_PREHEAT_RAW = jnp.float32(0.7)
MODERATE_PREHEAT_HOURS = 1.0

# Fast adaptive search. A complete no-preheat/moderate refresh is forced on the
# first controllable solve and every two hours. Between refreshes, the
# shifted warm start is tried first and rescue starts run only if no evaluated
# candidate has yet cleared the rule-based savings threshold.
ADAPTIVE_MULTISTART = True
FULL_MULTISTART_REFRESH_MINUTES = 120
USE_LAX_SCAN_FOR_FIXED_ROLLOUTS = True

# Candidate feasibility tolerances. Economic cost is used for ranking only
# after these explicit predicted-temperature checks have passed.
MAX_NORMAL_COMFORT_SHORTFALL_C = 0.5
MAX_DR_FLOOR_SHORTFALL_C = 0.1
MAX_SAFETY_EXCESS_C = 0.0

# A feasible MPC candidate must beat the fixed rule trajectory by both the
# relative/absolute safeguard below; the larger required saving is used.
MIN_PREDICTED_SAVINGS_FRACTION = 0.005
MIN_PREDICTED_SAVINGS_DOLLARS = 0.005

# Temperature safeguards. There is deliberately no normal soft-upper penalty:
# outside the preheat window the thermostat is fixed to the baseline, while
# inside it the tariff determines whether additional heat is economically useful.
T_HIGH = jnp.float32(25.0)
T_SOFT_UPPER_NORMAL = jnp.float32(21.5)   # logging/reference only
T_SOFT_UPPER_NIGHT = jnp.float32(18.5)    # logging/reference only
T_SOFT_UPPER_PREHEAT = PREHEAT_MAX_SETPOINT_C
T_SOFT_UPPER_DR = T_HIGH
T_SOFT_UPPER_GARAGE = jnp.float32(18.0)

# Objective weights. The tariff term is expressed directly in dollars. Comfort
# and safety terms are high-value soft constraints, so economic optimization
# occurs primarily among trajectories that respect the thermal requirements.
W_ECONOMIC_COST = jnp.float32(100.0)
W_COMFORT_BELOW = jnp.float32(25.0)
W_TERMINAL_COMFORT_MULTIPLIER = jnp.float32(2.0)
W_HIGH_SAFETY = jnp.float32(0.0)    # Legacy, not used
W_PREHEAT_SETPOINT_MOVE = jnp.float32(0.02)
W_CONTROL_REGULARIZATION = jnp.float32(1.0e-5)

# DR-floor and monotonic-decay protection.
DR_ALLOWED_RISE_C_PER_H = jnp.float32(0.4)
W_DR_TEMPERATURE_RISE = jnp.float32(0.0)    # Legacy, not used
DR_FLOOR_C = jnp.float32(17.9)
DR_FLOOR_MARGIN_C = jnp.float32(0.1)
DR_FLOOR_SMOOTHING_C = jnp.float32(0.10)
W_DR_TIME_BELOW = jnp.float32(0.0)  # Legacy, not used
W_DR_DEPTH_BELOW = jnp.float32(100.0)
W_DR_EVENT_END = jnp.float32(300.0)

# Desired state at the instant the final DR interval ends. The lower side of
# the band remains strongly protected by W_DR_EVENT_END, the one-sided upper
# term discourages preheating that would leave unused heat after the event.
DR_EXIT_FLOOR_C = jnp.float32(18.0)
DR_EXIT_TARGET_C = jnp.float32(18.2)
W_DR_EVENT_END_EXCESS = jnp.float32(75.0)

# Soft declining upper envelope during the event. It starts at the maximum
# allowed preheat temperature and reaches DR_EXIT_TARGET_C at event end. This
# does not force a zone to remain hot; it only penalizes temperature above the
# envelope, with the penalty becoming more restrictive as the event proceeds.
DR_DECAY_ENVELOPE_START_C = PREHEAT_MAX_SETPOINT_C
W_DR_STORED_HEAT = jnp.float32(0.0) # Legacy, not used

# Historical temperature-residual correction. Residuals are defined as
# EnergyPlus minus model prediction and are estimated separately by zone and
# forecast lead. The expected correction guides comfort/exit economics, the
# lower residual quantile protects the 18 degC DR floor.
ENABLE_HISTORICAL_TEMPERATURE_BIAS_CORRECTION = True
BIAS_MIN_SAMPLES_PER_ZONE_LEAD = 8
BIAS_PRIOR_STRENGTH = 12.0
BIAS_MAX_ABS_CORRECTION_C = 1.5
BIAS_FLOOR_RESIDUAL_QUANTILE = 0.1
BIAS_MAX_PREVIOUS_DAYS = 10

# Populated once per simulated day from earlier EnergyPlus runs only.  Keeping
# these arrays at the fixed horizon shape avoids JAX recompilation as the
# amount of historical data grows.
CURRENT_EXPECTED_TEMPERATURE_BIAS_C = np.zeros(
    (N_POINTS, NUM_ZONES), dtype=np.float32
)
CURRENT_FLOOR_TEMPERATURE_BIAS_C = np.zeros(
    (N_POINTS, NUM_ZONES), dtype=np.float32
)
CURRENT_TEMPERATURE_BIAS_SAMPLE_COUNT = np.zeros(
    (N_POINTS, NUM_ZONES), dtype=np.int32
)

# Never select an optimized trajectory whose predicted tariff cost exceeds the
# fixed rule reference. Among cost-safe candidates, the complete objective
# decides between energy savings and unused heat at event exit.
MAX_PREDICTED_COST_INCREASE_VS_RULE_DOLLARS = 0.0

# Only the state-comfort reference is ramped after an event; the thermostat
# itself returns to the normal 21 degC schedule immediately. This prevents the
# optimizer from preheating merely because the building cannot physically jump
# from 18 to 21 degC in one 15-minute interval.
POST_DR_RECOVERY_REFERENCE_HOURS = 1.0

# DR windows are loaded from DR_EVENTS_CSV_PATH (dateDebut/dateFin).
NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 6
DR_EVENT_WINDOWS = None
DAILY_DR_WINDOWS = None      # kept only for backward-compatible function calls
DAILY_PEAK_WINDOWS = None    # peak windows are the same CSV DR-event windows

# 2) Path helpers and IDF patcher
def local_path(path_like):
    """Convert Windows C:\\... paths to /mnt/c/... when running under WSL/Linux."""
    if path_like is None:
        return None
    p = str(path_like)
    if os.name != "nt":
        m = re.match(r"^([A-Za-z]):\\(.*)$", p)
        if m:
            drive = m.group(1).lower()
            rest = m.group(2).replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
    return p


IDF_PATH = local_path(IDF_PATH)
WEATHER_EPW_PATH = local_path(WEATHER_EPW_PATH)
DR_EVENTS_CSV_PATH = local_path(DR_EVENTS_CSV_PATH)
INITIAL_ZONE_TEMPERATURES_CSV_PATH = local_path(INITIAL_ZONE_TEMPERATURES_CSV_PATH)
SIMULATED_DATA_CSV_PATH = local_path(SIMULATED_DATA_CSV_PATH)
MODEL_PATH = local_path(MODEL_PATH)
LOG_DIR = local_path(LOG_DIR)
LEGACY_BIAS_LOG_DIR = local_path(LEGACY_BIAS_LOG_DIR)
BASE_LOG_DIR = LOG_DIR
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def find_first_epw_next_to_idf(idf_path):
    idf_dir = Path(idf_path).parent
    epws = sorted(idf_dir.glob("*.epw"))
    return str(epws[0]) if epws else None


def day_name(ts):
    return pd.Timestamp(ts).day_name()


def split_idf_objects_no_comments(idf_text):
    """Return IDF objects as raw comma-separated strings with comments removed."""
    no_comments = []
    for line in idf_text.splitlines():
        no_comments.append(line.split("!")[0])
    text_nc = "\n".join(no_comments)

    objects = []
    for raw_obj in text_nc.split(";"):
        obj = raw_obj.strip()
        if obj:
            objects.append(obj + ";")
    return objects


def idf_object_type(obj_text):
    return obj_text.split(",", 1)[0].strip()


def idf_object_name(obj_text):
    parts = obj_text.split(",")
    return parts[1].strip() if len(parts) > 1 else ""


def remove_idf_object_types(idf_text, object_types_to_remove):
    """
    Simple IDF object filter. It strips comments and removes complete objects
    whose first token matches one of object_types_to_remove.
    """
    object_types_to_remove = {x.lower() for x in object_types_to_remove}
    kept = []
    for obj in split_idf_objects_no_comments(idf_text):
        if idf_object_type(obj).lower() in object_types_to_remove:
            continue
        kept.append(obj)
    return "\n\n".join(kept) + "\n"


def remove_idf_named_objects(idf_text, names_to_remove, object_types=None):
    """Remove objects by object name, optionally restricted to object types."""
    names_to_remove = {x.lower() for x in names_to_remove}
    object_types_l = None if object_types is None else {x.lower() for x in object_types}
    kept = []
    for obj in split_idf_objects_no_comments(idf_text):
        typ = idf_object_type(obj).lower()
        name = idf_object_name(obj).lower()
        if name in names_to_remove and (object_types_l is None or typ in object_types_l):
            continue
        kept.append(obj)
    return "\n\n".join(kept) + "\n"


def make_schedule_constant_objects(schedule_names, value_c=DEFAULT_INITIAL_SETPOINT_C):
    blocks = []
    for schedule_name in schedule_names:
        blocks.append(
            f"""
Schedule:Constant,
  {schedule_name},
  Temperature,
  {float(value_c):.3f};
""".strip()
        )
    return "\n\n".join(blocks) + "\n"


def find_schedule_file_names(idf_text):
    """Return external file names referenced by remaining Schedule:File objects."""
    files = []
    for obj in split_idf_objects_no_comments(idf_text):
        if idf_object_type(obj).lower() != "schedule:file":
            continue
        parts = [p.strip() for p in obj.rstrip(";").split(",")]
        # Schedule:File fields: type, name, schedule type limits, file name, column,
        if len(parts) >= 4 and parts[3]:
            files.append(parts[3])
    return sorted(set(files))


def copy_remaining_schedule_files(idf_text, original_idf_path, work_dir):
    """
    Copy external Schedule:File dependencies beside the working IDF. This
    IDF still needs FoundationsBD2013.txt after the thermostat setpoint schedules
    are converted to Schedule:Constant.
    """
    original_dir = Path(original_idf_path).parent
    work_dir = Path(work_dir)
    copied = []
    missing = []
    for fname in find_schedule_file_names(idf_text):
        src = original_dir / fname
        dst = work_dir / Path(fname).name
        if src.exists():
            if src.resolve() != dst.resolve():
                dst.write_bytes(src.read_bytes())
            copied.append(str(dst))
        else:
            missing.append(str(src))
    if copied:
        print("Copied Schedule:File dependencies:")
        for p in copied:
            print(f"  {p}")
    if missing:
        print("WARNING: missing Schedule:File dependencies. EnergyPlus may fail unless these files are available:")
        for p in missing:
            print(f"  {p}")


def make_energyplus_working_idf(idf_path, out_dir, sim_start, sim_end):
    """
    Creates a working IDF copy with minimal modifications:
      - Timestep = 4, matching 15-min MPC steps,
      - one weather-file RunPeriod matching SIM_START_DT/SIM_END_DT,
      - requested output variables/meters for the Runtime API,
      - thermostat setpoint schedules initialized as 18 degC constants.

    The 18 degC constants are used so EnergyPlus warmup starts the plant from an
    18 degC thermostat condition instead of the original schedule files. After
    warmup, the MPC still actuates these schedule values and has priority over
    the preferred comfort profile.
    """
    idf_path = Path(idf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = idf_path.read_text(encoding="utf-8", errors="ignore")

    # Keep the original model intact except for timestep and RunPeriod.
    patched = remove_idf_object_types(raw, {"Timestep", "RunPeriod"})

    # Replace the thermostat setpoint schedules by 18 degC Schedule:Constant
    # objects before warmup. This prevents the original Schedule:File objects
    # from warming the house near 21 degC before the runtime MPC callbacks start.
    patched = remove_idf_named_objects(
        patched,
        IDF_SETPOINT_SCHEDULES_TO_ACTUATE,
        object_types=SCHEDULE_COMPONENT_TYPES_TO_TRY,
    )

    # EnergyPlus RunPeriod end day is inclusive. SIM_END_DT is exclusive here.
    end_inclusive = pd.Timestamp(sim_end) - pd.Timedelta(minutes=STEP_MINUTES)

    runperiod_and_outputs = f"""
Timestep,
  {int(60 / STEP_MINUTES)};

RunPeriod,
  MPC_Historical_Run,
  {int(pd.Timestamp(sim_start).month)},
  {int(pd.Timestamp(sim_start).day)},
  {int(pd.Timestamp(sim_start).year)},
  {int(end_inclusive.month)},
  {int(end_inclusive.day)},
  {int(end_inclusive.year)},
  {day_name(sim_start)},
  No,
  No,
  No,
  Yes,
  Yes,
  Yes;

Output:Variable,
  *,
  Zone Mean Air Temperature,
  Timestep;

Output:Variable,
  *,
  Zone Air Temperature,
  Timestep;

Output:Variable,
  *,
  Zone Thermostat Heating Setpoint Temperature,
  Timestep;

Output:Variable,
  *,
  Zone Air System Sensible Heating Rate,
  Timestep;

Output:Variable,
  *,
  Baseboard Electricity Rate,
  Timestep;

Output:Variable,
  *,
  Baseboard Electricity Energy,
  Timestep;

Output:Variable,
  *,
  Zone Baseboard Electricity Energy,
  Timestep;

Output:Variable,
  *,
  Zone Air System Sensible Heating Energy,
  Timestep;

Output:Meter,
  Heating:Electricity,
  Timestep;

Output:Meter,
  Baseboard:Electricity,
  Timestep;

Output:Meter,
  Electricity:HVAC,
  Timestep;

Output:Meter,
  Electricity:Facility,
  Timestep;
"""

    initial_schedule_block = make_schedule_constant_objects(
        IDF_SETPOINT_SCHEDULES_TO_ACTUATE,
        value_c=DEFAULT_INITIAL_SETPOINT_C,
    )

    final_text = patched + "\n\n" + initial_schedule_block + "\n\n" + runperiod_and_outputs
    out_path = out_dir / f"{idf_path.stem}_mpc_working_initial18_schedules.idf"
    out_path.write_text(final_text, encoding="utf-8")

    # Copy any remaining Schedule:File dependencies, excluding the thermostat
    # schedule files that were replaced by Schedule:Constant objects above.
    copy_remaining_schedule_files(final_text, idf_path, out_dir)

    print(f"Working IDF written to: {out_path}")
    print(f"Thermostat schedules initialized to {DEFAULT_INITIAL_SETPOINT_C:.1f} degC for EnergyPlus warmup.")
    print("Schedules to actuate:", ", ".join(IDF_SETPOINT_SCHEDULES_TO_ACTUATE))
    return str(out_path)


# 3) Load trained NODE/NCDE model and scalers
ts = jnp.linspace(0.0, float(HORIZON_HOURS), N_POINTS, dtype=jnp.float32)
dt = ts[1] - ts[0]  # hours

ts_dummy = jnp.array([0.0, 1.0], dtype=jnp.float32)
ctrl_dummy = jnp.zeros((2, NUM_ZONES, 2), dtype=jnp.float32)
control_interp_skel = diffrax.LinearInterpolation(ts_dummy, ctrl_dummy)

key = jr.PRNGKey(0)
skeleton = NeuralODE(
    in_size=1,
    out_size=1,
    hidden_sizes=None,
    data_size=2,
    masks=[],
    control_interp=control_interp_skel,
    mmlp_flag=False,
    num_zones=NUM_ZONES,
    key=key,
)

model = eqx.tree_deserialise_leaves(MODEL_PATH, skeleton)
func = model.func
n = func.n_zones
kz = func.kz
k_total = func.latent_dim
aux_dim = func.aux_dim

# Main ODE state: [T, P, z]
y_main_dim = 2 * n + k_total

# Augmented ODE state: [T, P, z, I_P]
I_dim = n
y_aug_dim = y_main_dim + I_dim

EOFF_STATE_INDEX = y_aug_dim + n
ECONOMIC_COST_STATE_INDEX = y_aug_dim + n + 1
T_prev_state_start = y_aug_dim + n + 2
x_dim = T_prev_state_start + n

(
    (time_train_float, time_val_float),
    (_, _),
    (x_train, x_val),
    (u_train, u_val),
    (x_scaler, u_scaler),
    (_, _),
    control_interp,
    data_size,
) = get_data(time_horizon=HORIZON_HOURS, stride=1, num_zones=NUM_ZONES)

T_min = jnp.asarray(x_scaler.data_min_[:n], dtype=jnp.float32)
T_rng = jnp.asarray(x_scaler.data_range_[:n], dtype=jnp.float32) + 1e-6

P_min = jnp.asarray(x_scaler.data_min_[n:2 * n], dtype=jnp.float32)
P_rng = jnp.asarray(x_scaler.data_range_[n:2 * n], dtype=jnp.float32) + 1e-6

sp_min = jnp.asarray(u_scaler.data_min_[:n], dtype=jnp.float32)
sp_rng = jnp.asarray(u_scaler.data_range_[:n], dtype=jnp.float32) + 1e-6


# 4) Model RHS and scaling helpers
def func_rhs_with_dynamic_dTSet(func, t, y, args, dTSet_vec):
    ts_local, interpolator_us, interpolator_aux, _ = args

    n_local = func.n_zones
    kz_local = func.kz
    ts0 = ts_local[0]

    aux = jnp.ravel(interpolator_aux.evaluate(t))
    uall = jnp.ravel(interpolator_us.evaluate(t))
    u_sp = uall[:n_local]

    y_base = y[:n_local]
    y_p = y[n_local:2 * n_local]
    z = y[2 * n_local:2 * n_local + func.latent_dim]
    z_mat = z.reshape(n_local, kz_local)

    node_in_raw = func.node_in_raw_buf.astype(y.dtype)
    node_in_raw = node_in_raw.at[:, :n_local].set(y_base[None, :])
    node_in_raw = node_in_raw.at[:, n_local:n_local + kz_local].set(z_mat)
    node_in_raw = node_in_raw.at[:, n_local + kz_local].set(y_p)
    node_in_raw = node_in_raw.at[:, n_local + kz_local + 1:n_local + kz_local + 1 + aux_dim].set(aux[None, :])

    node_out = eqx.filter_vmap(lambda m, x: m(x))(func.mmlp_NODE, node_in_raw)
    node_out = jnp.ravel(node_out)

    z_in_raw = func.z_in_raw_buf.astype(y.dtype)
    z_in_raw = z_in_raw.at[:, :n_local].set(y_base[None, :])
    z_in_raw = z_in_raw.at[:, n_local].set(y_p)
    z_in_raw = z_in_raw.at[:, n_local + 1:n_local + 1 + kz_local].set(z_mat)
    z_in_raw = z_in_raw.at[:, n_local + 1 + kz_local:n_local + 1 + kz_local + aux_dim].set(aux[None, :])

    dz_mat = eqx.filter_vmap(lambda m, x: m(x))(func.z_mlp, z_in_raw)
    dz = dz_mat.reshape(-1)

    dt_vec = jnp.ones((n_local,), dtype=y.dtype)
    dTSet = dTSet_vec.astype(y.dtype)

    d_aux = interpolator_aux.derivative(t)
    dT_ext = d_aux[0]

    ncde_in_raw = func.ncde_in_raw_buf.astype(y.dtype)
    ncde_in_raw = ncde_in_raw.at[:, 0].set((t - ts0) * func.ones_z.astype(y.dtype))
    ncde_in_raw = ncde_in_raw.at[:, 1].set(y_base)
    ncde_in_raw = ncde_in_raw.at[:, 2].set(u_sp)
    ncde_in_raw = ncde_in_raw.at[:, 3].set(u_sp - y_base)
    ncde_in_raw = ncde_in_raw.at[:, 4].set(node_out)
    ncde_in_raw = ncde_in_raw.at[:, 5].set(dTSet)
    ncde_in_raw = ncde_in_raw.at[:, 6].set(y_p)
    ncde_in_raw = ncde_in_raw.at[:, 7:7 + aux_dim].set(aux[None, :])

    ncde_vec = eqx.filter_vmap(lambda m, x: m(x))(func.mmlp_NCDE, ncde_in_raw)

    dX = func.dX_buf.astype(y.dtype)
    dX = dX.at[:, 0].set(dt_vec)
    dX = dX.at[:, 1].set(dTSet)
    dX = dX.at[:, 2].set(dT_ext * func.ones_z.astype(y.dtype))

    ncde_out = jnp.einsum("bi,bi->b", ncde_vec, dX)

    dT = node_out
    dP = ncde_out

    P = y_p
    x = jnp.clip((1.0 - P) / 0.02, 0.0, 1.0)
    up_room = x * x * (3.0 - 2.0 * x)
    x = jnp.clip((P - 0.0) / 0.02, 0.0, 1.0)
    down_room = x * x * (3.0 - 2.0 * x)
    scale = jnp.where(dP >= 0.0, up_room, down_room)
    dP = dP * scale

    return jnp.concatenate([dT, dP, dz], axis=0)


def T_scaled_to_deg(Ts):
    return Ts * T_rng + T_min


def P_scaled_to_physical(Ps):
    return Ps * P_rng + P_min


def P_scaled_to_kW(Ps):
    P_phys = P_scaled_to_physical(Ps)
    return P_phys if POWER_STATE_IS_KW else (P_phys / 1000.0)


def sp_scaled_to_deg(sp_s):
    return sp_s * sp_rng + sp_min


def sp_deg_to_scaled(sp_d):
    return (sp_d - sp_min) / sp_rng


def quantize_deg_ste(x_deg, step=STEP_C):
    q = step * jnp.round(x_deg / step)
    return x_deg + jax.lax.stop_gradient(q - x_deg)


def sp_quantized_from_u_raw(u_raw):
    sp_cont_scaled = jax.nn.sigmoid(u_raw)
    sp_cont_deg = sp_scaled_to_deg(sp_cont_scaled)
    sp_q_deg = quantize_deg_ste(sp_cont_deg, STEP_C)
    sp_q_deg = jnp.clip(sp_q_deg, MIN_SETPOINT_C, MAX_SETPOINT_C)
    sp_q_scaled = jnp.clip(sp_deg_to_scaled(sp_q_deg), 0.0, 1.0)
    return sp_q_scaled, sp_q_deg


def scale_T_deg(T0_degC, clip_scaled=True):
    T0_degC = np.asarray(T0_degC, dtype=np.float32)
    T_min_np = np.asarray(x_scaler.data_min_[:n], dtype=np.float32)
    T_rng_np = np.asarray(x_scaler.data_range_[:n], dtype=np.float32) + 1e-6
    out = (T0_degC - T_min_np) / T_rng_np
    return np.clip(out, 0.0, 1.0) if clip_scaled else out


def scale_sp_deg(sp_deg, clip_scaled=True):
    sp_deg = np.asarray(sp_deg, dtype=np.float32)
    sp_min_np = np.asarray(u_scaler.data_min_[:n], dtype=np.float32)
    sp_rng_np = np.asarray(u_scaler.data_range_[:n], dtype=np.float32) + 1e-6
    out = (sp_deg - sp_min_np) / sp_rng_np
    return np.clip(out, 0.0, 1.0) if clip_scaled else out


# 5) Time, comfort, and tariff helpers
def load_dr_event_windows(csv_path=DR_EVENTS_CSV_PATH, timezone=OPEN_METEO_TIMEZONE):
    """Load dateDebut/dateFin DR events from CSV as timezone-aware local timestamps."""
    csv_path = Path(local_path(csv_path))
    if not csv_path.exists():
        # Portable fallback when this script and the CSV are placed in the same folder.
        script_side_path = Path(__file__).resolve().parent / csv_path.name
        if script_side_path.exists():
            csv_path = script_side_path
    if not csv_path.exists():
        raise FileNotFoundError(f"DR-event CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"dateDebut", "dateFin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DR-event CSV must contain columns {sorted(required)}. Missing: {sorted(missing)}")

    starts = pd.to_datetime(df["dateDebut"], utc=True).dt.tz_convert(timezone)
    ends = pd.to_datetime(df["dateFin"], utc=True).dt.tz_convert(timezone)

    windows = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        if end <= start:
            raise ValueError(f"Invalid DR event at row {i}: dateFin must be after dateDebut")
        windows.append((pd.Timestamp(start), pd.Timestamp(end)))
    return windows


def get_dr_event_windows():
    """Lazy loader so the file is read after paths have been converted for WSL."""
    global DR_EVENT_WINDOWS
    if DR_EVENT_WINDOWS is None:
        DR_EVENT_WINDOWS = load_dr_event_windows(DR_EVENTS_CSV_PATH, OPEN_METEO_TIMEZONE)
    return DR_EVENT_WINDOWS


def unique_dr_days(event_windows=None):
    """Return each local calendar day touched by at least one DR event."""
    if event_windows is None:
        event_windows = get_dr_event_windows()
    days = set()
    for start, end in event_windows:
        first_day = start.normalize()
        last_day = (end - pd.Timedelta(nanoseconds=1)).normalize()
        day = first_day
        while day <= last_day:
            days.add(day)
            day += pd.Timedelta(days=1)
    return sorted(days)


def _normalize_day_key(dt_obj):
    """Normalize a timestamp/date to YYYY-MM-DD in the local DR timezone."""
    ts_obj = pd.Timestamp(dt_obj)
    if ts_obj.tzinfo is None:
        ts_obj = ts_obj.tz_localize(OPEN_METEO_TIMEZONE)
    else:
        ts_obj = ts_obj.tz_convert(OPEN_METEO_TIMEZONE)
    return ts_obj.date().isoformat()


def _candidate_initial_temperature_columns(zone_name):
    """Return likely column names for one zone in the initial-temperature CSV."""
    safe_zone = zone_name.lower().replace(" ", "_")
    compact_zone = safe_zone.replace("_", "")
    ep_zone = EP_ZONE_NAMES.get(zone_name, zone_name)
    safe_ep_zone = ep_zone.lower().replace(" ", "_")

    return [
        f"T_init_{safe_zone}_degC",
        f"T_initial_{safe_zone}_degC",
        f"T0_{safe_zone}_degC",
        f"T_ep_{safe_zone}_degC",
        f"T_{safe_zone}_degC",
        f"initial_{safe_zone}_degC",
        f"{safe_zone}_degC",
        safe_zone,
        compact_zone,
        zone_name,
        f"T_init_{safe_ep_zone}_degC",
        f"T_ep_{safe_ep_zone}_degC",
        f"{safe_ep_zone}_degC",
        ep_zone,
    ]


def _infer_initial_temperature_day_column(df):
    """Find the column that identifies the DR calendar day."""
    preferred = [
        "dr_day",
        "DR_day",
        "day",
        "date",
        "date_local",
        "dr_date",
        "event_day",
        "initial_day",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    timestamp_like = [
        "initial_time_local",
        "simulation_time_local",
        "time_local",
        "timestamp",
        "time",
        "control_time_local",
    ]
    for col in timestamp_like:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find the DR-day/date column in the initial-temperature CSV. "
        "Expected one of: "
        + ", ".join(preferred + timestamp_like)
    )


def load_initial_zone_temperatures_by_day(csv_path=INITIAL_ZONE_TEMPERATURES_CSV_PATH):
    """
    Load the no-MPC initial-temperature CSV.

    Expected wide format:
      - one row per DR day,
      - a day/date column such as dr_day,
      - zone columns such as T_init_garage_degC, T_init_base_1_degC, ...

    The parser also accepts a few fallback column names so the code is robust to
    slightly different versions of the CSV generator.
    """
    csv_path = Path(local_path(csv_path))
    if not csv_path.exists():
        script_side_path = Path(__file__).resolve().parent / csv_path.name
        if script_side_path.exists():
            csv_path = script_side_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Initial-zone-temperature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Initial-zone-temperature CSV is empty: {csv_path}")

    day_col = _infer_initial_temperature_day_column(df)
    out = {}

    for row_idx, row in df.iterrows():
        day_key = _normalize_day_key(row[day_col])
        if day_key in out:
            raise ValueError(
                f"Duplicate initial-temperature rows for DR day {day_key} in {csv_path}."
            )

        source_initial_time = None
        if "initial_time_local" in df.columns and pd.notna(row["initial_time_local"]):
            source_initial_time = _as_local_timestamp(row["initial_time_local"])
            expected_midnight = pd.Timestamp(day_key).tz_localize(OPEN_METEO_TIMEZONE)
            if abs(source_initial_time - expected_midnight) > pd.Timedelta(seconds=1):
                raise ValueError(
                    "The initial-temperature CSV row is not a 00:00 DR-day state: "
                    f"day={day_key}, initial_time_local={source_initial_time.isoformat()}."
                )

        temps = []
        missing = []
        used_cols = {}
        for zone_name in ZONE_ORDER:
            value = None
            used_col = None
            for col in _candidate_initial_temperature_columns(zone_name):
                if col in df.columns and pd.notna(row[col]):
                    value = float(row[col])
                    used_col = col
                    break
            if value is None:
                missing.append(zone_name)
            else:
                if not np.isfinite(value) or not (-50.0 <= value <= 60.0):
                    raise ValueError(
                        f"Invalid initial temperature for {zone_name} on {day_key}: {value}."
                    )
                temps.append(value)
                used_cols[zone_name] = used_col

        if missing:
            raise ValueError(
                f"Missing initial temperature columns for DR day {day_key}: {missing}. "
                f"Available columns are: {list(df.columns)}"
            )

        out[day_key] = {
            "T_init_degC": np.asarray(temps, dtype=np.float32),
            "csv_path": str(csv_path),
            "row_index": int(row_idx),
            "day_column": day_col,
            "used_columns": used_cols,
            "initial_time_local": (
                source_initial_time.isoformat()
                if source_initial_time is not None
                else None
            ),
        }

    print(f"Loaded initial zone temperatures for {len(out)} DR days from: {csv_path}")
    return out


def get_initial_zone_temperatures_for_day(day_start):
    """Return the CSV initial temperature vector for the selected DR day."""
    if not USE_CSV_INITIAL_ZONE_TEMPERATURES:
        return None, None

    global INITIAL_TEMPERATURES_BY_DAY
    if INITIAL_TEMPERATURES_BY_DAY is None:
        INITIAL_TEMPERATURES_BY_DAY = load_initial_zone_temperatures_by_day(
            INITIAL_ZONE_TEMPERATURES_CSV_PATH
        )

    day_key = _normalize_day_key(day_start)
    if day_key not in INITIAL_TEMPERATURES_BY_DAY:
        available = sorted(INITIAL_TEMPERATURES_BY_DAY.keys())
        raise KeyError(
            f"No initial zone temperatures found for DR day {day_key}. "
            f"Available DR days in the CSV are: {available}"
        )

    record = INITIAL_TEMPERATURES_BY_DAY[day_key]
    return record["T_init_degC"].copy(), record


def save_selected_initial_temperature_log(log_dir, day_start, T_init_degC, record):
    """Save the chosen CSV initial temperatures in the run output folder."""
    if T_init_degC is None:
        return None

    run_dir = Path(log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    safe_day = _normalize_day_key(day_start)
    wide_row = {
        "dr_day": safe_day,
        "initial_time_local": pd.Timestamp(day_start).tz_convert(OPEN_METEO_TIMEZONE).isoformat(),
        "source_csv": record.get("csv_path") if isinstance(record, dict) else None,
        "source_row_index": record.get("row_index") if isinstance(record, dict) else None,
    }
    for i, zone_name in enumerate(ZONE_ORDER):
        safe_zone = zone_name.lower().replace(" ", "_")
        wide_row[f"T_init_{safe_zone}_degC"] = float(T_init_degC[i])

    wide_path = run_dir / "selected_initial_zone_temperatures_wide.csv"
    pd.DataFrame([wide_row]).to_csv(wide_path, index=False)

    long_rows = []
    for i, zone_name in enumerate(ZONE_ORDER):
        safe_zone = zone_name.lower().replace(" ", "_")
        long_rows.append({
            "dr_day": safe_day,
            "initial_time_local": pd.Timestamp(day_start).tz_convert(OPEN_METEO_TIMEZONE).isoformat(),
            "zone_index": int(i),
            "zone_name": zone_name,
            "safe_zone": safe_zone,
            "T_init_degC": float(T_init_degC[i]),
            "source_csv": record.get("csv_path") if isinstance(record, dict) else None,
            "source_row_index": record.get("row_index") if isinstance(record, dict) else None,
        })

    long_path = run_dir / "selected_initial_zone_temperatures_long.csv"
    pd.DataFrame(long_rows).to_csv(long_path, index=False)

    print(f"Selected initial-temperature CSV row saved to: {wide_path}")
    return str(wide_path)


def _as_local_timestamp(dt_obj):
    ts_obj = pd.Timestamp(dt_obj)
    if ts_obj.tzinfo is None:
        return ts_obj.tz_localize(OPEN_METEO_TIMEZONE)
    return ts_obj.tz_convert(OPEN_METEO_TIMEZONE)


def is_in_csv_dr_event(dt_obj):
    ts_obj = _as_local_timestamp(dt_obj)
    return any(start <= ts_obj < end for start, end in get_dr_event_windows())


def is_dr_event_end(dt_obj):
    """Return True when a horizon node coincides with a CSV DR-event end.

    DR intervals use the half-open convention ``start <= t < end``. Therefore,
    the event-end timestamp is not itself an in-event node and needs a separate
    flag for the end-of-event terminal/floor penalty.
    """
    ts_obj = _as_local_timestamp(dt_obj)

    # Horizon nodes are expected to be aligned to the 15-minute control grid.
    # A small tolerance protects against timestamp representation differences
    # without marking either neighboring control-grid node.
    tolerance = pd.Timedelta(seconds=max(1.0, STEP_MINUTES * 60.0 * 0.1))

    return any(
        abs(ts_obj - _as_local_timestamp(end)) <= tolerance
        for _, end in get_dr_event_windows()
    )


def dr_event_progress_fraction(dt_obj):
    """Return event progress on [0, 1], with zero outside a DR event.

    The exact event-end node is assigned one so prediction logs show the end of
    the decay envelope even though the half-open DR flag is already false.
    """
    ts_obj = _as_local_timestamp(dt_obj)
    for start, end in get_dr_event_windows():
        start = _as_local_timestamp(start)
        end = _as_local_timestamp(end)
        if start <= ts_obj < end:
            duration_seconds = max((end - start).total_seconds(), 1.0)
            elapsed_seconds = (ts_obj - start).total_seconds()
            return float(np.clip(elapsed_seconds / duration_seconds, 0.0, 1.0))
        if is_dr_event_end(ts_obj) and abs(ts_obj - end) <= pd.Timedelta(
            seconds=max(1.0, STEP_MINUTES * 60.0 * 0.1)
        ):
            return 1.0
    return 0.0


def hours_until_next_dr_start(dt_obj):
    """Return hours until the next future DR-event start, or +inf.

    A DR event already in progress is handled separately by
    ``is_in_csv_dr_event`` and is therefore not returned as a future start.
    """
    ts_obj = _as_local_timestamp(dt_obj)
    future_hours = [
        (start - ts_obj).total_seconds() / 3600.0
        for start, _ in get_dr_event_windows()
        if start > ts_obj
    ]
    return min(future_hours) if future_hours else np.inf


def hours_since_most_recent_dr_end(dt_obj):
    """Return elapsed hours since the latest completed DR event."""
    ts_obj = _as_local_timestamp(dt_obj)
    if is_in_csv_dr_event(ts_obj):
        return np.inf
    elapsed_hours = [
        (ts_obj - _as_local_timestamp(end)).total_seconds() / 3600.0
        for _, end in get_dr_event_windows()
        if _as_local_timestamp(end) <= ts_obj
    ]
    return min(elapsed_hours) if elapsed_hours else np.inf


def is_preheat_window(dt_obj):
    """Return True inside the candidate window where preheating is optimized."""
    hours_to_event = hours_until_next_dr_start(dt_obj)
    return 0.0 < hours_to_event <= float(PREHEAT_SEARCH_WINDOW_HOURS)


def is_rule_based_preheat_seed_window(dt_obj):
    """Return True in the 24 degC/two-hour trajectory used to initialize iLQR."""
    hours_to_event = hours_until_next_dr_start(dt_obj)
    return 0.0 < hours_to_event <= float(RULE_BASED_PREHEAT_HOURS)


def is_night_setback(dt_obj):
    ts_obj = _as_local_timestamp(dt_obj)
    return ts_obj.hour >= NIGHT_START_HOUR or ts_obj.hour < NIGHT_END_HOUR


def is_in_any_window(dt_obj, windows=None):
    """
    Backward-compatible wrapper.
    The old script used daily clock windows. This version uses the exact CSV
    DR-event timestamps for both DR and peak-event checks.
    """
    return is_in_csv_dr_event(dt_obj)


def preferred_temperature_at_datetime(dt_obj):
    """
    Baseline lower comfort bound for non-garage zones.

    The economic controller uses the requested fixed schedule outside the
    candidate preheat window: 18 degC during DR and night setback, 21 degC
    otherwise. No post-event ramp is imposed; normal/night operation resumes
    immediately after an event.
    """
    ts_obj = _as_local_timestamp(dt_obj)

    if is_in_csv_dr_event(ts_obj):
        return float(T_LOW_DR)

    if is_night_setback(ts_obj):
        return float(NIGHT_SETBACK_C)

    return float(T_LOW_NORMAL)


def preferred_temperature_vector_at_datetime(dt_obj):
    """
    Per-zone lower comfort bound used by the MPC objective.

    The garage lower comfort bound is always 18 degC. Other zones use the
    normal/night/DR baseline returned by preferred_temperature_at_datetime().
    """
    base_pref = preferred_temperature_at_datetime(dt_obj)
    pref = np.ones((NUM_ZONES,), dtype=np.float32) * float(base_pref)
    garage_idx = ZONE_ORDER.index("Garage")
    pref[garage_idx] = float(GARAGE_LOWER_BOUND_C)
    return pref


def matched_preconditioning_setpoint_vector_at_datetime(dt_obj):
    """Reproduce the no-MPC rule schedule during the preconditioning day.

    Priority matches ``modified_energyplus_dr_night_setback_preDR24_18.py``:
      1) 18 degC during DR;
      2) 24 degC during the final two hours before DR;
      3) 18 degC during night setback;
      4) 21 degC otherwise.

    The garage remains at 18 degC in every case.
    """
    if is_in_csv_dr_event(dt_obj):
        default_sp = float(DR_EVENT_SETPOINT_C)
    elif is_rule_based_preheat_seed_window(dt_obj):
        default_sp = float(PREHEAT_MAX_SETPOINT_C)
    elif is_night_setback(dt_obj):
        default_sp = float(NIGHT_SETBACK_C)
    else:
        default_sp = float(T_LOW_NORMAL)

    sp = np.full((NUM_ZONES,), default_sp, dtype=np.float32)
    garage_idx = ZONE_ORDER.index("Garage")
    sp[garage_idx] = float(GARAGE_LOWER_BOUND_C)
    return sp


def upper_comfort_bound_vector_at_datetime(dt_obj):
    """Per-zone heavily penalized upper safety ceiling."""
    return np.ones((NUM_ZONES,), dtype=np.float32) * float(T_HIGH)


def soft_upper_temperature_vector_at_datetime(dt_obj):
    """Return reference upper values for logging; these are not cost terms."""
    if is_in_csv_dr_event(dt_obj):
        soft_upper = float(T_SOFT_UPPER_DR)
    elif is_preheat_window(dt_obj):
        soft_upper = float(T_SOFT_UPPER_PREHEAT)
    elif is_night_setback(dt_obj):
        soft_upper = float(T_SOFT_UPPER_NIGHT)
    else:
        soft_upper = float(T_SOFT_UPPER_NORMAL)

    values = np.ones((NUM_ZONES,), dtype=np.float32) * soft_upper
    garage_idx = ZONE_ORDER.index("Garage")
    values[garage_idx] = float(T_SOFT_UPPER_GARAGE)
    return values


def excess_temperature_weight_at_datetime(dt_obj):
    """Compatibility field for logs; the economic objective does not use it."""
    del dt_obj
    return 0.0


def T_low_at_datetime(dt_obj):
    """
    Backward-compatible scalar lower comfort bound for non-garage zones.

    build_time_profiles() uses preferred_temperature_vector_at_datetime() so
    the MPC objective receives a per-zone lower-bound vector with the garage
    fixed at 18 degC.
    """
    return preferred_temperature_at_datetime(dt_obj)

def setpoint_for_interval_from_u_raw(
    u_raw,
    t_idx,
    profiles,
):
    """Map free controls to the economic schedule used for one interval.

    Outside the candidate preheat window, the setpoint is fixed to the baseline
    schedule. During DR it is fixed to 18 degC. Only inside the candidate
    preheat window does ``u_raw`` select a value between the zone baseline and
    24 degC. This interval-wise choice makes duration and magnitude endogenous.
    """
    u_raw = jnp.asarray(u_raw, dtype=jnp.float32)
    is_dr = profiles["is_dr_nodes"][t_idx]
    is_preheat = profiles["is_preheat_nodes"][t_idx]
    baseline_deg = profiles["baseline_setpoint_t"][t_idx]
    garage_idx = ZONE_ORDER.index("Garage")

    preheat_max_deg = jnp.ones((n,), dtype=u_raw.dtype) * PREHEAT_MAX_SETPOINT_C
    preheat_max_deg = preheat_max_deg.at[garage_idx].set(GARAGE_LOWER_BOUND_C)

    preheat_fraction = jax.nn.sigmoid(u_raw)
    optimized_preheat_deg = baseline_deg + preheat_fraction * (
        preheat_max_deg - baseline_deg
    )
    optimized_preheat_deg = quantize_deg_ste(optimized_preheat_deg, STEP_C)
    optimized_preheat_deg = jnp.clip(
        optimized_preheat_deg,
        baseline_deg,
        preheat_max_deg,
    )

    dr_setpoint_deg = jnp.ones((n,), dtype=u_raw.dtype) * DR_EVENT_SETPOINT_C

    sp_deg = jnp.where(
        is_dr,
        dr_setpoint_deg,
        jnp.where(is_preheat, optimized_preheat_deg, baseline_deg),
    )

    # The garage is never used as a thermal-storage zone.
    sp_deg = sp_deg.at[garage_idx].set(GARAGE_LOWER_BOUND_C)

    sp_scaled = jnp.clip(
        sp_deg_to_scaled(sp_deg),
        0.0,
        1.0,
    )

    return sp_scaled, sp_deg


def raw_control_for_preheat_target(target_setpoint_deg, baseline_deg):
    """Map a physical preheat target to the unconstrained iLQR coordinate.

    The inverse logistic mapping is evaluated for every interval and room so
    the fixed RBC reference is 24 degC regardless of whether the baseline is
    18 or 21 degC. With a 24 degC optimization cap the result is within the
    configured numerical epsilon of 24 degC.
    """
    baseline_deg = jnp.asarray(baseline_deg, dtype=jnp.float32)
    preheat_max_deg = jnp.ones_like(baseline_deg) * PREHEAT_MAX_SETPOINT_C
    garage_idx = ZONE_ORDER.index("Garage")
    preheat_max_deg = preheat_max_deg.at[..., garage_idx].set(
        GARAGE_LOWER_BOUND_C
    )

    target_deg = jnp.ones_like(baseline_deg) * jnp.asarray(
        target_setpoint_deg,
        dtype=jnp.float32,
    )
    target_deg = jnp.clip(target_deg, baseline_deg, preheat_max_deg)
    span = jnp.maximum(preheat_max_deg - baseline_deg, PREHEAT_FRACTION_EPS)
    fraction = (target_deg - baseline_deg) / span
    fraction = jnp.clip(
        fraction,
        PREHEAT_FRACTION_EPS,
        1.0 - PREHEAT_FRACTION_EPS,
    )
    raw = jnp.log(fraction) - jnp.log1p(-fraction)
    return raw.at[..., garage_idx].set(0.0)


def build_preheat_seed_controls(profiles, active_raw, active_hours):
    """Construct one initial control trajectory for multi-start iLQR.

    Within the two-hour candidate window, intervals earlier than
    ``active_hours`` start near the baseline and the final ``active_hours``
    start at ``active_raw``. Outside the candidate window controls are zero
    because the fixed schedule masks them.
    """
    is_candidate = profiles["is_preheat_nodes"][:T_steps, None]
    hours_to_event = profiles["hours_to_next_dr_start_t"][:T_steps, None]
    is_seed_on = (
        is_candidate
        & (hours_to_event > 0.0)
        & (hours_to_event <= float(active_hours))
    )

    U_seed = jnp.zeros((T_steps, n), dtype=jnp.float32)
    U_seed = jnp.where(is_candidate, RULE_PREHEAT_OFF_RAW, U_seed)
    U_seed = jnp.where(
        is_seed_on,
        jnp.asarray(active_raw, dtype=jnp.float32),
        U_seed,
    )

    garage_idx = ZONE_ORDER.index("Garage")
    return U_seed.at[:, garage_idx].set(0.0)


def build_rule_based_initial_controls(profiles):
    """Return the fixed 24 degC/two-hour reference control trajectory."""
    baseline_deg = profiles["baseline_setpoint_t"][:T_steps]
    rule_target_raw = raw_control_for_preheat_target(
        RULE_BASED_PREHEAT_SETPOINT_C,
        baseline_deg,
    )
    return build_preheat_seed_controls(
        profiles=profiles,
        active_raw=rule_target_raw,
        active_hours=RULE_BASED_PREHEAT_HOURS,
    )


def build_horizon_datetimes(start_dt, n_points=N_POINTS, step_minutes=STEP_MINUTES):
    start_dt = pd.Timestamp(start_dt)
    return [start_dt.to_pydatetime() + timedelta(minutes=step_minutes * k) for k in range(n_points)]


def estimate_historical_temperature_bias(current_day):
    """Estimate EnergyPlus-minus-model residuals by zone and forecast lead.

    Only completed days strictly earlier than ``current_day`` are used, which
    prevents information from the day being controlled from leaking into the
    optimization.  The median residual corrects expected temperatures; a low
    residual quantile supplies the conservative temperature used for DR-floor
    feasibility checks.
    """
    expected = np.zeros((N_POINTS, NUM_ZONES), dtype=np.float32)
    floor = np.zeros_like(expected)
    counts = np.zeros((N_POINTS, NUM_ZONES), dtype=np.int32)
    audit_rows = []
    if not ENABLE_HISTORICAL_TEMPERATURE_BIAS_CORRECTION:
        return expected, floor, counts, pd.DataFrame(audit_rows)

    current_date = pd.Timestamp(current_day).date()
    candidate_dirs = []
    seen = set()
    for root in (BASE_LOG_DIR, LEGACY_BIAS_LOG_DIR):
        root_path = Path(root)
        if not root_path.exists():
            continue
        for plant_path in root_path.glob("dr_day_*/energyplus_plant_log.csv"):
            day_dir = plant_path.parent
            match = re.search(r"dr_day_(\d{8})", day_dir.name)
            if match is None:
                continue
            run_date = datetime.strptime(match.group(1), "%Y%m%d").date()
            key = str(day_dir.resolve())
            if run_date < current_date and key not in seen:
                candidate_dirs.append((run_date, day_dir))
                seen.add(key)

    candidate_dirs = sorted(candidate_dirs, key=lambda item: item[0])[
        -int(BIAS_MAX_PREVIOUS_DAYS):
    ]
    residuals = {(lead, i): [] for lead in range(N_POINTS) for i in range(NUM_ZONES)}

    for run_date, day_dir in candidate_dirs:
        try:
            plant = pd.read_csv(day_dir / "energyplus_plant_log.csv", low_memory=False)
        except Exception as exc:
            print(f"WARNING: bias calibration skipped {day_dir}: {exc!r}")
            continue
        if "simulation_time_local" not in plant.columns:
            continue
        plant = plant.copy()
        plant["_time_utc"] = pd.to_datetime(
            plant["simulation_time_local"], errors="coerce", utc=True
        )
        actual_cols = {
            i: f"T_ep_{zone.lower().replace(' ', '_')}_degC"
            for i, zone in enumerate(ZONE_ORDER)
        }
        keep_cols = ["_time_utc"] + [c for c in actual_cols.values() if c in plant.columns]
        actual = plant[keep_cols].dropna(subset=["_time_utc"]).drop_duplicates("_time_utc", keep="last")

        for prediction_path in sorted(day_dir.glob("mpc_prediction_run_*.csv")):
            try:
                prediction = pd.read_csv(prediction_path, low_memory=False)
            except Exception:
                continue
            if not {"horizon_step", "horizon_timestamp_local"}.issubset(prediction.columns):
                continue
            prediction = prediction.copy()
            prediction["_time_utc"] = pd.to_datetime(
                prediction["horizon_timestamp_local"], errors="coerce", utc=True
            )
            merged = prediction.merge(actual, on="_time_utc", how="inner", suffixes=("", "_actual"))
            for _, row in merged.iterrows():
                try:
                    lead = int(row["horizon_step"])
                except Exception:
                    continue
                if lead <= 0 or lead >= N_POINTS:
                    continue
                for i, zone in enumerate(ZONE_ORDER):
                    if zone == "Garage":
                        continue
                    safe = zone.lower().replace(" ", "_")
                    pred_col = f"Tpred_{safe}_degC"
                    actual_col = actual_cols[i]
                    if pred_col not in merged.columns or actual_col not in merged.columns:
                        continue
                    pred_value = row[pred_col]
                    actual_value = row[actual_col]
                    if pd.notna(pred_value) and pd.notna(actual_value):
                        residuals[(lead, i)].append(float(actual_value) - float(pred_value))

    for lead in range(1, N_POINTS):
        for i, zone in enumerate(ZONE_ORDER):
            values = np.asarray(residuals[(lead, i)], dtype=np.float32)
            count = int(values.size)
            counts[lead, i] = count
            raw_expected = np.nan
            raw_floor = np.nan
            if count >= int(BIAS_MIN_SAMPLES_PER_ZONE_LEAD):
                raw_expected = float(np.median(values))
                raw_floor = float(np.quantile(values, BIAS_FLOOR_RESIDUAL_QUANTILE))
                # raw_floor = float(np.median(values))
                shrinkage = count / (count + float(BIAS_PRIOR_STRENGTH))
                expected[lead, i] = np.clip(
                    shrinkage * raw_expected,
                    -float(BIAS_MAX_ABS_CORRECTION_C),
                    float(BIAS_MAX_ABS_CORRECTION_C),
                )
                floor[lead, i] = np.clip(
                    shrinkage * raw_floor,
                    -float(BIAS_MAX_ABS_CORRECTION_C),
                    float(BIAS_MAX_ABS_CORRECTION_C),
                )
            audit_rows.append({
                "forecast_lead_steps": lead,
                "forecast_lead_hours": lead * float(dt),
                "zone": zone,
                "sample_count": count,
                "raw_median_residual_C": raw_expected,
                "raw_floor_quantile_residual_C": raw_floor,
                "expected_bias_applied_C": float(expected[lead, i]),
                "floor_bias_applied_C": float(floor[lead, i]),
            })

    return expected, floor, counts, pd.DataFrame(audit_rows)


def build_time_profiles(horizon_datetimes):
    is_dr_nodes_np = np.array(
        [is_in_csv_dr_event(dt_obj) for dt_obj in horizon_datetimes],
        dtype=bool,
    )

    is_preheat_nodes_np = np.array(
        [is_preheat_window(dt_obj) for dt_obj in horizon_datetimes],
        dtype=bool,
    )

    is_rule_seed_preheat_nodes_np = np.array(
        [is_rule_based_preheat_seed_window(dt_obj) for dt_obj in horizon_datetimes],
        dtype=bool,
    )

    hours_to_next_dr_start_np = np.array(
        [hours_until_next_dr_start(dt_obj) for dt_obj in horizon_datetimes],
        dtype=np.float32,
    )

    hours_since_dr_end_np = np.array(
        [hours_since_most_recent_dr_end(dt_obj) for dt_obj in horizon_datetimes],
        dtype=np.float32,
    )

    is_night_nodes_np = np.array(
        [is_night_setback(dt_obj) for dt_obj in horizon_datetimes],
        dtype=bool,
    )

    # Event-end nodes are distinct from in-event nodes because DR windows use
    # the half-open interval convention [start, end). This array has N_POINTS
    # entries and can therefore be indexed by both stage and terminal costs.
    is_dr_end_nodes_np = np.array(
        [is_dr_event_end(dt_obj) for dt_obj in horizon_datetimes],
        dtype=bool,
    )

    dr_event_progress_nodes_np = np.array(
        [dr_event_progress_fraction(dt_obj) for dt_obj in horizon_datetimes],
        dtype=np.float32,
    )
    dr_decay_upper_cap_nodes_np = (
        float(DR_EXIT_TARGET_C)
        + (
            float(DR_DECAY_ENVELOPE_START_C)
            - float(DR_EXIT_TARGET_C)
        )
        * (1.0 - dr_event_progress_nodes_np)
    ).astype(np.float32)

    T_low_np = np.stack(
        [preferred_temperature_vector_at_datetime(dt_obj) for dt_obj in horizon_datetimes],
        axis=0,
    ).astype(np.float32)

    # Preserve the interval setpoint schedule before changing the state target
    # at event-end nodes. The interval that begins at the event-end timestamp
    # may return to 21 degC, but the state at that timestamp is the result of the
    # final DR interval and must be evaluated against the 18 degC exit floor.
    baseline_setpoint_np = T_low_np.copy()
    event_exit_floor_vector = np.full(
        (NUM_ZONES,),
        float(DR_EXIT_FLOOR_C),
        dtype=np.float32,
    )
    T_low_np[is_dr_end_nodes_np] = event_exit_floor_vector

    recovery_duration = max(float(POST_DR_RECOVERY_REFERENCE_HOURS), 1.0e-6)
    post_dr_recovery_nodes_np = (
        (hours_since_dr_end_np >= 0.0)
        & (hours_since_dr_end_np < recovery_duration)
        & (~is_dr_nodes_np)
    )
    recovery_fraction_np = np.clip(
        hours_since_dr_end_np / recovery_duration,
        0.0,
        1.0,
    ).astype(np.float32)
    recovery_reference_np = (
        float(DR_EXIT_FLOOR_C)
        + recovery_fraction_np[:, None]
        * (baseline_setpoint_np - float(DR_EXIT_FLOOR_C))
    ).astype(np.float32)
    T_low_np[post_dr_recovery_nodes_np] = np.minimum(
        T_low_np[post_dr_recovery_nodes_np],
        recovery_reference_np[post_dr_recovery_nodes_np],
    )

    # Preferred upper target used to discourage unnecessary overheating. This
    # is distinct from the 25 degC heavily penalized safety ceiling.
    T_soft_upper_np = np.stack(
        [soft_upper_temperature_vector_at_datetime(dt_obj) for dt_obj in horizon_datetimes],
        axis=0,
    ).astype(np.float32)

    T_high_np = np.stack(
        [upper_comfort_bound_vector_at_datetime(dt_obj) for dt_obj in horizon_datetimes],
        axis=0,
    ).astype(np.float32)

    # Compatibility fields retained in the rollout CSV. They no longer
    # contribute to the economic objective.
    w_track_np = np.ones((len(horizon_datetimes),), dtype=np.float32)
    w_temp_excess_np = np.zeros((len(horizon_datetimes),), dtype=np.float32)
    w_move_np = np.where(is_preheat_nodes_np, float(W_PREHEAT_SETPOINT_MOVE), 0.0).astype(np.float32)
    w_power_np = np.zeros((len(horizon_datetimes),), dtype=np.float32)

    interval_midpoints = [
        horizon_datetimes[k] + timedelta(minutes=STEP_MINUTES / 2.0)
        for k in range(len(horizon_datetimes) - 1)
    ]
    is_peak_event_step_np = np.array(
        [is_in_csv_dr_event(dt_obj) for dt_obj in interval_midpoints],
        dtype=bool,
    )

    return {
        "is_dr_nodes": jnp.asarray(is_dr_nodes_np),
        "is_preheat_nodes": jnp.asarray(is_preheat_nodes_np),
        "is_rule_seed_preheat_nodes": jnp.asarray(is_rule_seed_preheat_nodes_np),
        "is_night_nodes": jnp.asarray(is_night_nodes_np),
        "is_dr_end_nodes": jnp.asarray(is_dr_end_nodes_np),
        "dr_event_progress_nodes": jnp.asarray(
            dr_event_progress_nodes_np,
            dtype=jnp.float32,
        ),
        "dr_decay_upper_cap_nodes": jnp.asarray(
            dr_decay_upper_cap_nodes_np,
            dtype=jnp.float32,
        ),
        "post_dr_recovery_nodes": jnp.asarray(post_dr_recovery_nodes_np),
        "post_dr_recovery_fraction_nodes": jnp.asarray(
            recovery_fraction_np,
            dtype=jnp.float32,
        ),
        "hours_to_next_dr_start_t": jnp.asarray(hours_to_next_dr_start_np, dtype=jnp.float32),
        "baseline_setpoint_t": jnp.asarray(baseline_setpoint_np, dtype=jnp.float32),
        "T_low_t": jnp.asarray(T_low_np, dtype=jnp.float32),
        "T_soft_upper_t": jnp.asarray(T_soft_upper_np, dtype=jnp.float32),
        "T_high_t": jnp.asarray(T_high_np, dtype=jnp.float32),
        "temperature_expected_bias_t": jnp.asarray(
            CURRENT_EXPECTED_TEMPERATURE_BIAS_C, dtype=jnp.float32
        ),
        "temperature_floor_bias_t": jnp.asarray(
            CURRENT_FLOOR_TEMPERATURE_BIAS_C, dtype=jnp.float32
        ),
        "w_track_t": jnp.asarray(w_track_np, dtype=jnp.float32),
        "w_temp_excess_t": jnp.asarray(w_temp_excess_np, dtype=jnp.float32),
        "w_move_t": jnp.asarray(w_move_np, dtype=jnp.float32),
        "w_power_t": jnp.asarray(w_power_np, dtype=jnp.float32),
        "is_peak_event_step": jnp.asarray(is_peak_event_step_np),
    }


def flex_d_interval_cost(Eoff_before_kWh, E_step_kWh, is_peak):
    is_peak_f = jnp.asarray(is_peak, dtype=E_step_kWh.dtype)

    E_peak = is_peak_f * E_step_kWh
    E_off = (1.0 - is_peak_f) * E_step_kWh

    remaining_tier1 = jnp.maximum(jnp.float32(0.0), OFFPEAK_BLOCK_CAP_KWH - Eoff_before_kWh)
    E_tier1 = jnp.minimum(E_off, remaining_tier1)
    E_tier2 = jnp.maximum(jnp.float32(0.0), E_off - E_tier1)

    return (
        FLEX_PEAK_D_PER_KWH * E_peak
        + FLEX_OFF1_D_PER_KWH * E_tier1
        + FLEX_OFF2_D_PER_KWH * E_tier2
    )


# 6) Historical Open-Meteo weather
def fetch_open_meteo_historical_15min(lat, lon, start_dt, end_dt, timezone):
    """
    Fetch hourly historical weather from Open-Meteo Archive API and interpolate
    to the 15-minute MPC/control grid.

    Returned columns:
      temperature_2m, shortwave_radiation, direct_radiation, diffuse_radiation,
      direct_normal_irradiance
    """
    start_dt = pd.Timestamp(start_dt).tz_convert(timezone)
    end_dt = pd.Timestamp(end_dt).tz_convert(timezone)

    # Add one extra day to cover the final MPC horizon.
    request_start = start_dt.date().isoformat()
    request_end = (end_dt + pd.Timedelta(hours=HORIZON_HOURS + 2)).date().isoformat()

    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_vars = [
        "temperature_2m",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": request_start,
        "end_date": request_end,
        "hourly": ",".join(hourly_vars),
        "timezone": timezone,
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo response missing 'hourly': {data}")

    h = data["hourly"]
    idx = pd.to_datetime(h["time"])
    if idx.tz is None:
        idx = idx.tz_localize(timezone)
    else:
        idx = idx.tz_convert(timezone)

    df = pd.DataFrame(index=idx)
    for var in hourly_vars:
        df[var] = np.asarray(h.get(var, np.nan), dtype=np.float32)

    # Interpolate to 15-min grid.
    grid = pd.date_range(
        start=start_dt.floor(f"{STEP_MINUTES}min"),
        end=(end_dt + pd.Timedelta(hours=HORIZON_HOURS)).ceil(f"{STEP_MINUTES}min"),
        freq=f"{STEP_MINUTES}min",
        tz=timezone,
    )

    df15 = (
        df.sort_index()
        .reindex(df.index.union(grid))
        .interpolate(method="time")
        .reindex(grid)
        .ffill()
        .bfill()
    )

    # Keep solar non-negative.
    for col in ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance"]:
        df15[col] = df15[col].clip(lower=0.0)

    csv_path = Path(LOG_DIR) / "open_meteo_historical_15min.csv"
    df15.to_csv(csv_path)
    print(f"Historical Open-Meteo weather saved to: {csv_path}")
    return df15


def weather_at_times(weather_df, times):
    idx = pd.DatetimeIndex([pd.Timestamp(t) for t in times])
    if idx.tz is None:
        idx = idx.tz_localize(OPEN_METEO_TIMEZONE)
    else:
        idx = idx.tz_convert(OPEN_METEO_TIMEZONE)

    # Reindex with nearest because EnergyPlus callbacks can be off by numerical seconds.
    out = weather_df.reindex(idx, method="nearest", tolerance=pd.Timedelta(minutes=STEP_MINUTES))
    if out.isna().any().any():
        missing = out[out.isna().any(axis=1)]
        raise RuntimeError(f"Missing historical weather values around: {missing.index[:5].tolist()}")
    return out


def build_historical_aux_base(weather_df, horizon_datetimes):
    w = weather_at_times(weather_df, horizon_datetimes)
    aux_phys = w[["temperature_2m", "shortwave_radiation"]].to_numpy(dtype=np.float32)

    aux_min = np.asarray(u_scaler.data_min_[-2:], dtype=np.float32)
    aux_rng = np.asarray(u_scaler.data_range_[-2:], dtype=np.float32) + 1e-6

    aux_scaled = (aux_phys - aux_min[None, :]) / aux_rng[None, :]
    aux_scaled = np.clip(aux_scaled, 0.0, 1.0)

    return jnp.asarray(aux_scaled, dtype=jnp.float32), aux_phys, [t.isoformat() for t in w.index]


# 7) Initial state and MPC solve
def build_initial_state_from_ep(T0_degC, P0_scaled, sp_prev_deg, eoff_cum0_kwh):
    T0_scaled = scale_T_deg(T0_degC, clip_scaled=True)
    sp_prev0 = scale_sp_deg(sp_prev_deg, clip_scaled=True)

    P0_scaled = np.asarray(P0_scaled, dtype=np.float32)
    if P0_scaled.size == 1:
        P0_scaled = np.full((n,), float(P0_scaled), dtype=np.float32)
    P0_scaled = np.clip(P0_scaled, 0.0, 1.0)

    T0_scaled = jnp.asarray(T0_scaled, dtype=jnp.float32)
    P0_scaled = jnp.asarray(P0_scaled, dtype=jnp.float32)
    sp_prev0 = jnp.asarray(sp_prev0, dtype=jnp.float32)

    y0_2n = jnp.concatenate([T0_scaled, P0_scaled], axis=0)

    z0 = (
        model.z_encoder(y0_2n).astype(y0_2n.dtype)
        if k_total > 0
        else jnp.zeros((0,), dtype=y0_2n.dtype)
    )

    y0_main = jnp.concatenate([y0_2n, z0], axis=0)
    I0 = jnp.zeros((n,), dtype=y0_main.dtype)
    y0_aug = jnp.concatenate([y0_main, I0], axis=0)

    x0 = jnp.concatenate(
        [
            y0_aug,
            sp_prev0,
            jnp.array([jnp.float32(eoff_cum0_kwh)], dtype=jnp.float32),
            jnp.array([jnp.float32(0.0)], dtype=jnp.float32),
            T0_scaled,
        ],
        axis=0,
    )

    snapshot = {
        "T0_degC": np.asarray(T0_degC, dtype=np.float32),
        "sp_prev_deg": np.asarray(sp_prev_deg, dtype=np.float32),
        "P0_scaled": np.asarray(P0_scaled, dtype=np.float32),
    }

    return x0, snapshot


def shift_warm_start(U_opt):
    U_opt = jnp.asarray(U_opt)
    if U_opt.shape[0] == 1:
        return U_opt
    return jnp.concatenate([U_opt[1:], U_opt[-1:]], axis=0)


def solve_mpc_once_from_ep(
    apply_dt,
    eoff_cum0_kwh,
    T0_degC,
    P0_scaled,
    sp_prev_deg,
    weather_df,
    U0=None,
):
    solve_wall_start = time.perf_counter()
    horizon_datetimes = build_horizon_datetimes(apply_dt)
    profiles = build_time_profiles(horizon_datetimes)

    aux_base, aux_phys, forecast_times = build_historical_aux_base(weather_df, horizon_datetimes)

    x0, ep_snapshot = build_initial_state_from_ep(
        T0_degC=T0_degC,
        P0_scaled=P0_scaled,
        sp_prev_deg=sp_prev_deg,
        eoff_cum0_kwh=eoff_cum0_kwh,
    )

    T_low_t = profiles["T_low_t"]
    T_high_t = profiles["T_high_t"]
    is_peak_event_step = profiles["is_peak_event_step"]
    is_dr_end_nodes = profiles["is_dr_end_nodes"]
    dr_decay_upper_cap_nodes = profiles["dr_decay_upper_cap_nodes"]
    expected_temperature_bias_t = profiles["temperature_expected_bias_t"]
    floor_temperature_bias_t = profiles["temperature_floor_bias_t"]

    solver = diffrax.Tsit5()
    step_ctrl = diffrax.PIDController(rtol=1e-3, atol=1e-6)

    def temperature_penalty(
        T_deg,
        T_low,
        T_hard_upper,
        lower_multiplier=1.0,
    ):
        """Comfort-floor and 25 degC safety safeguards."""
        below = jnp.maximum(T_low - T_deg, 0.0)
        above_hard = jnp.maximum(T_deg - T_hard_upper, 0.0)

        return (
            lower_multiplier * W_COMFORT_BELOW * jnp.sum(below ** 2)
            + W_HIGH_SAFETY * jnp.sum(above_hard ** 2)
        )

    garage_idx = ZONE_ORDER.index("Garage")

    # The garage is fixed near 18°C and is not intended to store heat.
    # Exclude it from the DR decay-shape penalty.
    dr_decay_zone_mask = jnp.ones(
        (n,),
        dtype=jnp.float32,
    ).at[garage_idx].set(0.0)

    def dynamics(x, u_raw, t_idx):
        y_aug = x[:y_aug_dim]
        sp_prev = x[y_aug_dim:y_aug_dim + n]
        Eoff_cum = x[EOFF_STATE_INDEX]
        economic_cost_cum = x[ECONOMIC_COST_STATE_INDEX]

        T_current_scaled = y_aug[:n]

        y_main = y_aug[:y_main_dim]
        I_prev = y_aug[y_main_dim:y_aug_dim]

        sp, _ = setpoint_for_interval_from_u_raw(u_raw, t_idx, profiles)
        dTSet = (sp - sp_prev) / dt

        t0 = ts[t_idx]
        t1 = ts[t_idx + 1]
        ts_step = jnp.array([t0, t1], dtype=ts.dtype)

        u_full0 = jnp.concatenate([sp, aux_base[t_idx]], axis=0)
        u_full1 = jnp.concatenate([sp, aux_base[t_idx + 1]], axis=0)

        interp_us = diffrax.LinearInterpolation(
            ts_step,
            jnp.stack([u_full0, u_full1], axis=0),
        )
        interp_aux = diffrax.LinearInterpolation(
            ts_step,
            jnp.stack([aux_base[t_idx], aux_base[t_idx + 1]], axis=0),
        )

        args_rhs = (ts, interp_us, interp_aux, None)

        def vf(t, y_aug_, args_unused):
            y_main_ = y_aug_[:y_main_dim]
            dy_main_ = func_rhs_with_dynamic_dTSet(func, t, y_main_, args_rhs, dTSet)
            P_state_ = y_main_[n:2 * n]
            dI_ = P_state_
            return jnp.concatenate([dy_main_, dI_], axis=0)

        term = diffrax.ODETerm(vf)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=y_aug,
            args=None,
            stepsize_controller=step_ctrl,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=1000,
        )

        y_aug_next = sol.ys[0]
        I_next = y_aug_next[y_main_dim:y_aug_dim]

        P_avg_interval_scaled = (I_next - I_prev) / dt
        P_avg_interval_kW = jnp.maximum(P_scaled_to_kW(P_avg_interval_scaled), 0.0)
        E_step_kWh = jnp.sum(P_avg_interval_kW) * dt

        is_peak = jnp.asarray(jnp.take(is_peak_event_step, t_idx), dtype=y_aug.dtype)
        interval_tariff_cost = flex_d_interval_cost(
            Eoff_cum,
            E_step_kWh,
            is_peak,
        )
        Eoff_cum_next = Eoff_cum + (1.0 - is_peak) * E_step_kWh
        economic_cost_cum_next = economic_cost_cum + interval_tariff_cost

        return jnp.concatenate(
            [
                y_aug_next,
                sp,
                jnp.array([Eoff_cum_next], dtype=y_aug.dtype),
                jnp.array([economic_cost_cum_next], dtype=y_aug.dtype),
                T_current_scaled,
            ],
            axis=0,
        )

    # Economic objective

    def economic_stage_cost(x, u_raw, t_idx):
        y_main = x[:y_main_dim]
        T_scaled = y_main[:n]
        T_previous_scaled = x[T_prev_state_start:T_prev_state_start + n]

        T_deg = T_scaled_to_deg(T_scaled)
        T_previous_deg = T_scaled_to_deg(T_previous_scaled)
        T_expected_deg = T_deg + expected_temperature_bias_t[t_idx]
        T_floor_conservative_deg = T_deg + floor_temperature_bias_t[t_idx]
        T_low = T_low_t[t_idx]
        T_high = T_high_t[t_idx]

        is_preheat_node = jnp.asarray(
            profiles["is_preheat_nodes"][t_idx],
            dtype=T_deg.dtype,
        )
        is_dr_node = jnp.asarray(
            profiles["is_dr_nodes"][t_idx],
            dtype=T_deg.dtype,
        )
        is_dr_end_node = jnp.asarray(
            is_dr_end_nodes[t_idx],
            dtype=T_deg.dtype,
        )

        J_temperature = temperature_penalty(
            T_deg=T_expected_deg,
            T_low=T_low,
            T_hard_upper=T_high,
        )

        _, sp_deg = setpoint_for_interval_from_u_raw(
            u_raw,
            t_idx,
            profiles,
        )
        sp_prev_deg = sp_scaled_to_deg(x[y_aug_dim:y_aug_dim + n])
        J_preheat_move = (
            is_preheat_node
            * W_PREHEAT_SETPOINT_MOVE
            * jnp.sum((sp_deg - sp_prev_deg) ** 2)
        )

        J_control_regularization = (
            W_CONTROL_REGULARIZATION
            * jnp.sum(jnp.asarray(u_raw, dtype=T_deg.dtype) ** 2)
        )

        floor_for_optimization = DR_FLOOR_C + DR_FLOOR_MARGIN_C
        normalized_shortfall = (
            floor_for_optimization - T_floor_conservative_deg
        ) / DR_FLOOR_SMOOTHING_C
        below_floor_indicator = jax.nn.sigmoid(normalized_shortfall)
        floor_shortfall = jnp.maximum(
            floor_for_optimization - T_floor_conservative_deg,
            0.0,
        )

        J_time_below = (
            is_dr_node
            * W_DR_TIME_BELOW
            * dt
            * jnp.sum(dr_decay_zone_mask * below_floor_indicator)
        )
        J_depth_below = (
            is_dr_node
            * W_DR_DEPTH_BELOW
            * dt
            * jnp.sum(dr_decay_zone_mask * floor_shortfall ** 2)
        )
        J_event_end = (
            is_dr_end_node
            * W_DR_EVENT_END
            * jnp.sum(dr_decay_zone_mask * floor_shortfall ** 2)
        )

        event_end_excess = jnp.maximum(
            T_expected_deg - DR_EXIT_TARGET_C,
            0.0,
        )
        J_event_end_excess = (
            is_dr_end_node
            * W_DR_EVENT_END_EXCESS
            * jnp.sum(dr_decay_zone_mask * event_end_excess ** 2)
        )

        dr_decay_upper_cap = jnp.asarray(
            dr_decay_upper_cap_nodes[t_idx],
            dtype=T_deg.dtype,
        )
        stored_heat_excess = jnp.maximum(
            T_expected_deg - dr_decay_upper_cap,
            0.0,
        )
        J_stored_heat = (
            is_dr_node
            * W_DR_STORED_HEAT
            * dt
            * jnp.sum(dr_decay_zone_mask * stored_heat_excess ** 2)
        )

        previous_interval_idx = jnp.maximum(t_idx - 1, 0)
        previous_interval_was_dr = jnp.where(
            t_idx > 0,
            jnp.take(is_peak_event_step, previous_interval_idx),
            jnp.asarray(False),
        ).astype(T_deg.dtype)
        interval_temperature_rise = jnp.maximum(
            T_deg
            - T_previous_deg
            - DR_ALLOWED_RISE_C_PER_H * dt,
            0.0,
        )
        J_temperature_rise = (
            previous_interval_was_dr
            * W_DR_TEMPERATURE_RISE
            * jnp.sum(dr_decay_zone_mask * interval_temperature_rise ** 2)
        )

        return (
            J_temperature
            + J_preheat_move
            + J_control_regularization
            + J_time_below
            + J_depth_below
            + J_event_end
            + J_event_end_excess
            + J_stored_heat
            + J_temperature_rise
        )

    def economic_terminal_cost(x, u_raw):
        del u_raw

        y_main = x[:y_main_dim]
        T_scaled = y_main[:n]
        T_deg = T_scaled_to_deg(T_scaled)
        T_expected_deg = T_deg + expected_temperature_bias_t[T_steps]
        T_floor_conservative_deg = T_deg + floor_temperature_bias_t[T_steps]
        T_low = T_low_t[T_steps]
        T_high = T_high_t[T_steps]

        J_tariff = W_ECONOMIC_COST * x[ECONOMIC_COST_STATE_INDEX]

        J_terminal_temperature = temperature_penalty(
            T_deg=T_expected_deg,
            T_low=T_low,
            T_hard_upper=T_high,
            lower_multiplier=W_TERMINAL_COMFORT_MULTIPLIER,
        )

        is_dr_terminal = jnp.asarray(
            profiles["is_dr_nodes"][T_steps],
            dtype=T_deg.dtype,
        )
        is_dr_end_terminal = jnp.asarray(
            is_dr_end_nodes[T_steps],
            dtype=T_deg.dtype,
        )

        floor_for_optimization = DR_FLOOR_C + DR_FLOOR_MARGIN_C
        normalized_shortfall = (
            floor_for_optimization - T_floor_conservative_deg
        ) / DR_FLOOR_SMOOTHING_C
        below_floor_indicator = jax.nn.sigmoid(normalized_shortfall)
        floor_shortfall = jnp.maximum(
            floor_for_optimization - T_floor_conservative_deg,
            0.0,
        )

        J_terminal_time_below = (
            is_dr_terminal
            * W_DR_TIME_BELOW
            * dt
            * jnp.sum(dr_decay_zone_mask * below_floor_indicator)
        )
        J_terminal_depth_below = (
            is_dr_terminal
            * W_DR_DEPTH_BELOW
            * dt
            * jnp.sum(dr_decay_zone_mask * floor_shortfall ** 2)
        )
        J_terminal_event_end = (
            is_dr_end_terminal
            * W_DR_EVENT_END
            * jnp.sum(dr_decay_zone_mask * floor_shortfall ** 2)
        )

        terminal_event_end_excess = jnp.maximum(
            T_expected_deg - DR_EXIT_TARGET_C,
            0.0,
        )
        J_terminal_event_end_excess = (
            is_dr_end_terminal
            * W_DR_EVENT_END_EXCESS
            * jnp.sum(
                dr_decay_zone_mask * terminal_event_end_excess ** 2
            )
        )

        terminal_decay_upper_cap = jnp.asarray(
            dr_decay_upper_cap_nodes[T_steps],
            dtype=T_deg.dtype,
        )
        terminal_stored_heat_excess = jnp.maximum(
            T_expected_deg - terminal_decay_upper_cap,
            0.0,
        )
        J_terminal_stored_heat = (
            is_dr_terminal
            * W_DR_STORED_HEAT
            * dt
            * jnp.sum(
                dr_decay_zone_mask * terminal_stored_heat_excess ** 2
            )
        )

        T_previous_scaled = x[T_prev_state_start:T_prev_state_start + n]
        T_previous_deg = T_scaled_to_deg(T_previous_scaled)
        final_interval_was_dr = jnp.asarray(
            is_peak_event_step[T_steps - 1],
            dtype=T_deg.dtype,
        )
        terminal_temperature_rise = jnp.maximum(
            T_deg
            - T_previous_deg
            - DR_ALLOWED_RISE_C_PER_H * dt,
            0.0,
        )
        J_terminal_temperature_rise = (
            final_interval_was_dr
            * W_DR_TEMPERATURE_RISE
            * jnp.sum(dr_decay_zone_mask * terminal_temperature_rise ** 2)
        )

        return (
            J_tariff
            + J_terminal_temperature
            + J_terminal_time_below
            + J_terminal_depth_below
            + J_terminal_event_end
            + J_terminal_event_end_excess
            + J_terminal_stored_heat
            + J_terminal_temperature_rise
        )


    def cost(x, u_raw, t_idx):
        return jax.lax.cond(
            t_idx == T_steps,
            economic_terminal_cost,
            lambda x_, u_: economic_stage_cost(x_, u_, t_idx),
            x,
            u_raw,
        )

    # Multi-start solve and rule-based economic fallback
    def rollout_fixed_controls(U_sequence):
        """Propagate a fixed open-loop control sequence through dynamics()."""
        U_sequence = jnp.asarray(U_sequence, dtype=jnp.float32)

        if USE_LAX_SCAN_FOR_FIXED_ROLLOUTS:
            time_indices = jnp.arange(T_steps, dtype=jnp.int32)

            def rollout_step(x_current, inputs):
                u_current, t_idx = inputs
                x_next = dynamics(x_current, u_current, t_idx)
                return x_next, x_next

            _, X_tail = jax.lax.scan(
                rollout_step,
                x0,
                (U_sequence, time_indices),
            )
            return jnp.concatenate([x0[None, :], X_tail], axis=0)

        x_current = x0
        X_nodes = [x_current]
        for rollout_idx in range(T_steps):
            x_current = dynamics(
                x_current,
                U_sequence[rollout_idx],
                rollout_idx,
            )
            X_nodes.append(x_current)
        return jnp.stack(X_nodes, axis=0)

    def fixed_rollout_objective(X_sequence, U_sequence):
        """Evaluate the same combined objective used by iLQR."""
        time_indices = jnp.arange(T_steps, dtype=jnp.int32)
        stage_costs = jax.vmap(economic_stage_cost)(
            X_sequence[:-1],
            U_sequence,
            time_indices,
        )
        return jnp.sum(stage_costs) + economic_terminal_cost(
            X_sequence[-1],
            jnp.zeros((n,), dtype=jnp.float32),
        )

    T_low_host = np.asarray(jax.device_get(T_low_t), dtype=np.float32)
    T_high_host = np.asarray(jax.device_get(T_high_t), dtype=np.float32)
    expected_bias_host = np.asarray(
        jax.device_get(expected_temperature_bias_t), dtype=np.float32
    )
    floor_bias_host = np.asarray(
        jax.device_get(floor_temperature_bias_t), dtype=np.float32
    )
    is_dr_host = np.asarray(
        jax.device_get(profiles["is_dr_nodes"]),
        dtype=bool,
    )
    is_dr_end_host = np.asarray(
        jax.device_get(profiles["is_dr_end_nodes"]),
        dtype=bool,
    )
    is_peak_event_step_host = np.asarray(
        jax.device_get(profiles["is_peak_event_step"]),
        dtype=bool,
    )
    non_garage_mask = np.ones((n,), dtype=bool)
    non_garage_mask[garage_idx] = False
    dr_floor_for_selection = float(DR_FLOOR_C + DR_FLOOR_MARGIN_C)

    def evaluate_candidate(
        name,
        X_sequence,
        U_sequence,
        combined_objective,
        optimized,
        solve_seconds=np.nan,
    ):
        """Detach one candidate and calculate tariff/feasibility metrics."""
        combined_objective = jax.block_until_ready(combined_objective)
        X_host = np.asarray(jax.device_get(X_sequence), dtype=np.float32)
        U_host = np.asarray(jax.device_get(U_sequence), dtype=np.float32)
        objective_host = float(np.asarray(jax.device_get(combined_objective)))

        numerically_valid = bool(
            np.isfinite(objective_host)
            and np.all(np.isfinite(X_host))
            and np.all(np.isfinite(U_host))
        )

        if numerically_valid:
            T_deg_host = np.asarray(
                T_scaled_to_deg(jnp.asarray(X_host[:, :n])),
                dtype=np.float32,
            )
            predicted_tariff_cost = float(
                X_host[-1, ECONOMIC_COST_STATE_INDEX]
            )
            integrated_power_scaled = X_host[:, y_main_dim:y_aug_dim]
            interval_average_power_scaled = (
                np.diff(integrated_power_scaled, axis=0) / float(dt)
            )
            interval_average_power_kW = np.maximum(
                np.asarray(
                    P_scaled_to_kW(
                        jnp.asarray(interval_average_power_scaled)
                    ),
                    dtype=np.float32,
                ),
                0.0,
            )
            interval_heating_energy_kWh = (
                np.sum(interval_average_power_kW, axis=1) * float(dt)
            )
            predicted_total_heating_energy_kWh = float(
                np.sum(interval_heating_energy_kWh)
            )
            predicted_dr_event_energy_kWh = float(
                np.sum(
                    interval_heating_energy_kWh[is_peak_event_step_host]
                )
            )
            numerically_valid = bool(
                np.isfinite(predicted_tariff_cost)
                and np.all(np.isfinite(T_deg_host))
                and np.isfinite(predicted_total_heating_energy_kWh)
                and np.isfinite(predicted_dr_event_energy_kWh)
            )
        else:
            T_deg_host = np.full((N_POINTS, n), np.nan, dtype=np.float32)
            predicted_tariff_cost = np.nan
            predicted_total_heating_energy_kWh = np.nan
            predicted_dr_event_energy_kWh = np.nan

        if numerically_valid:
            T_expected_host = T_deg_host + expected_bias_host
            T_floor_host = T_deg_host + floor_bias_host
            T_future = T_expected_host[1:]
            T_floor_future = T_floor_host[1:]
            T_low_future = T_low_host[1:]
            T_high_future = T_high_host[1:]
            max_comfort_shortfall = float(
                np.max(np.maximum(T_low_future - T_future, 0.0))
            )
            max_safety_excess = float(
                np.max(np.maximum(T_future - T_high_future, 0.0))
            )

            dr_mask = is_dr_host[1:, None] & non_garage_mask[None, :]
            if np.any(dr_mask):
                minimum_dr_temperature = float(np.min(T_floor_future[dr_mask]))
                max_dr_floor_shortfall = max(
                    dr_floor_for_selection - minimum_dr_temperature,
                    0.0,
                )
            else:
                minimum_dr_temperature = np.nan
                max_dr_floor_shortfall = 0.0

            event_end_future_mask = is_dr_end_host[1:]
            if np.any(event_end_future_mask):
                event_exit_temperatures = T_future[
                    event_end_future_mask
                ][:, non_garage_mask]
                mean_event_exit_temperature = float(
                    np.mean(event_exit_temperatures)
                )
                max_event_exit_temperature = float(
                    np.max(event_exit_temperatures)
                )
                event_exit_band_error = (
                    np.maximum(
                        float(DR_EXIT_FLOOR_C) - event_exit_temperatures,
                        0.0,
                    )
                    + np.maximum(
                        event_exit_temperatures - float(DR_EXIT_TARGET_C),
                        0.0,
                    )
                )
                mean_event_exit_band_error = float(
                    np.mean(event_exit_band_error)
                )
                max_event_exit_band_error = float(
                    np.max(event_exit_band_error)
                )
                max_event_exit_excess = max(
                    max_event_exit_temperature - float(DR_EXIT_TARGET_C),
                    0.0,
                )
            else:
                mean_event_exit_temperature = np.nan
                max_event_exit_temperature = np.nan
                mean_event_exit_band_error = 0.0
                max_event_exit_band_error = 0.0
                max_event_exit_excess = 0.0
        else:
            max_comfort_shortfall = np.inf
            max_safety_excess = np.inf
            minimum_dr_temperature = np.nan
            max_dr_floor_shortfall = np.inf
            mean_event_exit_temperature = np.nan
            max_event_exit_temperature = np.nan
            mean_event_exit_band_error = np.inf
            max_event_exit_band_error = np.inf
            max_event_exit_excess = np.inf

        comfort_violation = max(
            max_comfort_shortfall - MAX_NORMAL_COMFORT_SHORTFALL_C,
            0.0,
        )
        dr_violation = max(
            max_dr_floor_shortfall - MAX_DR_FLOOR_SHORTFALL_C,
            0.0,
        )
        safety_violation = max(
            max_safety_excess - MAX_SAFETY_EXCESS_C,
            0.0,
        )
        constraint_violation_score = (
            comfort_violation
            + 10.0 * dr_violation
            + 100.0 * safety_violation
        )

        feasible = bool(
            numerically_valid
            and comfort_violation <= 1.0e-6
            and dr_violation <= 1.0e-6
            and safety_violation <= 1.0e-6
        )

        return {
            "name": str(name),
            "X": X_host,
            "U": U_host,
            "obj": objective_host,
            "predicted_tariff_cost": predicted_tariff_cost,
            "predicted_total_heating_energy_kWh": (
                predicted_total_heating_energy_kWh
            ),
            "predicted_dr_event_energy_kWh": predicted_dr_event_energy_kWh,
            "feasible": feasible,
            "numerically_valid": numerically_valid,
            "optimized": bool(optimized),
            "solve_seconds": float(solve_seconds),
            "max_comfort_shortfall_C": max_comfort_shortfall,
            "minimum_dr_temperature_C": minimum_dr_temperature,
            "max_dr_floor_shortfall_C": max_dr_floor_shortfall,
            "mean_event_exit_temperature_C": mean_event_exit_temperature,
            "max_event_exit_temperature_C": max_event_exit_temperature,
            "mean_event_exit_band_error_C": mean_event_exit_band_error,
            "max_event_exit_band_error_C": max_event_exit_band_error,
            "max_event_exit_excess_above_target_C": max_event_exit_excess,
            "max_safety_excess_C": max_safety_excess,
            "constraint_violation_score": float(constraint_violation_score),
        }

    candidates = []
    solver_failures = []

    # Fixed reference: this trajectory is evaluated but never modified by iLQR.
    U_rule = build_rule_based_initial_controls(profiles)
    fixed_rule_start = time.perf_counter()
    X_rule = rollout_fixed_controls(U_rule)
    obj_rule = fixed_rollout_objective(X_rule, U_rule)
    rule_candidate = evaluate_candidate(
        name="fixed_rule_24C_2h",
        X_sequence=X_rule,
        U_sequence=U_rule,
        combined_objective=obj_rule,
        optimized=False,
    )
    rule_candidate["solve_seconds"] = float(
        time.perf_counter() - fixed_rule_start
    )
    candidates.append(rule_candidate)

    # Fixed no-preheat trajectory is also eligible when it satisfies the same
    # explicit feasibility checks. Avoid a duplicate rollout when controls are
    # identical because no preheat interval appears in the horizon.
    U_no_preheat = build_preheat_seed_controls(
        profiles=profiles,
        active_raw=RULE_PREHEAT_OFF_RAW,
        active_hours=0.0,
    )
    rule_and_no_preheat_are_equal = bool(
        np.allclose(
            np.asarray(jax.device_get(U_rule)),
            np.asarray(jax.device_get(U_no_preheat)),
            rtol=0.0,
            atol=1.0e-7,
        )
    )
    if not rule_and_no_preheat_are_equal:
        fixed_no_preheat_start = time.perf_counter()
        X_no_preheat = rollout_fixed_controls(U_no_preheat)
        obj_no_preheat = fixed_rollout_objective(
            X_no_preheat,
            U_no_preheat,
        )
        no_preheat_candidate = evaluate_candidate(
            name="fixed_no_preheat",
            X_sequence=X_no_preheat,
            U_sequence=U_no_preheat,
            combined_objective=obj_no_preheat,
            optimized=False,
        )
        no_preheat_candidate["solve_seconds"] = float(
            time.perf_counter() - fixed_no_preheat_start
        )
        candidates.append(no_preheat_candidate)

    U_moderate = build_preheat_seed_controls(
        profiles=profiles,
        active_raw=MODERATE_PREHEAT_RAW,
        active_hours=MODERATE_PREHEAT_HOURS,
    )

    if U0 is not None:
        U0 = jnp.asarray(U0, dtype=jnp.float32)
        if U0.shape != (T_steps, n):
            raise ValueError(
                f"U0 must have shape {(T_steps, n)}, got {U0.shape}"
            )

    reference_cost = rule_candidate["predicted_tariff_cost"]
    required_savings = max(
        float(MIN_PREDICTED_SAVINGS_DOLLARS),
        float(MIN_PREDICTED_SAVINGS_FRACTION)
        * max(float(reference_cost), 0.0)
        if np.isfinite(reference_cost)
        else float(MIN_PREDICTED_SAVINGS_DOLLARS),
    )

    preheat_visible = bool(
        np.any(
            np.asarray(
                jax.device_get(profiles["is_preheat_nodes"][:T_steps]),
                dtype=bool,
            )
        )
    )
    # Solving before the current interval enters the preheat window cannot
    # change the command applied now. Receding-horizon MPC will solve again at
    # the first actionable interval with a horizon that reaches DR event end.
    preheat_actionable_now = bool(
        np.asarray(
            jax.device_get(profiles["is_preheat_nodes"][0]),
            dtype=bool,
        )
    )

    apply_timestamp = pd.Timestamp(apply_dt)
    minutes_since_midnight = (
        int(apply_timestamp.hour) * 60
        + int(apply_timestamp.minute)
    )
    refresh_period_minutes = max(
        int(FULL_MULTISTART_REFRESH_MINUTES),
        STEP_MINUTES,
    )
    periodic_refresh_due = bool(
        minutes_since_midnight % refresh_period_minutes == 0
    )
    full_multistart_refresh = bool(
        preheat_actionable_now
        and ENABLE_MULTISTART
        and (
            not ADAPTIVE_MULTISTART
            or U0 is None
            or periodic_refresh_due
        )
    )

    executed_seed_names = []
    adaptive_rescue_triggered = False

    def run_ilqr_seed(seed_name, U_seed, maxiter):
        """Run one iLQR seed and retain failures as diagnostics."""
        seed_start = time.perf_counter()
        try:
            (
                X_candidate,
                U_candidate,
                obj_candidate,
                *_optimizer_info,
            ) = optimizers.ilqr(
                cost,
                dynamics,
                x0,
                jnp.asarray(U_seed, dtype=jnp.float32),
                maxiter=int(maxiter),
            )
            candidate = evaluate_candidate(
                name=seed_name,
                X_sequence=X_candidate,
                U_sequence=U_candidate,
                combined_objective=obj_candidate,
                optimized=True,
            )
            candidate["solve_seconds"] = float(
                time.perf_counter() - seed_start
            )
            candidates.append(candidate)
            executed_seed_names.append(str(seed_name))
            del _optimizer_info
            return candidate
        except Exception as exc:
            executed_seed_names.append(str(seed_name))
            solver_failures.append({
                "name": str(seed_name),
                "status": "failed",
                "error": repr(exc),
                "solve_seconds": time.perf_counter() - seed_start,
            })
            print(f"WARNING: {seed_name} failed: {exc!r}")
            return None

    def candidate_clears_rule(candidate):
        if candidate is None or not candidate["feasible"]:
            return False
        if not rule_candidate["feasible"]:
            return True
        return bool(
            candidate["predicted_tariff_cost"]
            <= reference_cost + float(MAX_PREDICTED_COST_INCREASE_VS_RULE_DOLLARS)
            and (
                candidate["predicted_tariff_cost"]
                <= reference_cost - required_savings
                or candidate["obj"] < rule_candidate["obj"] - 1.0e-6
            )
        )

    # No iLQR is useful when every setpoint in the horizon is fixed by the
    # normal/night/DR schedule. The fixed rule rollout already represents that
    # unique command trajectory.
    if preheat_actionable_now:
        if U0 is not None:
            primary_name = "ilqr_shifted_warm_start"
            primary_seed = U0
        else:
            primary_name = "ilqr_moderate_start"
            primary_seed = U_moderate

        primary_candidate = run_ilqr_seed(
            primary_name,
            primary_seed,
            REOPT_MAXITER,
        )

        any_candidate_clears_rule = any(
            candidate_clears_rule(candidate)
            for candidate in candidates
        )

        run_rescue_starts = bool(
            ENABLE_MULTISTART
            and (
                full_multistart_refresh
                or not any_candidate_clears_rule
            )
        )

        if run_rescue_starts:
            adaptive_rescue_triggered = not full_multistart_refresh
            rescue_seeds = [
                ("ilqr_moderate_start", U_moderate),
                ("ilqr_no_preheat_start", U_no_preheat),
            ]

            primary_seed_host = np.asarray(
                jax.device_get(jnp.asarray(primary_seed)),
                dtype=np.float32,
            )
            for rescue_name, rescue_seed in rescue_seeds:
                rescue_seed_host = np.asarray(
                    jax.device_get(jnp.asarray(rescue_seed)),
                    dtype=np.float32,
                )
                if np.allclose(
                    rescue_seed_host,
                    primary_seed_host,
                    rtol=0.0,
                    atol=1.0e-7,
                ):
                    continue

                rescue_candidate = run_ilqr_seed(
                    rescue_name,
                    rescue_seed,
                    MULTISTART_MAXITER,
                )

                # Between periodic refreshes, stop as soon as a rescue candidate
                # clears the rule safeguard. Full refreshes evaluate both starts.
                if (
                    ADAPTIVE_MULTISTART
                    and not full_multistart_refresh
                    and candidate_clears_rule(rescue_candidate)
                ):
                    break

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["numerically_valid"]
    ]
    feasible_candidates = [
        candidate
        for candidate in valid_candidates
        if candidate["feasible"]
    ]

    if not valid_candidates:
        raise RuntimeError(
            "Every fixed and optimized MPC candidate produced non-finite values."
        )

    if rule_candidate["feasible"]:
        acceptable_alternatives = [
            candidate
            for candidate in feasible_candidates
            if candidate["name"] != rule_candidate["name"]
            and candidate_clears_rule(candidate)
        ]
        if acceptable_alternatives:
            selected = min(
                acceptable_alternatives,
                key=lambda candidate: (
                    candidate["obj"],
                    candidate["predicted_tariff_cost"],
                ),
            )
            selection_reason = "feasible_candidate_improves_complete_objective"
        else:
            selected = rule_candidate
            selection_reason = "rule_fallback_no_safe_complete_improvement"
    elif feasible_candidates:
        selected = min(
            feasible_candidates,
            key=lambda candidate: (
                candidate["obj"],
                candidate["predicted_tariff_cost"],
            ),
        )
        selection_reason = "rule_infeasible_selected_feasible_candidate"
    else:
        # If every candidate violates at least one predicted thermal
        # constraint, choose the numerically valid trajectory with the smallest
        # aggregate violation. Predicted tariff cost breaks equal-violation
        # ties. This preserves the original full-logging fallback behavior.
        selected = min(
            valid_candidates,
            key=lambda candidate: (
                candidate["constraint_violation_score"],
                candidate["predicted_tariff_cost"],
            ),
        )
        selection_reason = "no_feasible_candidate_minimum_violation_fallback"

    if not preheat_actionable_now:
        selection_reason = "fixed_schedule_no_actionable_preheat"

    X_opt = selected["X"]
    U_opt = selected["U"]
    obj = float(selected["obj"])
    predicted_energy_cost_dollars = float(
        selected["predicted_tariff_cost"]
    )
    predicted_savings_vs_rule_dollars = (
        float(reference_cost - predicted_energy_cost_dollars)
        if np.isfinite(reference_cost)
        else np.nan
    )

    candidate_summary = []
    for candidate in candidates:
        candidate_summary.append({
            "name": candidate["name"],
            "selected": bool(candidate is selected),
            "optimized": bool(candidate["optimized"]),
            "solve_seconds": float(candidate["solve_seconds"]),
            "numerically_valid": bool(candidate["numerically_valid"]),
            "feasible": bool(candidate["feasible"]),
            "predicted_tariff_cost_dollars": float(
                candidate["predicted_tariff_cost"]
            ),
            "predicted_total_heating_energy_kWh": float(
                candidate["predicted_total_heating_energy_kWh"]
            ),
            "predicted_dr_event_energy_kWh": float(
                candidate["predicted_dr_event_energy_kWh"]
            ),
            "combined_objective": float(candidate["obj"]),
            "predicted_savings_vs_rule_dollars": (
                float(
                    reference_cost
                    - candidate["predicted_tariff_cost"]
                )
                if np.isfinite(reference_cost)
                and np.isfinite(candidate["predicted_tariff_cost"])
                else np.nan
            ),
            "max_comfort_shortfall_C": float(
                candidate["max_comfort_shortfall_C"]
            ),
            "minimum_dr_temperature_C": float(
                candidate["minimum_dr_temperature_C"]
            ),
            "max_dr_floor_shortfall_C": float(
                candidate["max_dr_floor_shortfall_C"]
            ),
            "mean_event_exit_temperature_C": float(
                candidate["mean_event_exit_temperature_C"]
            ),
            "max_event_exit_temperature_C": float(
                candidate["max_event_exit_temperature_C"]
            ),
            "mean_event_exit_band_error_C": float(
                candidate["mean_event_exit_band_error_C"]
            ),
            "max_event_exit_band_error_C": float(
                candidate["max_event_exit_band_error_C"]
            ),
            "max_event_exit_excess_above_target_C": float(
                candidate["max_event_exit_excess_above_target_C"]
            ),
            "max_safety_excess_C": float(
                candidate["max_safety_excess_C"]
            ),
            "constraint_violation_score": float(
                candidate["constraint_violation_score"]
            ),
        })

    for failure in solver_failures:
        candidate_summary.append({
            "name": failure["name"],
            "selected": False,
            "optimized": True,
            "solve_seconds": float(failure["solve_seconds"]),
            "numerically_valid": False,
            "feasible": False,
            "predicted_tariff_cost_dollars": np.nan,
            "predicted_total_heating_energy_kWh": np.nan,
            "predicted_dr_event_energy_kWh": np.nan,
            "combined_objective": np.nan,
            "predicted_savings_vs_rule_dollars": np.nan,
            "max_comfort_shortfall_C": np.nan,
            "minimum_dr_temperature_C": np.nan,
            "max_dr_floor_shortfall_C": np.nan,
            "mean_event_exit_temperature_C": np.nan,
            "max_event_exit_temperature_C": np.nan,
            "mean_event_exit_band_error_C": np.nan,
            "max_event_exit_band_error_C": np.nan,
            "max_event_exit_excess_above_target_C": np.nan,
            "max_safety_excess_C": np.nan,
            "constraint_violation_score": np.inf,
            "error": failure["error"],
        })

    # Release non-selected candidate trajectories before the next MPC solve.
    for candidate in candidates:
        if candidate is not selected:
            candidate.pop("X", None)
            candidate.pop("U", None)

    return {
        "apply_dt": pd.Timestamp(apply_dt),
        "horizon_datetimes": horizon_datetimes,
        "profiles": profiles,
        "aux_base": aux_base,
        "aux_phys": aux_phys,
        "forecast_times": forecast_times,
        "x0": x0,
        "ep_snapshot": ep_snapshot,
        "X_opt": X_opt,
        "U_opt": U_opt,
        "obj": obj,
        "predicted_energy_cost_dollars": predicted_energy_cost_dollars,
        "selected_candidate_name": selected["name"],
        "selection_reason": selection_reason,
        "rule_predicted_cost_dollars": float(reference_cost),
        "required_savings_dollars": float(required_savings),
        "predicted_savings_vs_rule_dollars": (
            predicted_savings_vs_rule_dollars
        ),
        "selected_candidate_feasible": bool(selected["feasible"]),
        "selected_mean_event_exit_temperature_C": float(
            selected["mean_event_exit_temperature_C"]
        ),
        "selected_max_event_exit_temperature_C": float(
            selected["max_event_exit_temperature_C"]
        ),
        "selected_mean_event_exit_band_error_C": float(
            selected["mean_event_exit_band_error_C"]
        ),
        "selected_max_event_exit_band_error_C": float(
            selected["max_event_exit_band_error_C"]
        ),
        "selected_max_event_exit_excess_C": float(
            selected["max_event_exit_excess_above_target_C"]
        ),
        "selected_predicted_total_heating_energy_kWh": float(
            selected["predicted_total_heating_energy_kWh"]
        ),
        "selected_predicted_dr_event_energy_kWh": float(
            selected["predicted_dr_event_energy_kWh"]
        ),
        "rule_predicted_total_heating_energy_kWh": float(
            rule_candidate["predicted_total_heating_energy_kWh"]
        ),
        "rule_predicted_dr_event_energy_kWh": float(
            rule_candidate["predicted_dr_event_energy_kWh"]
        ),
        "preheat_visible_in_horizon": preheat_visible,
        "preheat_actionable_now": preheat_actionable_now,
        "multistart_seed_count": int(len(executed_seed_names)),
        "ilqr_solve_count": int(len(executed_seed_names)),
        "executed_seed_names": list(executed_seed_names),
        "full_multistart_refresh": bool(full_multistart_refresh),
        "adaptive_rescue_triggered": bool(adaptive_rescue_triggered),
        "solve_wall_seconds": float(time.perf_counter() - solve_wall_start),
        "candidate_summary": candidate_summary,
        "info": None,
    }


# 8) Prediction logging
def build_prediction_dataframe(run_idx, solve_result):
    X_opt = np.asarray(solve_result["X_opt"], dtype=np.float32)
    U_opt = np.asarray(solve_result["U_opt"], dtype=np.float32)
    aux_phys = np.asarray(solve_result["aux_phys"], dtype=np.float32)
    horizon_datetimes = solve_result["horizon_datetimes"]
    is_peak_event_step = np.asarray(solve_result["profiles"]["is_peak_event_step"])
    is_dr_nodes = np.asarray(solve_result["profiles"]["is_dr_nodes"])
    is_dr_end_nodes = np.asarray(
        solve_result["profiles"]["is_dr_end_nodes"],
        dtype=bool,
    )
    dr_event_progress_nodes = np.asarray(
        solve_result["profiles"]["dr_event_progress_nodes"],
        dtype=np.float32,
    )
    dr_decay_upper_cap_nodes = np.asarray(
        solve_result["profiles"]["dr_decay_upper_cap_nodes"],
        dtype=np.float32,
    )
    post_dr_recovery_nodes = np.asarray(
        solve_result["profiles"]["post_dr_recovery_nodes"],
        dtype=bool,
    )
    post_dr_recovery_fraction_nodes = np.asarray(
        solve_result["profiles"]["post_dr_recovery_fraction_nodes"],
        dtype=np.float32,
    )
    is_preheat_nodes = np.asarray(solve_result["profiles"]["is_preheat_nodes"])
    is_rule_seed_preheat_nodes = np.asarray(
        solve_result["profiles"]["is_rule_seed_preheat_nodes"]
    )
    is_night_nodes = np.asarray(solve_result["profiles"]["is_night_nodes"])
    hours_to_next_dr_start_t = np.asarray(
        solve_result["profiles"]["hours_to_next_dr_start_t"],
        dtype=np.float32,
    )
    baseline_setpoint_t = np.asarray(
        solve_result["profiles"]["baseline_setpoint_t"],
        dtype=np.float32,
    )
    T_low_t = np.asarray(solve_result["profiles"]["T_low_t"], dtype=np.float32)
    T_soft_upper_t = np.asarray(
        solve_result["profiles"]["T_soft_upper_t"],
        dtype=np.float32,
    )
    T_high_t = np.asarray(solve_result["profiles"]["T_high_t"], dtype=np.float32)
    expected_bias_t = np.asarray(
        solve_result["profiles"]["temperature_expected_bias_t"], dtype=np.float32
    )
    floor_bias_t = np.asarray(
        solve_result["profiles"]["temperature_floor_bias_t"], dtype=np.float32
    )
    w_temp_excess_t = np.asarray(
        solve_result["profiles"]["w_temp_excess_t"],
        dtype=np.float32,
    )

    T_scaled_nodes = X_opt[:, :n]
    P_scaled_nodes = X_opt[:, n:2 * n]
    I_nodes = X_opt[:, y_main_dim:y_aug_dim]

    T_deg_nodes = np.asarray(T_scaled_to_deg(jnp.asarray(T_scaled_nodes)), dtype=np.float32)
    T_expected_nodes = T_deg_nodes + expected_bias_t
    T_floor_conservative_nodes = T_deg_nodes + floor_bias_t
    P_kW_nodes = np.asarray(P_scaled_to_kW(jnp.asarray(P_scaled_nodes)), dtype=np.float32)
    P_kW_nodes = np.maximum(P_kW_nodes, 0.0)

    sp_deg_intervals = np.asarray(
        [
            np.asarray(
                setpoint_for_interval_from_u_raw(jnp.asarray(U_opt[k]), k, solve_result["profiles"])[1],
                dtype=np.float32,
            )
            for k in range(T_steps)
        ],
        dtype=np.float32,
    )

    preheat_increment_degC = np.maximum(
        sp_deg_intervals - baseline_setpoint_t[:T_steps],
        0.0,
    )
    optimized_preheat_active = (
        is_preheat_nodes[:T_steps, None]
        & (preheat_increment_degC >= (float(STEP_C) / 2.0))
    )
    optimized_preheat_duration_h = (
        np.sum(optimized_preheat_active, axis=0).astype(np.float32)
        * float(dt)
    )
    optimized_preheat_max_setpoint_C = np.full((n,), np.nan, dtype=np.float32)
    for i in range(n):
        if np.any(optimized_preheat_active[:, i]):
            optimized_preheat_max_setpoint_C[i] = float(
                np.max(sp_deg_intervals[optimized_preheat_active[:, i], i])
            )

    interval_avg_power_kW = []
    interval_energy_kWh = []
    interval_cost_dollars = []
    Eoff_roll = float(X_opt[0, EOFF_STATE_INDEX])

    for k in range(T_steps):
        I_prev = I_nodes[k]
        I_next = I_nodes[k + 1]
        P_avg_scaled = (I_next - I_prev) / float(dt)
        P_avg_kW_zone = np.asarray(P_scaled_to_kW(jnp.asarray(P_avg_scaled)), dtype=np.float32)
        P_avg_kW_zone = np.maximum(P_avg_kW_zone, 0.0)
        E_step_kWh = float(np.sum(P_avg_kW_zone) * float(dt))
        J_cost = float(
            flex_d_interval_cost(
                jnp.float32(Eoff_roll),
                jnp.float32(E_step_kWh),
                bool(is_peak_event_step[k]),
            )
        )

        interval_avg_power_kW.append(P_avg_kW_zone)
        interval_energy_kWh.append(E_step_kWh)
        interval_cost_dollars.append(J_cost)

        if not bool(is_peak_event_step[k]):
            Eoff_roll += E_step_kWh

    rows = []
    for k in range(N_POINTS):
        row = {
            "run_idx": run_idx,
            "apply_time_local": pd.Timestamp(solve_result["apply_dt"]).isoformat(),
            "horizon_step": k,
            "horizon_timestamp_local": pd.Timestamp(horizon_datetimes[k]).isoformat(),
            "historical_weather_time": solve_result["forecast_times"][k],
            "ext_temp_C": float(aux_phys[k, 0]),
            "ghi_Wm2": float(aux_phys[k, 1]),
            "dr_active_node": bool(is_dr_nodes[k]),
            "dr_event_end_node": bool(is_dr_end_nodes[k]),
            "dr_event_progress_fraction": float(
                dr_event_progress_nodes[k]
            ),
            "dr_decay_upper_cap_C": float(
                dr_decay_upper_cap_nodes[k]
            ),
            "dr_exit_floor_C": float(DR_EXIT_FLOOR_C),
            "dr_exit_target_C": float(DR_EXIT_TARGET_C),
            "post_dr_recovery_reference_node": bool(
                post_dr_recovery_nodes[k]
            ),
            "post_dr_recovery_fraction": float(
                post_dr_recovery_fraction_nodes[k]
            ),
            "preheat_window_node": bool(is_preheat_nodes[k]),
            "preheat_search_window_node": bool(is_preheat_nodes[k]),
            "rule_seed_preheat_node": bool(is_rule_seed_preheat_nodes[k]),
            "hours_to_next_dr_start": (
                float(hours_to_next_dr_start_t[k])
                if np.isfinite(hours_to_next_dr_start_t[k])
                else np.nan
            ),
            "night_setback_node": bool(is_night_nodes[k]),
            "setpoint_forced_by_night_setback": False,
            "lower_comfort_bound_mean_C": float(np.mean(T_low_t[k])),
            "soft_upper_temperature_mean_C": float(np.mean(T_soft_upper_t[k])),
            "upper_comfort_bound_mean_C": float(np.mean(T_high_t[k])),
            "temperature_excess_weight": float(w_temp_excess_t[k]),
            "preferred_temperature_mean_C": float(np.mean(T_low_t[k])),  # backward-compatible column name
            "P_node_total_kW": float(np.sum(P_kW_nodes[k])),
            "objective": float(solve_result["obj"]),
            "optimized_horizon_energy_cost_$": float(
                solve_result["predicted_energy_cost_dollars"]
            ),
            "selected_predicted_total_heating_energy_kWh": float(
                solve_result[
                    "selected_predicted_total_heating_energy_kWh"
                ]
            ),
            "selected_predicted_dr_event_energy_kWh": float(
                solve_result[
                    "selected_predicted_dr_event_energy_kWh"
                ]
            ),
            "rule_predicted_total_heating_energy_kWh": float(
                solve_result[
                    "rule_predicted_total_heating_energy_kWh"
                ]
            ),
            "rule_predicted_dr_event_energy_kWh": float(
                solve_result["rule_predicted_dr_event_energy_kWh"]
            ),
            "selected_candidate_name": solve_result["selected_candidate_name"],
            "selection_reason": solve_result["selection_reason"],
            "selected_candidate_feasible": bool(
                solve_result["selected_candidate_feasible"]
            ),
            "selected_mean_event_exit_temperature_C": float(
                solve_result["selected_mean_event_exit_temperature_C"]
            ),
            "selected_max_event_exit_temperature_C": float(
                solve_result["selected_max_event_exit_temperature_C"]
            ),
            "selected_mean_event_exit_band_error_C": float(
                solve_result["selected_mean_event_exit_band_error_C"]
            ),
            "selected_max_event_exit_band_error_C": float(
                solve_result["selected_max_event_exit_band_error_C"]
            ),
            "selected_max_event_exit_excess_C": float(
                solve_result["selected_max_event_exit_excess_C"]
            ),
            "rule_horizon_energy_cost_$": float(
                solve_result["rule_predicted_cost_dollars"]
            ),
            "predicted_savings_vs_rule_$": float(
                solve_result["predicted_savings_vs_rule_dollars"]
            ),
            "required_savings_vs_rule_$": float(
                solve_result["required_savings_dollars"]
            ),
            "preheat_visible_in_horizon": bool(
                solve_result["preheat_visible_in_horizon"]
            ),
            "preheat_actionable_now": bool(
                solve_result["preheat_actionable_now"]
            ),
            "multistart_seed_count": int(
                solve_result["multistart_seed_count"]
            ),
            "ilqr_solve_count": int(solve_result["ilqr_solve_count"]),
            "executed_seed_names": "|".join(
                solve_result["executed_seed_names"]
            ),
            "full_multistart_refresh": bool(
                solve_result["full_multistart_refresh"]
            ),
            "adaptive_rescue_triggered": bool(
                solve_result["adaptive_rescue_triggered"]
            ),
            "mpc_solve_wall_seconds": float(
                solve_result["solve_wall_seconds"]
            ),
        }

        for i, zone_name in enumerate(ZONE_ORDER):
            safe_zone = zone_name.lower().replace(" ", "_")
            row[f"Tpred_{safe_zone}_degC"] = float(T_deg_nodes[k, i])
            row[f"Tpred_bias_corrected_{safe_zone}_degC"] = float(
                T_expected_nodes[k, i]
            )
            row[f"Tpred_floor_conservative_{safe_zone}_degC"] = float(
                T_floor_conservative_nodes[k, i]
            )
            row[f"temperature_expected_bias_{safe_zone}_C"] = float(
                expected_bias_t[k, i]
            )
            row[f"temperature_floor_bias_{safe_zone}_C"] = float(
                floor_bias_t[k, i]
            )
            row[f"Pnode_{safe_zone}_kW"] = float(P_kW_nodes[k, i])
            row[f"lower_comfort_bound_{safe_zone}_degC"] = float(T_low_t[k, i])
            row[f"soft_upper_temperature_{safe_zone}_degC"] = float(T_soft_upper_t[k, i])
            row[f"upper_comfort_bound_{safe_zone}_degC"] = float(T_high_t[k, i])
            row[f"preferred_T_{safe_zone}_degC"] = float(T_low_t[k, i])  # backward-compatible column name
            row[f"baseline_sp_{safe_zone}_degC"] = float(baseline_setpoint_t[k, i])
            row[f"stored_heat_excess_{safe_zone}_C"] = (
                float(
                    max(
                        T_deg_nodes[k, i]
                        - dr_decay_upper_cap_nodes[k],
                        0.0,
                    )
                )
                if bool(is_dr_nodes[k])
                else np.nan
            )
            row[f"event_exit_excess_{safe_zone}_C"] = (
                float(
                    max(
                        T_deg_nodes[k, i] - float(DR_EXIT_TARGET_C),
                        0.0,
                    )
                )
                if bool(is_dr_end_nodes[k])
                else np.nan
            )
            row[f"optimized_preheat_duration_{safe_zone}_h"] = float(
                optimized_preheat_duration_h[i]
            )
            row[f"optimized_preheat_max_sp_{safe_zone}_degC"] = float(
                optimized_preheat_max_setpoint_C[i]
            )

        if k < T_steps:
            row["interval_start_local"] = pd.Timestamp(horizon_datetimes[k]).isoformat()
            row["interval_end_local"] = pd.Timestamp(horizon_datetimes[k + 1]).isoformat()
            row["interval_peak_event"] = bool(is_peak_event_step[k])
            row["interval_avg_power_total_kW"] = float(np.sum(interval_avg_power_kW[k]))
            row["interval_energy_total_kWh_model"] = float(interval_energy_kWh[k])
            row["interval_flex_cost_model_$"] = float(interval_cost_dollars[k])

            for i, zone_name in enumerate(ZONE_ORDER):
                safe_zone = zone_name.lower().replace(" ", "_")
                row[f"sp_{safe_zone}_degC"] = float(sp_deg_intervals[k, i])
                row[f"preheat_increment_{safe_zone}_degC"] = float(
                    preheat_increment_degC[k, i]
                )
                row[f"optimized_preheat_active_{safe_zone}"] = bool(
                    optimized_preheat_active[k, i]
                )
                row[f"Pavg_{safe_zone}_kW_model"] = float(interval_avg_power_kW[k][i])
                row[f"interval_energy_{safe_zone}_kWh_model"] = float(
                    interval_avg_power_kW[k][i] * float(dt)
                )
        else:
            row["interval_start_local"] = None
            row["interval_end_local"] = None
            row["interval_peak_event"] = None
            row["interval_avg_power_total_kW"] = np.nan
            row["interval_energy_total_kWh_model"] = np.nan
            row["interval_flex_cost_model_$"] = np.nan
            for zone_name in ZONE_ORDER:
                safe_zone = zone_name.lower().replace(" ", "_")
                row[f"sp_{safe_zone}_degC"] = np.nan
                row[f"preheat_increment_{safe_zone}_degC"] = np.nan
                row[f"optimized_preheat_active_{safe_zone}"] = None
                row[f"Pavg_{safe_zone}_kW_model"] = np.nan
                row[f"interval_energy_{safe_zone}_kWh_model"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def save_prediction_csv(run_idx, solve_result):
    df = build_prediction_dataframe(run_idx, solve_result)
    stamp = pd.Timestamp(solve_result["apply_dt"]).strftime("%Y%m%d_%H%M")
    path = Path(LOG_DIR) / f"mpc_prediction_run_{run_idx:03d}_{stamp}.csv"
    df.to_csv(path, index=False)

    candidate_rows = []
    for candidate in solve_result.get("candidate_summary", []):
        row = {
            "run_idx": int(run_idx),
            "apply_time_local": pd.Timestamp(
                solve_result["apply_dt"]
            ).isoformat(),
            "selected_candidate_name": solve_result[
                "selected_candidate_name"
            ],
            "selection_reason": solve_result["selection_reason"],
            "rule_predicted_cost_dollars": float(
                solve_result["rule_predicted_cost_dollars"]
            ),
            "required_savings_dollars": float(
                solve_result["required_savings_dollars"]
            ),
            "mpc_solve_wall_seconds": float(
                solve_result["solve_wall_seconds"]
            ),
        }
        row.update(candidate)
        candidate_rows.append(row)

    candidate_path = (
        Path(LOG_DIR)
        / f"mpc_candidate_summary_run_{run_idx:03d}_{stamp}.csv"
    )
    pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)
    solve_result["candidate_summary_csv"] = str(candidate_path)
    return str(path), df


def _timestamp_key_utc(series):
    """Return timezone-normalized UTC timestamps for robust CSV merging."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def build_measured_interval_lookup_from_plant_log(plant_log_df):
    """
    Build a compact interval-level measurement table from energyplus_plant_log.csv.

    This table is later merged into every full MPC rollout, so each
    predicted interval can be compared with the corresponding measured
    EnergyPlus interval when that interval exists in the simulated day.
    """
    df = plant_log_df.copy()
    if "interval_start_local" not in df.columns or "interval_end_local" not in df.columns:
        raise ValueError("Plant log must contain interval_start_local and interval_end_local columns.")

    df["__interval_start_utc"] = _timestamp_key_utc(df["interval_start_local"])
    df["__interval_end_utc"] = _timestamp_key_utc(df["interval_end_local"])

    interval_hours = float(CONTROL_STEP / pd.Timedelta(hours=1))

    measured = pd.DataFrame({
        "__interval_start_utc": df["__interval_start_utc"],
        "__interval_end_utc": df["__interval_end_utc"],
        "measured_interval_start_local": df["interval_start_local"],
        "measured_interval_end_local": df["interval_end_local"],
        "measured_total_interval_energy_kWh": pd.to_numeric(
            df.get("ep_interval_energy_kWh", np.nan), errors="coerce"
        ),
        "measured_total_cumulative_energy_kWh": pd.to_numeric(
            df.get("ep_cumulative_heating_kWh", np.nan), errors="coerce"
        ),
        "measured_sum_zone_interval_energy_kWh": pd.to_numeric(
            df.get("ep_sum_zone_interval_energy_kWh", np.nan), errors="coerce"
        ),
        "measured_sum_zone_cumulative_energy_kWh": pd.to_numeric(
            df.get("ep_sum_zone_cumulative_heating_kWh", np.nan), errors="coerce"
        ),
        "measured_meter_minus_sum_zone_interval_energy_kWh": pd.to_numeric(
            df.get("ep_meter_minus_sum_zone_interval_energy_kWh", np.nan), errors="coerce"
        ),
    })

    measured["measured_total_interval_avg_power_kW"] = (
        measured["measured_total_interval_energy_kWh"] / interval_hours
    )
    measured["measured_sum_zone_interval_avg_power_kW"] = (
        measured["measured_sum_zone_interval_energy_kWh"] / interval_hours
    )

    for zone_name in ZONE_ORDER:
        safe_zone = zone_name.lower().replace(" ", "_")
        energy_col = f"ep_interval_energy_{safe_zone}_kWh"
        cumulative_col = f"ep_cumulative_heating_{safe_zone}_kWh"
        rate_col = f"ep_heating_rate_{safe_zone}_W"
        source_col = f"ep_zone_energy_source_{safe_zone}"

        measured[f"measured_interval_energy_{safe_zone}_kWh"] = pd.to_numeric(
            df[energy_col] if energy_col in df.columns else np.nan,
            errors="coerce",
        )
        measured[f"measured_cumulative_energy_{safe_zone}_kWh"] = pd.to_numeric(
            df[cumulative_col] if cumulative_col in df.columns else np.nan,
            errors="coerce",
        )
        measured[f"measured_interval_avg_power_{safe_zone}_kW"] = (
            measured[f"measured_interval_energy_{safe_zone}_kWh"] / interval_hours
        )
        measured[f"measured_ep_reported_rate_{safe_zone}_W"] = pd.to_numeric(
            df[rate_col] if rate_col in df.columns else np.nan,
            errors="coerce",
        )
        measured[f"measured_ep_reported_rate_{safe_zone}_kW"] = (
            measured[f"measured_ep_reported_rate_{safe_zone}_W"] / 1000.0
        )
        measured[f"measured_energy_source_{safe_zone}"] = (
            df[source_col].astype(str) if source_col in df.columns else "missing"
        )

    return measured


def enrich_prediction_rollouts_with_measurements(log_dir=LOG_DIR):
    """
    Add measured EnergyPlus power/energy to every full MPC rollout CSV.

    EnergyPlus measurements for future rollout steps are only available after the
    simulation has completed. Therefore this function is called after
    run_energyplus() finishes. It creates:
      - mpc_prediction_vs_measurement_run_XXX_YYYYMMDD_HHMM.csv
      - mpc_prediction_vs_measurement_all_rollouts.csv

    Rows whose predicted interval lies outside the simulated plant log keep NaN
    measured values.
    """
    if not LOG_FULL_ROLLOUT_POWER_ENERGY_MEASUREMENTS:
        return []

    log_dir = Path(log_dir)
    plant_log_path = log_dir / "energyplus_plant_log.csv"
    if not plant_log_path.exists():
        print(f"WARNING: cannot enrich rollouts because plant log is missing: {plant_log_path}")
        return []

    plant_log_df = pd.read_csv(plant_log_path)
    measured_lookup = build_measured_interval_lookup_from_plant_log(plant_log_df)

    enriched_paths = []
    all_enriched = []

    pred_paths = sorted(log_dir.glob("mpc_prediction_run_*.csv"))
    for pred_path in pred_paths:
        pred_df = pd.read_csv(pred_path)
        if "interval_start_local" not in pred_df.columns or "interval_end_local" not in pred_df.columns:
            continue

        pred_df["__interval_start_utc"] = _timestamp_key_utc(pred_df["interval_start_local"])
        pred_df["__interval_end_utc"] = _timestamp_key_utc(pred_df["interval_end_local"])

        enriched = pred_df.merge(
            measured_lookup,
            on=["__interval_start_utc", "__interval_end_utc"],
            how="left",
        )

        # Error columns: model minus measured, using interval-average quantities.
        if "interval_avg_power_total_kW" in enriched.columns:
            enriched["error_interval_avg_power_total_kW_model_minus_measured"] = (
                pd.to_numeric(enriched["interval_avg_power_total_kW"], errors="coerce")
                - pd.to_numeric(enriched["measured_total_interval_avg_power_kW"], errors="coerce")
            )
        if "interval_energy_total_kWh_model" in enriched.columns:
            enriched["error_interval_energy_total_kWh_model_minus_measured"] = (
                pd.to_numeric(enriched["interval_energy_total_kWh_model"], errors="coerce")
                - pd.to_numeric(enriched["measured_total_interval_energy_kWh"], errors="coerce")
            )

        for zone_name in ZONE_ORDER:
            safe_zone = zone_name.lower().replace(" ", "_")
            pred_power_col = f"Pavg_{safe_zone}_kW_model"
            pred_energy_col = f"interval_energy_{safe_zone}_kWh_model"
            meas_power_col = f"measured_interval_avg_power_{safe_zone}_kW"
            meas_energy_col = f"measured_interval_energy_{safe_zone}_kWh"

            if pred_power_col in enriched.columns and meas_power_col in enriched.columns:
                enriched[f"error_Pavg_{safe_zone}_kW_model_minus_measured"] = (
                    pd.to_numeric(enriched[pred_power_col], errors="coerce")
                    - pd.to_numeric(enriched[meas_power_col], errors="coerce")
                )
            if pred_energy_col in enriched.columns and meas_energy_col in enriched.columns:
                enriched[f"error_interval_energy_{safe_zone}_kWh_model_minus_measured"] = (
                    pd.to_numeric(enriched[pred_energy_col], errors="coerce")
                    - pd.to_numeric(enriched[meas_energy_col], errors="coerce")
                )

        enriched = enriched.drop(columns=["__interval_start_utc", "__interval_end_utc"], errors="ignore")

        out_name = pred_path.name.replace("mpc_prediction_run_", "mpc_prediction_vs_measurement_run_")
        out_path = log_dir / out_name
        enriched.to_csv(out_path, index=False)
        enriched_paths.append(str(out_path))
        all_enriched.append(enriched)

    if all_enriched:
        all_df = pd.concat(all_enriched, ignore_index=True)
        all_path = log_dir / "mpc_prediction_vs_measurement_all_rollouts.csv"
        all_df.to_csv(all_path, index=False)
        enriched_paths.append(str(all_path))

    print(f"Full-rollout prediction/measurement CSVs written: {len(enriched_paths)}")
    return enriched_paths


def append_action_log(action_log_path, row_dict):
    df_row = pd.DataFrame([row_dict])
    action_log_path = Path(action_log_path)
    if action_log_path.exists():
        df_row.to_csv(action_log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(action_log_path, mode="w", header=True, index=False)


# 9) EnergyPlus runtime controller
def current_process_rss_gb():
    """Return current resident memory from /proc, or NaN when unavailable."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        pass
    return float("nan")


def clear_mpc_runtime_caches(run_number):
    """Periodically release Python and JAX compilation-cache memory."""
    if MPC_CACHE_CLEAR_INTERVAL_RUNS <= 0:
        return
    if run_number % int(MPC_CACHE_CLEAR_INTERVAL_RUNS) != 0:
        return

    try:
        import gc
        gc.collect()
        jax.clear_caches()
        print(
            f"Cleared JAX/Python caches after MPC run {run_number}; "
            f"process RSS={current_process_rss_gb():.2f} GiB"
        )
    except Exception as exc:
        print(f"WARNING: MPC cache cleanup failed: {exc}")


@dataclass
class EnergyPlusMPCPlantController:
    api: object
    weather_df: pd.DataFrame
    sim_start_dt: pd.Timestamp
    sim_end_dt: pd.Timestamp
    initial_zone_temperatures_degC: Optional[np.ndarray] = None
    initial_temperature_record: Optional[dict] = None
    initialized: bool = False
    warmup_complete: bool = False
    handles_ready: bool = False
    weather_handles_ready: bool = False
    weather_warning_printed: bool = False

    temp_handles: Dict[str, int] = field(default_factory=dict)
    heating_rate_handles: Dict[str, int] = field(default_factory=dict)
    heating_energy_handles: Dict[str, int] = field(default_factory=dict)
    heating_energy_sources: Dict[str, str] = field(default_factory=dict)
    setpoint_handles: Dict[str, int] = field(default_factory=dict)
    unmanaged_setpoint_handles: Dict[str, int] = field(default_factory=dict)
    weather_handles: Dict[str, int] = field(default_factory=dict)
    initial_temperature_actuator_handles: Dict[str, int] = field(default_factory=dict)
    initial_temperature_actuator_specs: Dict[str, Tuple[str, str, str]] = field(default_factory=dict)
    initial_temperatures_applied_to_ep: bool = False
    initial_temperature_actuators_need_reset: bool = False
    initial_temperature_warning_printed: bool = False
    csv_initial_match_validated: bool = False
    csv_initial_max_abs_error_C: float = np.nan
    meter_handle: int = -1
    meter_name: Optional[str] = None

    current_sp_deg: np.ndarray = field(default_factory=lambda: np.ones((NUM_ZONES,), dtype=np.float32) * DEFAULT_INITIAL_SETPOINT_C)
    last_T_degC: Optional[np.ndarray] = None
    last_P_scaled: np.ndarray = field(default_factory=lambda: np.ones((NUM_ZONES,), dtype=np.float32) * DEFAULT_INITIAL_POWER_SCALED)
    U0_warm: Optional[jnp.ndarray] = None

    current_model_first_pavg_zone_kW: np.ndarray = field(
        default_factory=lambda: np.full((NUM_ZONES,), np.nan, dtype=np.float64)
    )
    current_model_first_energy_zone_kWh: np.ndarray = field(
        default_factory=lambda: np.full((NUM_ZONES,), np.nan, dtype=np.float64)
    )
    current_model_first_pavg_total_kW: float = np.nan
    current_model_first_energy_total_kWh: float = np.nan

    next_control_dt: pd.Timestamp = None
    run_idx: int = 0
    plant_log_idx: int = 0
    preconditioning_control_idx: int = 0
    preconditioning_log_idx: int = 0

    eoff_cum_kwh: float = float(EOFF_CUM0_KWH)
    cumulative_ep_heating_kwh: float = 0.0
    cumulative_ep_dr_heating_kwh: float = 0.0
    cumulative_ep_zone_heating_kwh: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_ZONES,), dtype=np.float64)
    )

    last_control_step_key: Optional[Tuple] = None
    last_log_step_key: Optional[Tuple] = None
    last_preconditioning_control_step_key: Optional[Tuple] = None
    last_preconditioning_log_step_key: Optional[Tuple] = None

    def __post_init__(self):
        self.exchange = self.api.exchange
        self.runtime = self.api.runtime
        self.next_control_dt = control_grid_dt(0)

    def request_variables(self, state):
        # Must be called before run_energyplus.
        for zone_name in ZONE_ORDER:
            ep_zone = EP_ZONE_NAMES[zone_name]
            for var_name in ZONE_TEMPERATURE_VARIABLE_CANDIDATES:
                self.exchange.request_variable(state, var_name, ep_zone)
            self.exchange.request_variable(state, "Zone Thermostat Heating Setpoint Temperature", ep_zone)
            for var_name in ZONE_HEATING_RATE_VARIABLE_CANDIDATES:
                # The key for Baseboard Electricity Rate may be the zone name or
                # the generated baseboard object name, exact handles are resolved
                # later from available_api_data.csv when possible.
                self.exchange.request_variable(state, var_name, ep_zone)
            for var_name in ZONE_HEATING_ENERGY_VARIABLE_CANDIDATES:
                # Exact interval energy is preferred when EnergyPlus exposes it.
                # Some equipment variables are keyed by the generated equipment
                # name rather than the zone name, exact handles are resolved later
                # from available_api_data.csv when possible.
                self.exchange.request_variable(state, var_name, ep_zone)

    def sim_datetime(self, state):
        # current_sim_time is hours from start of environment.
        h = float(self.exchange.current_sim_time(state))
        return (pd.Timestamp(self.sim_start_dt).tz_convert(OPEN_METEO_TIMEZONE) + pd.Timedelta(hours=h))

    def apply_matched_preconditioning_setpoints(self, state, grid_dt):
        """Apply one interval of the no-MPC rule-based preconditioning schedule."""
        self.current_sp_deg = matched_preconditioning_setpoint_vector_at_datetime(
            grid_dt
        )
        self.apply_current_setpoints(state)

    def validate_preconditioned_state_against_csv(self, T_ep):
        """Log and enforce the 00:00 EnergyPlus-versus-CSV temperature match."""
        if self.csv_initial_match_validated:
            return
        if self.initial_zone_temperatures_degC is None:
            if REQUIRE_CSV_INITIAL_TEMPERATURES:
                raise RuntimeError(
                    "CSV initial temperatures are required, but no row was loaded for this DR day."
                )
            return

        T_ep = np.asarray(T_ep, dtype=np.float32)
        T_csv = np.asarray(self.initial_zone_temperatures_degC, dtype=np.float32)
        abs_error = np.abs(T_ep - T_csv)
        max_error = float(np.max(abs_error))
        within_tolerance = bool(max_error <= float(INITIAL_TEMPERATURE_MATCH_TOLERANCE_C))

        row = {
            "control_start_local": pd.Timestamp(SIM_START_DT).tz_convert(
                OPEN_METEO_TIMEZONE
            ).isoformat(),
            "csv_path": (
                self.initial_temperature_record.get("csv_path")
                if isinstance(self.initial_temperature_record, dict)
                else None
            ),
            "csv_source_time_local": (
                self.initial_temperature_record.get("initial_time_local")
                if isinstance(self.initial_temperature_record, dict)
                else None
            ),
            "tolerance_C": float(INITIAL_TEMPERATURE_MATCH_TOLERANCE_C),
            "max_abs_error_C": max_error,
            "within_tolerance": within_tolerance,
        }
        for i, zone_name in enumerate(ZONE_ORDER):
            safe_zone = zone_name.lower().replace(" ", "_")
            row[f"T_csv_{safe_zone}_degC"] = float(T_csv[i])
            row[f"T_ep_preconditioned_{safe_zone}_degC"] = float(T_ep[i])
            row[f"abs_error_{safe_zone}_degC"] = float(abs_error[i])

        pd.DataFrame([row]).to_csv(
            Path(LOG_DIR) / "initial_temperature_match_check.csv",
            index=False,
        )

        self.csv_initial_match_validated = True
        self.csv_initial_max_abs_error_C = max_error
        print(
            "00:00 initial-temperature match check: "
            f"max |EnergyPlus - CSV| = {max_error:.3f} degC "
            f"(tolerance {float(INITIAL_TEMPERATURE_MATCH_TOLERANCE_C):.3f} degC)."
        )

        if REQUIRE_PRECONDITIONED_EP_CSV_MATCH and not within_tolerance:
            raise RuntimeError(
                "The preconditioned MPC EnergyPlus state does not match the no-MPC "
                f"CSV at 00:00. Maximum absolute zone error is {max_error:.3f} degC, "
                f"greater than the {float(INITIAL_TEMPERATURE_MATCH_TOLERANCE_C):.3f} degC "
                "tolerance. See initial_temperature_match_check.csv."
            )

    def ep_zone_timestep_key(self, state):
        """
        Unique key for the current EnergyPlus zone timestep.

        This prevents duplicate MPC solves or duplicate log rows if EnergyPlus calls
        the same callback more than once during the same zone timestep.
        """
        try:
            return (
                int(self.exchange.current_environment_num(state)),
                int(self.exchange.day_of_year(state)),
                int(self.exchange.hour(state)),
                int(self.exchange.zone_time_step_number(state)),
            )
        except Exception:
            # Fallback if some clock functions are unavailable.
            return ("sim_h", round(float(self.exchange.current_sim_time(state)), 9))


    def is_new_control_zone_timestep(self, state) -> bool:
        key = self.ep_zone_timestep_key(state)
        if key == self.last_control_step_key:
            return False
        self.last_control_step_key = key
        return True


    def is_new_log_zone_timestep(self, state) -> bool:
        key = self.ep_zone_timestep_key(state)
        if key == self.last_log_step_key:
            return False
        self.last_log_step_key = key
        return True

    def dump_api_data(self, state):
        try:
            raw = self.exchange.list_available_api_data_csv(state)
            path = Path(LOG_DIR) / "available_api_data.csv"
            path.write_bytes(raw)
            print(f"Available EnergyPlus API data written to: {path}")
        except Exception as exc:
            print(f"Could not dump available API data: {exc}")

    def get_first_variable_handle(self, state, variable_names, keys):
        for var_name in variable_names:
            for key in keys:
                if not key:
                    continue
                h = self.exchange.get_variable_handle(state, var_name, key)
                if h >= 0:
                    return h, var_name, key
        return -1, None, None

    def find_variable_handle_from_api_csv(self, state, variable_name_tokens, key_tokens):
        """
        Best-effort lookup in available_api_data.csv for cases where the output
        variable key is the generated equipment name rather than the zone name.
        """
        path = Path(LOG_DIR) / "available_api_data.csv"
        if not path.exists():
            return -1, None, None
        text = path.read_text(encoding="utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)

        variable_name_tokens = [t.lower() for t in variable_name_tokens]
        key_tokens = [t.lower() for t in key_tokens]

        for row in rows:
            vals = {str(k).lower(): str(v) for k, v in row.items()}
            all_text = " ".join(vals.values()).lower()
            if "outputvariable" not in all_text and "variable" not in all_text:
                continue
            if not all(tok in all_text for tok in variable_name_tokens):
                continue
            if not any(tok in all_text for tok in key_tokens):
                continue

            # Common EnergyPlus API CSV column names vary by version. Try all
            # likely columns for variable name and key.
            candidate_names = [
                vals.get("name", ""),
                vals.get("variable name", ""),
                vals.get("what", ""),
            ]
            candidate_keys = [
                vals.get("key", ""),
                vals.get("key value", ""),
                vals.get("name", ""),
            ]
            for var_name in candidate_names:
                for key in candidate_keys:
                    if not var_name or not key:
                        continue
                    h = self.exchange.get_variable_handle(state, var_name, key)
                    if h >= 0:
                        return h, var_name, key
        return -1, None, None

    def get_temperature_handle(self, state, zone_name):
        ep_zone = EP_ZONE_NAMES[zone_name]
        h, var_name, key = self.get_first_variable_handle(
            state,
            ZONE_TEMPERATURE_VARIABLE_CANDIDATES,
            [ep_zone, ep_zone.upper(), ep_zone.lower()],
        )
        return h, var_name, key

    def get_heating_rate_handle(self, state, zone_name):
        ep_zone = EP_ZONE_NAMES[zone_name]
        keys_to_try = [
            ep_zone,
            ep_zone.upper(),
            ep_zone.lower(),
            f"{ep_zone} Baseboard",
            f"{ep_zone} Electric Baseboard",
            f"{ep_zone} ZoneHVAC Baseboard",
        ]
        h, var_name, key = self.get_first_variable_handle(
            state,
            ZONE_HEATING_RATE_VARIABLE_CANDIDATES,
            keys_to_try,
        )
        if h >= 0:
            return h, var_name, key

        # Search the API data dump for a baseboard/electricity rate variable
        # associated with this zone/equipment name.
        return self.find_variable_handle_from_api_csv(
            state,
            variable_name_tokens=["baseboard", "electricity", "rate"],
            key_tokens=[ep_zone.lower(), zone_name.lower()],
        )

    def get_heating_energy_handle(self, state, zone_name):
        ep_zone = EP_ZONE_NAMES[zone_name]
        keys_to_try = [
            ep_zone,
            ep_zone.upper(),
            ep_zone.lower(),
            f"{ep_zone} Baseboard",
            f"{ep_zone} Electric Baseboard",
            f"{ep_zone} ZoneHVAC Baseboard",
        ]
        h, var_name, key = self.get_first_variable_handle(
            state,
            ZONE_HEATING_ENERGY_VARIABLE_CANDIDATES,
            keys_to_try,
        )
        if h >= 0:
            return h, var_name, key

        # Search the API data dump for an exact per-zone/per-equipment energy
        # variable. This is preferred over rate integration because it is the
        # actual EnergyPlus reporting-timestep energy in joules.
        return self.find_variable_handle_from_api_csv(
            state,
            variable_name_tokens=["baseboard", "electricity", "energy"],
            key_tokens=[ep_zone.lower(), zone_name.lower()],
        )

    def get_initial_temperature_actuator_handle(self, state, zone_name):
        """Best-effort lookup for a zone-air-temperature actuator."""
        ep_zone = EP_ZONE_NAMES[zone_name]
        keys_to_try = [ep_zone, ep_zone.upper(), ep_zone.lower()]

        for comp_type, ctrl_type in ZONE_INITIAL_TEMPERATURE_ACTUATOR_CANDIDATES:
            for key in keys_to_try:
                h = self.exchange.get_actuator_handle(state, comp_type, ctrl_type, key)
                if h >= 0:
                    return h, (comp_type, ctrl_type, key)

        # Best-effort auto-discovery from the API data dump.
        path = Path(LOG_DIR) / "available_api_data.csv"
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            rows = list(csv.DictReader(io.StringIO(text)))
            for row in rows:
                row_l = {str(k).lower(): str(v) for k, v in row.items()}
                all_text = " ".join(row_l.values()).lower()
                if "actuator" not in all_text:
                    continue
                if ep_zone.lower() not in all_text and zone_name.lower() not in all_text:
                    continue
                if not any(tok in all_text for tok in ["temperature", "mean air", "zone air"]):
                    continue

                vals = {str(k).lower(): str(v) for k, v in row.items()}
                candidate_names = [vals.get("component type", ""), vals.get("name", ""), vals.get("type", ""), vals.get("what", "")]
                candidate_ctrls = [vals.get("control type", ""), vals.get("actuator type", ""), vals.get("type", ""), vals.get("what", "")]
                candidate_keys = [vals.get("key", ""), vals.get("key value", ""), vals.get("name", "")]
                for comp_type in candidate_names:
                    for ctrl_type in candidate_ctrls:
                        for key in candidate_keys:
                            if not comp_type or not ctrl_type or not key:
                                continue
                            h = self.exchange.get_actuator_handle(state, comp_type, ctrl_type, key)
                            if h >= 0:
                                return h, (comp_type, ctrl_type, key)

        return -1, None

    def initialize_initial_temperature_actuators(self, state):
        """
        Try to find actuators that can set the zone air temperatures.

        This is optional. EnergyPlus does not always expose a direct actuator for
        zone initial air temperature. If not found, the code still uses the CSV
        temperatures for the first MPC state and logs a warning.
        """
        if not TRY_APPLY_INITIAL_ZONE_TEMPERATURES_TO_ENERGYPLUS:
            return
        if self.initial_zone_temperatures_degC is None:
            return

        for zone_name in ZONE_ORDER:
            h, spec = self.get_initial_temperature_actuator_handle(state, zone_name)
            if h >= 0:
                self.initial_temperature_actuator_handles[zone_name] = h
                self.initial_temperature_actuator_specs[zone_name] = spec
                print(f"Initial-temperature actuator for {zone_name}: {spec}")
            else:
                print(f"WARNING: no zone-air-temperature actuator found for initial condition of {zone_name}.")

    def apply_initial_temperatures_to_ep_if_possible(self, state):
        """
        Push the CSV initial temperatures into EnergyPlus once, at the beginning
        of the first zone timestep, if compatible actuators were found.
        """
        if not TRY_APPLY_INITIAL_ZONE_TEMPERATURES_TO_ENERGYPLUS:
            return
        if self.initial_temperatures_applied_to_ep:
            return
        if self.initial_zone_temperatures_degC is None:
            return

        applied = []
        missing = []
        for i, zone_name in enumerate(ZONE_ORDER):
            h = self.initial_temperature_actuator_handles.get(zone_name, -1)
            if h >= 0:
                self.exchange.set_actuator_value(
                    state,
                    h,
                    float(self.initial_zone_temperatures_degC[i]),
                )
                applied.append(zone_name)
            else:
                missing.append(zone_name)

        self.initial_temperatures_applied_to_ep = True

        if applied:
            self.initial_temperature_actuators_need_reset = True
            print("Applied CSV initial zone temperatures to EnergyPlus actuators for:")
            print(applied)

        if missing and not self.initial_temperature_warning_printed:
            self.initial_temperature_warning_printed = True
            print(
                "WARNING: Could not apply CSV initial temperatures directly to all EnergyPlus zones. "
                "The first MPC state will still use the CSV temperatures, but zones without a "
                "temperature actuator will start from the EnergyPlus warmup state."
            )
            print(f"Zones without initial-temperature actuator: {missing}")

    def release_initial_temperature_actuators_if_needed(self, state):
        """
        Release one-time zone-air-temperature actuators after the first timestep.

        This prevents a successful initial-temperature actuator from holding the
        zone temperature fixed for the whole DR-day simulation.
        """
        if not self.initial_temperature_actuators_need_reset:
            return

        reset_fn = getattr(self.exchange, "reset_actuator", None)
        if reset_fn is None:
            print(
                "WARNING: EnergyPlus API does not expose reset_actuator in this environment. "
                "Initial-temperature actuators may persist; check the plant log."
            )
            self.initial_temperature_actuators_need_reset = False
            return

        for zone_name, h in self.initial_temperature_actuator_handles.items():
            if h >= 0:
                try:
                    reset_fn(state, h)
                except Exception as exc:
                    print(f"WARNING: could not reset initial-temperature actuator for {zone_name}: {exc}")

        self.initial_temperature_actuators_need_reset = False
        print("Released one-time initial-temperature actuators after the first timestep.")

    def initialize_handles(self, state):
        if self.handles_ready:
            return
        if not self.exchange.api_data_fully_ready(state):
            return

        self.dump_api_data(state)

        # Output variable handles.
        for zone_name in ZONE_ORDER:
            ep_zone = EP_ZONE_NAMES[zone_name]
            h_temp, temp_var, temp_key = self.get_temperature_handle(state, zone_name)
            if h_temp < 0:
                raise RuntimeError(
                    f"Could not get a zone air temperature handle for EnergyPlus zone '{ep_zone}'. "
                    f"Tried {ZONE_TEMPERATURE_VARIABLE_CANDIDATES}. "
                    f"Check EP_ZONE_NAMES and {Path(LOG_DIR) / 'available_api_data.csv'}."
                )
            self.temp_handles[zone_name] = h_temp
            print(f"Temperature variable for {zone_name}: ({temp_var}, {temp_key})")

            h_heat, heat_var, heat_key = self.get_heating_rate_handle(state, zone_name)
            self.heating_rate_handles[zone_name] = h_heat  # can be -1, not fatal
            if h_heat >= 0:
                print(f"Heating-rate variable for {zone_name}: ({heat_var}, {heat_key})")
            else:
                print(f"WARNING: no per-zone baseboard/heating-rate variable found for {zone_name}; P0 will use default.")

            h_energy, energy_var, energy_key = self.get_heating_energy_handle(state, zone_name)
            self.heating_energy_handles[zone_name] = h_energy  # can be -1, not fatal
            if h_energy >= 0:
                self.heating_energy_sources[zone_name] = "energy_variable"
                print(f"Heating-energy variable for {zone_name}: ({energy_var}, {energy_key})")
            elif h_heat >= 0:
                self.heating_energy_sources[zone_name] = "rate_integrated"
                print(
                    f"WARNING: no exact per-zone heating-energy variable found for {zone_name}; "
                    "plant log will estimate zone energy from heating rate."
                )
            else:
                self.heating_energy_sources[zone_name] = "missing"
                print(
                    f"WARNING: no per-zone heating-energy or heating-rate variable found for {zone_name}; "
                    "zone energy will be NaN in the plant log."
                )

        # Meter handle.
        for meter in HEATING_METER_CANDIDATES:
            h = self.exchange.get_meter_handle(state, meter)
            if h >= 0:
                self.meter_handle = h
                self.meter_name = meter
                print(f"Using EnergyPlus meter for plant energy: {meter}")
                break
        if self.meter_handle < 0:
            print("WARNING: no heating/electricity meter handle found. Energy logging from EnergyPlus will be NaN.")

        # Setpoint actuator handles for MPC-controlled zones.
        for zone_name in ZONE_ORDER:
            handle = self.get_setpoint_actuator_handle(state, zone_name)
            if handle < 0:
                raise RuntimeError(
                    f"Could not find a setpoint actuator for '{zone_name}'. "
                    f"Tried schedule-value actuation for schedule "
                    f"'{EP_SETPOINT_SCHEDULE_MAP.get(zone_name)}' and direct thermostat actuators. Inspect "
                    f"{Path(LOG_DIR) / 'available_api_data.csv'}."
                )
            self.setpoint_handles[zone_name] = handle

        # Setpoint actuator handles for IDF-only zones held constant.
        for zone_name, schedule_name in UNMANAGED_SETPOINT_SCHEDULE_MAP.items():
            handle = self.try_schedule_actuator(state, schedule_name)
            if handle >= 0:
                self.unmanaged_setpoint_handles[zone_name] = handle
                print(f"Unmanaged setpoint actuator for {zone_name}: schedule '{schedule_name}'")
            else:
                print(f"WARNING: could not actuate unmanaged setpoint schedule '{schedule_name}' for {zone_name}.")

        self.initialize_initial_temperature_actuators(state)

        self.handles_ready = True
        self.last_T_degC = self.read_zone_temperatures(state)
        print("EnergyPlus handles initialized successfully.")
        print("Initial EnergyPlus warmup temperatures [degC]:")
        print(dict(zip(ZONE_ORDER, [float(x) for x in self.last_T_degC])))

        if self.initial_zone_temperatures_degC is not None:
            print("CSV initial temperatures selected for the first MPC state [degC]:")
            print(dict(zip(ZONE_ORDER, [float(x) for x in self.initial_zone_temperatures_degC])))

    def try_schedule_actuator(self, state, schedule_name):
        if not schedule_name:
            return -1
        for comp_type in SCHEDULE_COMPONENT_TYPES_TO_TRY:
            h = self.exchange.get_actuator_handle(state, comp_type, "Schedule Value", schedule_name)
            if h >= 0:
                return h
        return -1

    def get_setpoint_actuator_handle(self, state, zone_name):
        # 1) Preferred IDF schedule name.
        schedule_name = EP_SETPOINT_SCHEDULE_MAP.get(zone_name)
        h = self.try_schedule_actuator(state, schedule_name)
        if h >= 0:
            print(f"Setpoint actuator for {zone_name}: schedule '{schedule_name}'")
            return h

        ep_zone = EP_ZONE_NAMES[zone_name]

        # 2) Direct zone thermostat actuator candidates.
        direct_candidates = [
            ("Zone Temperature Control", "Heating Setpoint", ep_zone),
            ("Zone Temperature Control", "Heating Setpoint Temperature", ep_zone),
            ("ZoneControl:Thermostat", "Heating Setpoint", ep_zone),
            ("ZoneControl:Thermostat", "Heating Setpoint Temperature", ep_zone),
        ]
        for comp_type, ctrl_type, key in direct_candidates:
            h = self.exchange.get_actuator_handle(state, comp_type, ctrl_type, key)
            if h >= 0:
                print(f"Setpoint actuator for {zone_name}: ({comp_type}, {ctrl_type}, {key})")
                return h

        # 3) Best-effort auto-discovery from available_api_data.csv.
        path = Path(LOG_DIR) / "available_api_data.csv"
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
            for row in rows:
                row_l = {str(k).lower(): str(v) for k, v in row.items()}
                all_text = " ".join(row_l.values()).lower()
                if "actuator" not in all_text:
                    continue
                if schedule_name and schedule_name.lower() in all_text:
                    h = self.try_schedule_actuator(state, schedule_name)
                    if h >= 0:
                        return h
                if ep_zone.lower() not in all_text and zone_name.lower() not in all_text:
                    continue
                if not any(token in all_text for token in ["setpoint", "heating", "schedule"]):
                    continue

                vals = {k.lower(): v for k, v in row.items()}
                name = vals.get("name", "")
                key = vals.get("key", "") or vals.get("key value", "")
                typ = vals.get("type", "") or vals.get("what", "")

                specs_to_try = [
                    (name, typ, key),
                    (typ, name, key),
                    (name, key, typ),
                ]
                for comp_type, ctrl_type, act_key in specs_to_try:
                    h = self.exchange.get_actuator_handle(state, comp_type, ctrl_type, act_key)
                    if h >= 0:
                        print(f"Auto-discovered actuator for {zone_name}: ({comp_type}, {ctrl_type}, {act_key})")
                        return h

        return -1

    def initialize_weather_handles(self, state):
        if self.weather_handles_ready or not OVERRIDE_EP_WEATHER_WITH_OPEN_METEO:
            return
        if not self.exchange.api_data_fully_ready(state):
            return

        candidates = {
            "temperature_2m": ("Weather Data", "Outdoor Dry Bulb", "Environment"),
            "direct_radiation": ("Weather Data", "Direct Solar Radiation Rate per Area", "Environment"),
            "diffuse_radiation": ("Weather Data", "Diffuse Solar Radiation Rate per Area", "Environment"),
        }
        ok = True
        for key, spec in candidates.items():
            h = self.exchange.get_actuator_handle(state, *spec)
            self.weather_handles[key] = h
            ok = ok and (h >= 0)

        self.weather_handles_ready = ok
        if ok:
            print("EnergyPlus Weather Data actuators initialized successfully.")
        elif not self.weather_warning_printed:
            self.weather_warning_printed = True
            print(
                "WARNING: Weather Data actuators were not all available. "
                "EnergyPlus will use the EPW weather, but MPC will still use Open-Meteo historical weather."
            )

    def override_weather_if_possible(self, state):
        if not OVERRIDE_EP_WEATHER_WITH_OPEN_METEO:
            return
        self.initialize_weather_handles(state)
        if not self.weather_handles_ready:
            return

        now = self.sim_datetime(state)
        w = weather_at_times(self.weather_df, [now]).iloc[0]

        self.exchange.set_actuator_value(state, self.weather_handles["temperature_2m"], float(w["temperature_2m"]))
        self.exchange.set_actuator_value(state, self.weather_handles["direct_radiation"], float(w["direct_radiation"]))
        self.exchange.set_actuator_value(state, self.weather_handles["diffuse_radiation"], float(w["diffuse_radiation"]))

    def read_zone_temperatures(self, state):
        vals = []
        for zone_name in ZONE_ORDER:
            h = self.temp_handles[zone_name]
            vals.append(float(self.exchange.get_variable_value(state, h)))
        return np.asarray(vals, dtype=np.float32)

    def read_zone_heating_rates_w(self, state):
        """Read per-zone heating rates in W. Missing handles are returned as NaN."""
        rates_w = []
        for zone_name in ZONE_ORDER:
            h = self.heating_rate_handles.get(zone_name, -1)
            if h < 0:
                rates_w.append(np.nan)
            else:
                rates_w.append(max(0.0, float(self.exchange.get_variable_value(state, h))))
        return np.asarray(rates_w, dtype=np.float64)

    def scale_zone_heating_rates_w(self, rates_w):
        """
        Convert physical per-zone rates in W to the scaled NODE/NCDE power state.
        If any rate is missing, fall back to DEFAULT_INITIAL_POWER_SCALED.
        """
        rates_w = np.asarray(rates_w, dtype=np.float32)
        if np.any(~np.isfinite(rates_w)):
            return np.ones((n,), dtype=np.float32) * DEFAULT_INITIAL_POWER_SCALED

        P_min_np = np.asarray(x_scaler.data_min_[n:2 * n], dtype=np.float32)
        P_rng_np = np.asarray(x_scaler.data_range_[n:2 * n], dtype=np.float32) + 1e-6

        P_phys = rates_w / 1000.0 if POWER_STATE_IS_KW else rates_w
        return np.clip((P_phys - P_min_np) / P_rng_np, 0.0, 1.0)

    def read_zone_heating_rates_scaled(self, state):
        # Optional: if handles exist, use zone heating rate to initialize power state.
        return self.scale_zone_heating_rates_w(self.read_zone_heating_rates_w(state))

    def read_zone_interval_heating_energy_kwh(self, state, interval_hours):
        """
        Return per-zone interval heating energy in kWh.

        Preferred source:
          EnergyPlus per-zone/per-equipment heating-energy output variable [J].

        Fallback:
          heating-rate output variable [W] multiplied by interval_hours.

        The fallback is an interval-average approximation and is logged through
        the matching `ep_zone_energy_source_*` columns.
        """
        interval_hours = float(interval_hours)
        rates_w = self.read_zone_heating_rates_w(state)

        energies_kwh = []
        sources = []
        for i, zone_name in enumerate(ZONE_ORDER):
            h_energy = self.heating_energy_handles.get(zone_name, -1)
            if h_energy >= 0:
                val_j = max(0.0, float(self.exchange.get_variable_value(state, h_energy)))
                energies_kwh.append(val_j / 3.6e6)
                sources.append("energy_variable")
            elif np.isfinite(rates_w[i]):
                energies_kwh.append((rates_w[i] / 1000.0) * interval_hours)
                sources.append("rate_integrated")
            else:
                energies_kwh.append(np.nan)
                sources.append("missing")

        return (
            np.asarray(energies_kwh, dtype=np.float64),
            rates_w.astype(np.float64),
            sources,
        )

    def apply_current_setpoints(self, state):
        if not self.handles_ready:
            return
        for i, zone_name in enumerate(ZONE_ORDER):
            self.exchange.set_actuator_value(
                state,
                self.setpoint_handles[zone_name],
                float(self.current_sp_deg[i]),
            )
        for zone_name, handle in self.unmanaged_setpoint_handles.items():
            self.exchange.set_actuator_value(
                state,
                handle,
                float(DEFAULT_UNMANAGED_SETPOINT_C),
            )

    def update_energy_accounting(self, state, accounting_dt=None):
        if self.meter_handle < 0:
            return np.nan

        val_j = float(self.exchange.get_meter_value(state, self.meter_handle))
        val_kwh = max(0.0, val_j / 3.6e6)

        if accounting_dt is None:
            accounting_dt = self.sim_datetime(state).tz_convert(OPEN_METEO_TIMEZONE)
        else:
            accounting_dt = pd.Timestamp(accounting_dt).tz_convert(OPEN_METEO_TIMEZONE)

        is_peak = is_in_any_window(accounting_dt.to_pydatetime(), DAILY_PEAK_WINDOWS)

        if not is_peak:
            self.eoff_cum_kwh += val_kwh
        else:
            self.cumulative_ep_dr_heating_kwh += val_kwh

        self.cumulative_ep_heating_kwh += val_kwh
        return val_kwh

    def solve_and_update_action_if_needed(self, state):
        if not self.handles_ready:
            return

        # One MPC update per EnergyPlus zone timestep.
        if not self.is_new_control_zone_timestep(state):
            return

        if self.run_idx >= TOTAL_MPC_RUNS:
            return

        grid_dt = control_grid_dt(self.run_idx)
        interval_end_dt = grid_dt + CONTROL_STEP

        if grid_dt >= pd.Timestamp(self.sim_end_dt).tz_convert(OPEN_METEO_TIMEZONE):
            return

        # Keep raw EnergyPlus callback time only for debugging.
        ep_callback_dt = self.sim_datetime(state).tz_convert(OPEN_METEO_TIMEZONE)

        T_ep = self.read_zone_temperatures(state)
        if self.run_idx == 0:
            self.validate_preconditioned_state_against_csv(T_ep)

        use_csv_initial_state = (
            self.run_idx == 0
            and self.initial_zone_temperatures_degC is not None
            and USE_CSV_INITIAL_ZONE_TEMPERATURES
        )
        T0 = self.initial_zone_temperatures_degC.copy() if use_csv_initial_state else T_ep
        P0 = self.read_zone_heating_rates_scaled(state)
        sp_prev = self.current_sp_deg.copy()

        print("\n" + "-" * 80)
        print(f"EnergyPlus MPC run {self.run_idx + 1}/{TOTAL_MPC_RUNS}")
        print(f"Control grid time: {grid_dt.isoformat()}")
        print(f"EnergyPlus callback time: {ep_callback_dt.isoformat()}")
        print("Current EnergyPlus warmup/plant temperatures [degC]:")
        print(dict(zip(ZONE_ORDER, [float(x) for x in T_ep])))
        if use_csv_initial_state:
            print("Using CSV initial temperatures for the first MPC state [degC]:")
            print(dict(zip(ZONE_ORDER, [float(x) for x in T0])))

        solve_result = solve_mpc_once_from_ep(
            apply_dt=grid_dt,  # IMPORTANT: use exact 15-min control-grid time
            eoff_cum0_kwh=self.eoff_cum_kwh,
            T0_degC=T0,
            P0_scaled=P0,
            sp_prev_deg=sp_prev,
            weather_df=self.weather_df,
            U0=self.U0_warm,
        )

        first_sp_deg = np.asarray(
            setpoint_for_interval_from_u_raw(solve_result["U_opt"][0], 0, solve_result["profiles"])[1],
            dtype=np.float32,
        )

        self.current_sp_deg = first_sp_deg
        self.U0_warm = shift_warm_start(solve_result["U_opt"])

        csv_path, pred_df = save_prediction_csv(self.run_idx, solve_result)

        # Keep the first applied MPC interval prediction so the plant log can
        # compare the actual EnergyPlus measurement against the prediction that
        # was actually applied during the same interval. Full-horizon rollout
        # prediction/measurement matching is done after the simulation finishes.
        first_interval_pred = pred_df[pred_df["horizon_step"] == 0].iloc[0]
        self.current_model_first_pavg_zone_kW = np.asarray(
            [first_interval_pred.get(f"Pavg_{zone_name.lower().replace(' ', '_')}_kW_model", np.nan)
             for zone_name in ZONE_ORDER],
            dtype=np.float64,
        )
        self.current_model_first_energy_zone_kWh = np.asarray(
            [first_interval_pred.get(f"interval_energy_{zone_name.lower().replace(' ', '_')}_kWh_model", np.nan)
             for zone_name in ZONE_ORDER],
            dtype=np.float64,
        )
        self.current_model_first_pavg_total_kW = float(
            first_interval_pred.get("interval_avg_power_total_kW", np.nan)
        )
        self.current_model_first_energy_total_kWh = float(
            first_interval_pred.get("interval_energy_total_kWh_model", np.nan)
        )

        action_row = {
            "run_idx": int(self.run_idx),

            # Official MPC/control-grid time
            "control_time_local": grid_dt.isoformat(),
            "control_interval_start_local": grid_dt.isoformat(),
            "control_interval_end_local": interval_end_dt.isoformat(),

            # Raw EnergyPlus callback time kept only for debugging
            "ep_callback_time_local": ep_callback_dt.isoformat(),

            "objective": float(solve_result["obj"]),
            "predicted_horizon_energy_cost_$": float(
                solve_result["predicted_energy_cost_dollars"]
            ),
            "selected_predicted_total_heating_energy_kWh": float(
                solve_result[
                    "selected_predicted_total_heating_energy_kWh"
                ]
            ),
            "selected_predicted_dr_event_energy_kWh": float(
                solve_result[
                    "selected_predicted_dr_event_energy_kWh"
                ]
            ),
            "rule_predicted_total_heating_energy_kWh": float(
                solve_result[
                    "rule_predicted_total_heating_energy_kWh"
                ]
            ),
            "rule_predicted_dr_event_energy_kWh": float(
                solve_result["rule_predicted_dr_event_energy_kWh"]
            ),
            "selected_candidate_name": solve_result["selected_candidate_name"],
            "selection_reason": solve_result["selection_reason"],
            "selected_candidate_feasible": bool(
                solve_result["selected_candidate_feasible"]
            ),
            "selected_mean_event_exit_temperature_C": float(
                solve_result["selected_mean_event_exit_temperature_C"]
            ),
            "selected_max_event_exit_temperature_C": float(
                solve_result["selected_max_event_exit_temperature_C"]
            ),
            "selected_mean_event_exit_band_error_C": float(
                solve_result["selected_mean_event_exit_band_error_C"]
            ),
            "selected_max_event_exit_band_error_C": float(
                solve_result["selected_max_event_exit_band_error_C"]
            ),
            "selected_max_event_exit_excess_C": float(
                solve_result["selected_max_event_exit_excess_C"]
            ),
            "rule_predicted_horizon_energy_cost_$": float(
                solve_result["rule_predicted_cost_dollars"]
            ),
            "predicted_savings_vs_rule_$": float(
                solve_result["predicted_savings_vs_rule_dollars"]
            ),
            "required_savings_vs_rule_$": float(
                solve_result["required_savings_dollars"]
            ),
            "preheat_visible_in_horizon": bool(
                solve_result["preheat_visible_in_horizon"]
            ),
            "preheat_actionable_now": bool(
                solve_result["preheat_actionable_now"]
            ),
            "multistart_seed_count": int(
                solve_result["multistart_seed_count"]
            ),
            "ilqr_solve_count": int(solve_result["ilqr_solve_count"]),
            "executed_seed_names": "|".join(
                solve_result["executed_seed_names"]
            ),
            "full_multistart_refresh": bool(
                solve_result["full_multistart_refresh"]
            ),
            "adaptive_rescue_triggered": bool(
                solve_result["adaptive_rescue_triggered"]
            ),
            "mpc_solve_wall_seconds": float(
                solve_result["solve_wall_seconds"]
            ),
            "candidate_summary_csv": str(
                Path(solve_result["candidate_summary_csv"]).resolve()
            ),
            "prediction_csv": str(Path(csv_path).resolve()),
            "ep_meter_name": self.meter_name,
            "ep_cumulative_heating_kWh": float(self.cumulative_ep_heating_kwh),
            "ep_cumulative_dr_heating_kWh": float(
                self.cumulative_ep_dr_heating_kwh
            ),
            "eoff_cum_kWh": float(self.eoff_cum_kwh),
            "dr_event_active": bool(is_in_csv_dr_event(grid_dt)),
            "preheat_search_window_active": bool(is_preheat_window(grid_dt)),
            "hours_to_next_dr_start": float(hours_until_next_dr_start(grid_dt)),
            "night_setback_active": bool(is_night_setback(grid_dt)),
            "mpc_initial_temperature_source": "csv_initial_temperatures" if use_csv_initial_state else "energyplus_runtime",
            "initial_temperatures_applied_to_ep": bool(self.initial_temperatures_applied_to_ep),
        }

        for i, zone_name in enumerate(ZONE_ORDER):
            safe_zone = zone_name.lower().replace(" ", "_")
            action_row[f"T_mpc_initial_{safe_zone}_degC"] = float(T0[i])
            action_row[f"T_ep_runtime_{safe_zone}_degC"] = float(T_ep[i])
            action_row[f"applied_sp_{safe_zone}_degC"] = float(first_sp_deg[i])
            action_row[f"optimized_preheat_duration_{safe_zone}_h"] = float(
                first_interval_pred.get(
                    f"optimized_preheat_duration_{safe_zone}_h",
                    np.nan,
                )
            )
            action_row[f"optimized_preheat_max_sp_{safe_zone}_degC"] = float(
                first_interval_pred.get(
                    f"optimized_preheat_max_sp_{safe_zone}_degC",
                    np.nan,
                )
            )

        append_action_log(Path(LOG_DIR) / "mpc_action_log.csv", action_row)

        print(f"MPC objective: {float(solve_result['obj']):.6f}")
        print(
            "Predicted Flex D energy cost over horizon: "
            f"${float(solve_result['predicted_energy_cost_dollars']):.4f}"
        )
        print(
            "Selected candidate: "
            f"{solve_result['selected_candidate_name']} "
            f"({solve_result['selection_reason']})"
        )
        print(
            "Fixed-rule predicted cost / selected savings: "
            f"${float(solve_result['rule_predicted_cost_dollars']):.4f} / "
            f"${float(solve_result['predicted_savings_vs_rule_dollars']):.4f}"
        )
        print(
            "Predicted event-exit mean / maximum / maximum band error: "
            f"{float(solve_result['selected_mean_event_exit_temperature_C']):.3f} / "
            f"{float(solve_result['selected_max_event_exit_temperature_C']):.3f} / "
            f"{float(solve_result['selected_max_event_exit_band_error_C']):.3f} degC"
        )
        print(
            "Predicted selected/RBC DR heating energy: "
            f"{float(solve_result['selected_predicted_dr_event_energy_kWh']):.3f} / "
            f"{float(solve_result['rule_predicted_dr_event_energy_kWh']):.3f} kWh "
            "(diagnostic only)"
        )
        print(
            "Adaptive solve work: "
            f"iLQR solves={int(solve_result['ilqr_solve_count'])}, "
            f"full_refresh={bool(solve_result['full_multistart_refresh'])}, "
            f"rescue={bool(solve_result['adaptive_rescue_triggered'])}, "
            f"wall={float(solve_result['solve_wall_seconds']):.2f} s"
        )
        print("Candidate comparison:")
        for candidate in solve_result["candidate_summary"]:
            print(
                f"  {candidate['name']}: "
                f"cost=${float(candidate['predicted_tariff_cost_dollars']):.4f}, "
                f"feasible={bool(candidate['feasible'])}, "
                f"selected={bool(candidate['selected'])}"
            )
        print(f"Prediction CSV saved to: {csv_path}")
        print(
            "Candidate summary CSV saved to: "
            f"{solve_result['candidate_summary_csv']}"
        )
        print("Applied setpoints [degC]:")
        print(dict(zip(ZONE_ORDER, [float(x) for x in first_sp_deg])))

        # Drop per-solve host objects before the next MPC optimization.
        del first_interval_pred, pred_df, solve_result

        self.run_idx += 1
        self.next_control_dt = control_grid_dt(self.run_idx)

        if PRINT_MPC_MEMORY_USAGE:
            print(
                f"MPC process RSS after run {self.run_idx}: "
                f"{current_process_rss_gb():.2f} GiB"
            )
        clear_mpc_runtime_caches(self.run_idx)

    def log_ep_state(self, state):
        if not self.handles_ready:
            return

        # One plant log row per EnergyPlus zone timestep.
        if not self.is_new_log_zone_timestep(state):
            return

        if self.plant_log_idx >= TOTAL_MPC_RUNS:
            return

        interval_start_dt = control_grid_dt(self.plant_log_idx)
        interval_end_dt = interval_start_dt + CONTROL_STEP
        interval_mid_dt = interval_start_dt + CONTROL_STEP / 2

        ep_callback_dt = self.sim_datetime(state).tz_convert(OPEN_METEO_TIMEZONE)

        T = self.read_zone_temperatures(state)
        self.last_T_degC = T

        interval_hours = CONTROL_STEP / pd.Timedelta(hours=1)
        zone_interval_kwh, zone_heating_rates_w, zone_energy_sources = (
            self.read_zone_interval_heating_energy_kwh(
                state,
                interval_hours=interval_hours,
            )
        )
        self.last_P_scaled = self.scale_zone_heating_rates_w(zone_heating_rates_w)
        self.cumulative_ep_zone_heating_kwh += np.nan_to_num(
            zone_interval_kwh,
            nan=0.0,
        )

        ep_interval_kwh = self.update_energy_accounting(
            state,
            accounting_dt=interval_mid_dt,
        )
        interval_is_dr = bool(is_in_csv_dr_event(interval_mid_dt))

        zone_interval_sum_kwh = float(np.nansum(zone_interval_kwh))
        zone_cumulative_sum_kwh = float(np.nansum(self.cumulative_ep_zone_heating_kwh))

        row = {
            "log_idx": int(self.plant_log_idx),

            # Official plant timestamp: end of the 15-min interval.
            "simulation_time_local": interval_end_dt.isoformat(),

            # Explicit interval labels
            "interval_start_local": interval_start_dt.isoformat(),
            "interval_end_local": interval_end_dt.isoformat(),

            # Raw EnergyPlus callback time kept only for debugging
            "ep_callback_time_local": ep_callback_dt.isoformat(),

            "ep_interval_energy_kWh": float(ep_interval_kwh) if not np.isnan(ep_interval_kwh) else np.nan,
            "ep_interval_dr_heating_kWh": (
                float(ep_interval_kwh)
                if interval_is_dr and not np.isnan(ep_interval_kwh)
                else (0.0 if not np.isnan(ep_interval_kwh) else np.nan)
            ),
            "ep_cumulative_heating_kWh": float(self.cumulative_ep_heating_kwh),
            "ep_cumulative_dr_heating_kWh": float(
                self.cumulative_ep_dr_heating_kwh
            ),

            # MPC prediction for the interval that was actually applied.
            # Full-horizon rollout predictions are matched with measurements in
            # mpc_prediction_vs_measurement_run_*.csv after the simulation ends.
            "mpc_applied_interval_avg_power_total_kW_model": float(self.current_model_first_pavg_total_kW),
            "mpc_applied_interval_energy_total_kWh_model": float(self.current_model_first_energy_total_kWh),

            # Sum of the zone-level heating energies logged below. This should
            # be close to the plant meter only if the selected zone/equipment
            # variables cover all heating electricity represented by the meter.
            "ep_sum_zone_interval_energy_kWh": zone_interval_sum_kwh,
            "ep_sum_zone_cumulative_heating_kWh": zone_cumulative_sum_kwh,
            "ep_meter_minus_sum_zone_interval_energy_kWh": (
                float(ep_interval_kwh) - zone_interval_sum_kwh
                if not np.isnan(ep_interval_kwh)
                else np.nan
            ),

            "eoff_cum_kWh": float(self.eoff_cum_kwh),
            "meter_name": self.meter_name,
            "dr_event_active": interval_is_dr,
            "night_setback_active": bool(is_night_setback(interval_start_dt)),
        }

        for i, zone_name in enumerate(ZONE_ORDER):
            safe_zone = zone_name.lower().replace(" ", "_")
            row[f"T_ep_{safe_zone}_degC"] = float(T[i])
            row[f"sp_applied_{safe_zone}_degC"] = float(self.current_sp_deg[i])

            row[f"ep_heating_rate_{safe_zone}_W"] = (
                float(zone_heating_rates_w[i])
                if np.isfinite(zone_heating_rates_w[i])
                else np.nan
            )
            row[f"ep_interval_energy_{safe_zone}_kWh"] = (
                float(zone_interval_kwh[i])
                if np.isfinite(zone_interval_kwh[i])
                else np.nan
            )
            row[f"ep_cumulative_heating_{safe_zone}_kWh"] = float(
                self.cumulative_ep_zone_heating_kwh[i]
            )
            row[f"ep_zone_energy_source_{safe_zone}"] = zone_energy_sources[i]

            if LOG_APPLIED_INTERVAL_POWER_ENERGY_IN_PLANT_LOG:
                row[f"mpc_applied_Pavg_{safe_zone}_kW_model"] = float(
                    self.current_model_first_pavg_zone_kW[i]
                )
                row[f"mpc_applied_interval_energy_{safe_zone}_kWh_model"] = float(
                    self.current_model_first_energy_zone_kWh[i]
                )
                measured_avg_power_kW = (
                    float(zone_interval_kwh[i]) / interval_hours
                    if np.isfinite(zone_interval_kwh[i])
                    else np.nan
                )
                row[f"measured_interval_avg_power_{safe_zone}_kW"] = measured_avg_power_kW
                row[f"error_applied_Pavg_{safe_zone}_kW_model_minus_measured"] = (
                    float(self.current_model_first_pavg_zone_kW[i]) - measured_avg_power_kW
                    if np.isfinite(self.current_model_first_pavg_zone_kW[i]) and np.isfinite(measured_avg_power_kW)
                    else np.nan
                )
                row[f"error_applied_interval_energy_{safe_zone}_kWh_model_minus_measured"] = (
                    float(self.current_model_first_energy_zone_kWh[i]) - float(zone_interval_kwh[i])
                    if np.isfinite(self.current_model_first_energy_zone_kWh[i]) and np.isfinite(zone_interval_kwh[i])
                    else np.nan
                )

        append_action_log(Path(LOG_DIR) / "energyplus_plant_log.csv", row)

        self.plant_log_idx += 1

    # ---- EnergyPlus callbacks ----
    def callback_after_warmup(self, state):
        self.warmup_complete = True
        print("EnergyPlus warmup complete; MPC callbacks are active.")

    def callback_begin_zone_timestep_before_set_current_weather(self, state):
        if not self.warmup_complete:
            return
        self.override_weather_if_possible(state)

    def callback_begin_zone_timestep_before_init_heat_balance(self, state):
        if not self.warmup_complete:
            return
        self.initialize_handles(state)
        if not self.handles_ready:
            return

        if self.preconditioning_control_idx < PRECONDITIONING_STEPS:
            key = self.ep_zone_timestep_key(state)
            if key == self.last_preconditioning_control_step_key:
                return
            self.last_preconditioning_control_step_key = key

            preconditioning_dt = (
                pd.Timestamp(self.sim_start_dt).tz_convert(OPEN_METEO_TIMEZONE)
                + self.preconditioning_control_idx * CONTROL_STEP
            )
            self.apply_matched_preconditioning_setpoints(
                state,
                preconditioning_dt,
            )
            self.preconditioning_control_idx += 1
            return

        self.apply_initial_temperatures_to_ep_if_possible(state)
        self.solve_and_update_action_if_needed(state)
        self.apply_current_setpoints(state)

    def callback_end_zone_timestep_after_zone_reporting(self, state):
        if not self.warmup_complete or not self.handles_ready:
            return
        if self.preconditioning_log_idx < PRECONDITIONING_STEPS:
            key = self.ep_zone_timestep_key(state)
            if key == self.last_preconditioning_log_step_key:
                return
            self.last_preconditioning_log_step_key = key
            self.preconditioning_log_idx += 1
            return
        self.log_ep_state(state)
        self.release_initial_temperature_actuators_if_needed(state)


# 10) Run EnergyPlus co-simulation
def add_energyplus_to_python_path():
    """
    Configure Python so it imports the Linux EnergyPlus Runtime API in WSL.

    This function deliberately removes stale Windows EnergyPlus paths from
    sys.path and clears any already-imported pyenergyplus modules. That matters
    in notebooks because the kernel can keep /mnt/c/EnergyPlusV26-1-0 cached
    from a previous run.
    """
    if not ENERGYPLUS_INSTALL_DIR:
        return

    ep = os.path.abspath(os.path.expanduser(ENERGYPLUS_INSTALL_DIR))

    if os.name != "nt" and "/mnt/c/EnergyPlus" in ep:
        raise RuntimeError(
            "You are running in WSL/Linux but ENERGYPLUS_INSTALL_DIR points to the Windows EnergyPlus folder:\n"
            f"  {ep}\n"
            "Use the Linux install instead, usually: ~/EnergyPlus-26-1-0"
        )

    # Fail early with a clear message if the wrong EnergyPlus build is selected.
    if os.name != "nt":
        api_lib = Path(ep) / "libenergyplusapi.so"
        if not api_lib.exists():
            raise FileNotFoundError(
                f"EnergyPlus Linux API library not found: {api_lib}\n"
                "Check that EnergyPlus was installed inside WSL at ~/EnergyPlus-26-1-0."
            )
    else:
        api_dll = Path(ep) / "energyplusapi.dll"
        if not api_dll.exists():
            print(f"WARNING: expected EnergyPlus API DLL not found: {api_dll}")

    # Remove stale Windows EnergyPlus paths from this running Python process.
    stale_fragments = [
        "/mnt/c/EnergyPlus",
        "C:\\EnergyPlus",
        "C:/EnergyPlus",
    ]
    sys.path[:] = [
        p for p in sys.path
        if not any(fragment.lower() in str(p).lower() for fragment in stale_fragments)
    ]

    # Clear cached pyenergyplus modules if a notebook imported the Windows copy earlier.
    for module_name in list(sys.modules):
        if module_name == "pyenergyplus" or module_name.startswith("pyenergyplus."):
            del sys.modules[module_name]

    # Make the value visible to pyenergyplus and to any child processes.
    os.environ["ENERGYPLUS_INSTALL_DIR"] = ep

    # Help Python find the pyenergyplus package. Put Linux path first.
    candidates = [
        ep,
        str(Path(ep) / "Python"),
        str(Path(ep) / "python"),
    ]
    for c in reversed(candidates):
        if c in sys.path:
            sys.path.remove(c)
        sys.path.insert(0, c)

    # Help the dynamic loader find EnergyPlus shared libraries.
    os.environ["PATH"] = ep + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ep + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    print(f"Configured EnergyPlus Runtime API from: {ep}")



def pre_expand_idf_for_runtime_api(working_idf, work_dir):
    """
    Expand HVACTemplate objects before launching the EnergyPlus Runtime API.

    The IDF uses HVACTemplate objects, so it must be expanded before
    the Runtime API run. We pre-expand using the standalone ExpandObjects tool
    and then pass expanded.idf to the Runtime API. The workflow is:
      1) copy the working IDF to work_dir/in.idf,
      2) run the standalone ExpandObjects executable,
      3) run the Runtime API on work_dir/expanded.idf without -x.
    """
    ep = Path(os.environ.get("ENERGYPLUS_INSTALL_DIR", ENERGYPLUS_INSTALL_DIR)).expanduser().resolve()
    work_dir = Path(work_dir).resolve()
    working_idf = Path(working_idf).resolve()

    expand_exe = ep / ("ExpandObjects.exe" if os.name == "nt" else "ExpandObjects")
    if not expand_exe.exists():
        raise FileNotFoundError(
            f"Could not find ExpandObjects executable:\n{expand_exe}\n"
            "Check ENERGYPLUS_INSTALL_DIR."
        )

    energy_idd = ep / "Energy+.idd"
    if not energy_idd.exists():
        raise FileNotFoundError(
            f"Could not find Energy+.idd:\n{energy_idd}\n"
            "Check ENERGYPLUS_INSTALL_DIR."
        )

    # ExpandObjects expects in.idf in the current working directory.
    in_idf = work_dir / "in.idf"
    expanded_idf = work_dir / "expanded.idf"
    expanded_err = work_dir / "expandedidf.err"

    # Remove old expansion artifacts so we cannot accidentally use stale output.
    for old in [in_idf, expanded_idf, expanded_err, work_dir / "audit.out"]:
        if old.exists():
            old.unlink()

    in_idf.write_text(working_idf.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    # Some ExpandObjects builds look for Energy+.idd in the current directory.
    # Copy it locally to make the standalone preprocessor deterministic.
    local_idd = work_dir / "Energy+.idd"
    if not local_idd.exists():
        local_idd.write_bytes(energy_idd.read_bytes())

    env = os.environ.copy()
    env["ENERGYPLUS_INSTALL_DIR"] = str(ep)
    env["PATH"] = str(ep) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(ep) + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        [str(expand_exe)],
        cwd=str(work_dir),
        env=env,
        capture_output=True,
        text=True,
    )

    (work_dir / "ExpandObjects_stdout.txt").write_text(result.stdout or "", encoding="utf-8", errors="ignore")
    (work_dir / "ExpandObjects_stderr.txt").write_text(result.stderr or "", encoding="utf-8", errors="ignore")

    if result.returncode != 0:
        raise RuntimeError(
            f"ExpandObjects failed with return code {result.returncode}.\n"
            f"Check {work_dir / 'ExpandObjects_stdout.txt'} and {work_dir / 'ExpandObjects_stderr.txt'}."
        )

    if not expanded_idf.exists():
        extra = ""
        if expanded_err.exists():
            extra = "\n\nexpandedidf.err:\n" + expanded_err.read_text(encoding="utf-8", errors="ignore")[:4000]
        raise FileNotFoundError(
            f"ExpandObjects completed but did not create:\n{expanded_idf}" + extra
        )

    print(f"Expanded IDF written to: {expanded_idf}")
    return str(expanded_idf)

def run_energyplus_mpc():
    for log_name in [
        "mpc_action_log.csv",
        "energyplus_plant_log.csv",
        "initial_temperature_match_check.csv",
    ]:
        p = Path(LOG_DIR) / log_name
        if p.exists():
            p.unlink()

    for old_pred in Path(LOG_DIR).glob("mpc_prediction_run_*.csv"):
        old_pred.unlink()

    for old_candidates in Path(LOG_DIR).glob(
        "mpc_candidate_summary_run_*.csv"
    ):
        old_candidates.unlink()

    for old_enriched in Path(LOG_DIR).glob("mpc_prediction_vs_measurement_run_*.csv"):
        old_enriched.unlink()

    old_all_rollouts = Path(LOG_DIR) / "mpc_prediction_vs_measurement_all_rollouts.csv"
    if old_all_rollouts.exists():
        old_all_rollouts.unlink()
        
    add_energyplus_to_python_path()
    print(f"Using EnergyPlus installation: {os.environ.get('ENERGYPLUS_INSTALL_DIR', ENERGYPLUS_INSTALL_DIR)}")
    try:
        from pyenergyplus.api import EnergyPlusAPI
    except Exception as exc:
        raise RuntimeError(
            "Could not import pyenergyplus. Set ENERGYPLUS_INSTALL_DIR at the top of this cell "
            "or set the ENERGYPLUS_INSTALL_DIR environment variable to your EnergyPlus install folder."
        ) from exc

    if not Path(IDF_PATH).exists():
        raise FileNotFoundError(f"IDF not found: {IDF_PATH}")

    epw = WEATHER_EPW_PATH or find_first_epw_next_to_idf(IDF_PATH)
    if epw is None or not Path(epw).exists():
        raise FileNotFoundError(
            "No EPW file found. Set WEATHER_EPW_PATH at the top of this cell. "
            "EnergyPlus requires an EPW to run the weather-file RunPeriod, even if "
            "Open-Meteo overrides dry-bulb/solar during the simulation."
        )

    initial_T_degC, initial_T_record = get_initial_zone_temperatures_for_day(SIM_START_DT)
    if REQUIRE_CSV_INITIAL_TEMPERATURES and initial_T_degC is None:
        raise RuntimeError(
            "The matched-initial-condition run requires a CSV temperature row "
            f"for {pd.Timestamp(SIM_START_DT).date()}."
        )
    if initial_T_degC is not None:
        save_selected_initial_temperature_log(
            LOG_DIR,
            SIM_START_DT,
            initial_T_degC,
            initial_T_record,
        )

    plant_sim_start_dt = pd.Timestamp(SIM_START_DT) - pd.Timedelta(
        days=int(PRECONDITIONING_DAYS)
    )

    weather_df = fetch_open_meteo_historical_15min(
        LAT,
        LON,
        plant_sim_start_dt,
        SIM_END_DT,
        OPEN_METEO_TIMEZONE,
    )

    work_dir = Path(LOG_DIR) / "energyplus_run"
    work_dir.mkdir(parents=True, exist_ok=True)

    working_idf = make_energyplus_working_idf(
        IDF_PATH,
        work_dir,
        plant_sim_start_dt,
        SIM_END_DT,
    )

    # The IDF contains HVACTemplate objects. Pre-expand them once,
    # then run the Runtime API on the expanded IDF. This keeps the working IDF
    # close to the original model that was verified to run with the 2026 EPW.
    runtime_idf = pre_expand_idf_for_runtime_api(working_idf, work_dir)

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    controller = EnergyPlusMPCPlantController(
        api=api,
        weather_df=weather_df,
        sim_start_dt=plant_sim_start_dt,
        sim_end_dt=SIM_END_DT,
        initial_zone_temperatures_degC=initial_T_degC,
        initial_temperature_record=initial_T_record,
    )

    controller.request_variables(state)

    # Clean old callbacks if this notebook cell is rerun.
    try:
        api.runtime.clear_callbacks()
    except Exception:
        pass

    api.runtime.callback_after_new_environment_warmup_complete(state, controller.callback_after_warmup)
    api.runtime.callback_begin_zone_timestep_before_set_current_weather(
        state,
        controller.callback_begin_zone_timestep_before_set_current_weather,
    )
    api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
        state,
        controller.callback_begin_zone_timestep_before_init_heat_balance,
    )
    api.runtime.callback_end_zone_timestep_after_zone_reporting(
        state,
        controller.callback_end_zone_timestep_after_zone_reporting,
    )

    cmd = [
        "-d",
        str(work_dir),
        "-w",
        str(epw),
        str(runtime_idf),
    ]

    print("=" * 80)
    print("Starting economic EnergyPlus MPC with optimized preheat duration and intensity")
    print(f"IDF: {runtime_idf}")
    print(f"EPW: {epw}")
    print(
        f"EnergyPlus simulation: {plant_sim_start_dt.isoformat()} to "
        f"{SIM_END_DT.isoformat()}"
    )
    print(
        f"Matched rule-based preconditioning: {PRECONDITIONING_DAYS} day(s); "
        f"MPC control begins at {SIM_START_DT.isoformat()}"
    )
    print(
        "Initial-temperature CSV match tolerance: "
        f"{float(INITIAL_TEMPERATURE_MATCH_TOLERANCE_C):.3f} degC"
    )
    print(f"Control step: {STEP_MINUTES} min | MPC horizon: {HORIZON_HOURS} h | Runs: {TOTAL_MPC_RUNS}")
    print(
        f"Preheat search window: {PREHEAT_SEARCH_WINDOW_HOURS:g} h | "
        f"rule-based reference: {float(RULE_BASED_PREHEAT_SETPOINT_C):.1f} degC for "
        f"{RULE_BASED_PREHEAT_HOURS:g} h"
    )
    print(
        "Fast adaptive multi-start/fallback: "
        f"enabled={ENABLE_MULTISTART}, "
        f"primary maxiter={REOPT_MAXITER}, "
        f"multi-start maxiter={MULTISTART_MAXITER}, "
        f"moderate seed={float(MODERATE_PREHEAT_RAW):.2f} raw for "
        f"{MODERATE_PREHEAT_HOURS:g} h"
    )
    print(
        f"Full multi-start refresh every {FULL_MULTISTART_REFRESH_MINUTES} min | "
        f"fixed-rollout lax.scan={USE_LAX_SCAN_FOR_FIXED_ROLLOUTS} | "
        f"JAX cache clearing interval={MPC_CACHE_CLEAR_INTERVAL_RUNS}"
    )
    print(
        "Rule replacement threshold: "
        f"max({100.0 * MIN_PREDICTED_SAVINGS_FRACTION:.1f}% of rule cost, "
        f"${MIN_PREDICTED_SAVINGS_DOLLARS:.3f})"
    )
    print(
        "DR event-exit target: "
        f"{float(DR_EXIT_FLOOR_C):.1f}--"
        f"{float(DR_EXIT_TARGET_C):.1f} degC | "
        f"exit-excess weight={float(W_DR_EVENT_END_EXCESS):.1f} | "
        f"stored-heat envelope weight={float(W_DR_STORED_HEAT):.1f} | "
        f"recovery reference ramp={POST_DR_RECOVERY_REFERENCE_HOURS:g} h"
    )
    print(
        f"Flex D prices: off-peak=${float(FLEX_OFF1_D_PER_KWH):.5f}/kWh | "
        f"DR=${float(FLEX_PEAK_D_PER_KWH):.5f}/kWh"
    )
    print(
        "DR heating energy: prediction and EnergyPlus measurement logging "
        "enabled; no separate DR-energy objective penalty"
    )
    print(f"Logs: {Path(LOG_DIR).resolve()}")
    print("=" * 80)

    exit_code = api.runtime.run_energyplus(state, cmd)

    print("=" * 80)
    print(f"EnergyPlus finished with exit code: {exit_code}")
    print(f"Action log: {Path(LOG_DIR) / 'mpc_action_log.csv'}")
    print(f"Plant log : {Path(LOG_DIR) / 'energyplus_plant_log.csv'}")
    print("=" * 80)

    api.state_manager.delete_state(state)
    if exit_code != 0:
        raise RuntimeError(f"EnergyPlus failed with exit code {exit_code}. Check the .err file in {work_dir}.")

    enriched_paths = enrich_prediction_rollouts_with_measurements(LOG_DIR)
    if enriched_paths:
        print("Full-rollout prediction/measurement logs:")
        for p in enriched_paths[:10]:
            print(f"  {p}")
        if len(enriched_paths) > 10:
            print(f"  ... plus {len(enriched_paths) - 10} more files")

    return controller


# 11) Batch runner over all DR days from CSV
def configure_simulation_day(day_start):
    """Update global simulation dates/log directory for one DR day."""
    global SIM_START_DT, SIM_END_DT, SIM_DAYS, TOTAL_MPC_RUNS, LOG_DIR
    global CURRENT_INITIAL_ZONE_TEMPERATURES_DEGC
    global CURRENT_EXPECTED_TEMPERATURE_BIAS_C
    global CURRENT_FLOOR_TEMPERATURE_BIAS_C
    global CURRENT_TEMPERATURE_BIAS_SAMPLE_COUNT

    day_start = pd.Timestamp(day_start)
    if day_start.tzinfo is None:
        day_start = day_start.tz_localize(OPEN_METEO_TIMEZONE)
    else:
        day_start = day_start.tz_convert(OPEN_METEO_TIMEZONE)

    SIM_START_DT = day_start
    SIM_DAYS = 1
    SIM_END_DT = SIM_START_DT + pd.Timedelta(days=1)
    TOTAL_MPC_RUNS = int((SIM_END_DT - SIM_START_DT).total_seconds() / (STEP_MINUTES * 60))

    run_label = SIM_START_DT.strftime("%Y%m%d")
    LOG_DIR = str(
        Path(BASE_LOG_DIR)
        / f"dr_day_{run_label}_mpc_light_bias_corrected"
    )
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    (
        CURRENT_EXPECTED_TEMPERATURE_BIAS_C,
        CURRENT_FLOOR_TEMPERATURE_BIAS_C,
        CURRENT_TEMPERATURE_BIAS_SAMPLE_COUNT,
        bias_audit,
    ) = estimate_historical_temperature_bias(SIM_START_DT)
    bias_audit_path = Path(LOG_DIR) / "temperature_bias_profile.csv"
    bias_audit.to_csv(bias_audit_path, index=False)
    calibrated_cells = int(
        np.sum(
            CURRENT_TEMPERATURE_BIAS_SAMPLE_COUNT
            >= int(BIAS_MIN_SAMPLES_PER_ZONE_LEAD)
        )
    )
    print(
        "Historical temperature-bias calibration: "
        f"{calibrated_cells} zone/lead cells enabled; audit={bias_audit_path}"
    )

    if USE_CSV_INITIAL_ZONE_TEMPERATURES:
        CURRENT_INITIAL_ZONE_TEMPERATURES_DEGC, _ = get_initial_zone_temperatures_for_day(SIM_START_DT)
        print("CSV initial temperatures found for this DR day [degC]:")
        print(dict(zip(ZONE_ORDER, [float(x) for x in CURRENT_INITIAL_ZONE_TEMPERATURES_DEGC])))
    else:
        CURRENT_INITIAL_ZONE_TEMPERATURES_DEGC = None

    return run_label


def run_energyplus_mpc_for_day(day_start):
    run_label = configure_simulation_day(day_start)
    print(f"Configured MPC run with CSV initial temperatures for DR day {run_label}")
    return run_energyplus_mpc()


def run_energyplus_mpc_for_day_summary(day_start_iso):
    """
    Process-pool-friendly worker.

    It returns only serializable paths/status instead of the controller object,
    because EnergyPlus API objects cannot be safely pickled back to the parent.
    """
    import traceback

    try:
        day_start = pd.Timestamp(day_start_iso)
        run_energyplus_mpc_for_day(day_start)
        return {
            "dr_day": pd.Timestamp(day_start).date().isoformat(),
            "status": "ok",
            "log_dir": str(Path(LOG_DIR).resolve()),
            "plant_log": str((Path(LOG_DIR) / "energyplus_plant_log.csv").resolve()),
            "action_log": str((Path(LOG_DIR) / "mpc_action_log.csv").resolve()),
            "initial_temperature_log": str((Path(LOG_DIR) / "selected_initial_zone_temperatures_wide.csv").resolve()),
            "error": "",
        }
    except Exception as exc:
        # Keep the error visible in the batch summary without hiding the full traceback.
        return {
            "dr_day": pd.Timestamp(day_start_iso).date().isoformat(),
            "status": "failed",
            "log_dir": str(Path(LOG_DIR).resolve()) if "LOG_DIR" in globals() else "",
            "plant_log": "",
            "action_log": "",
            "initial_temperature_log": "",
            "error": repr(exc) + "\n" + traceback.format_exc(),
        }


def _dr_day_log_dir(day_start):
    """Return the output directory used by configure_simulation_day for one DR day."""
    day_start = pd.Timestamp(day_start)
    if day_start.tzinfo is None:
        day_start = day_start.tz_localize(OPEN_METEO_TIMEZONE)
    else:
        day_start = day_start.tz_convert(OPEN_METEO_TIMEZONE)
    run_label = day_start.strftime("%Y%m%d")
    return (
        Path(BASE_LOG_DIR)
        / f"dr_day_{run_label}_mpc_light_bias_corrected"
    )


def run_energyplus_mpc_for_day_subprocess(day_start_iso):
    """
    Run one DR day by launching this same script in a fresh Python process.

    This avoids keeping EnergyPlus Runtime API objects, callbacks, JAX/XLA
    compiled functions, and device memory alive between DR days. It is slower
    than running all days in-process, but is much more robust for long batches.
    """
    day_start = pd.Timestamp(day_start_iso)
    run_label = day_start.strftime("%Y%m%d")
    log_dir = _dr_day_log_dir(day_start)
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir / f"subprocess_stdout_{run_label}.txt"
    stderr_path = log_dir / f"subprocess_stderr_{run_label}.txt"

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--single-day",
        pd.Timestamp(day_start).isoformat(),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Leave user-selected XLA memory settings untouched if already defined.
    # This optional default reduces the chance that the child grabs most memory.
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")

    print("=" * 80)
    print(f"Launching isolated subprocess for DR day {pd.Timestamp(day_start).date()}")
    print("Command:", " ".join(cmd))
    print(f"Stdout log: {stdout_path}")
    print(f"Stderr log: {stderr_path}")
    print("=" * 80)

    try:
        with stdout_path.open("w", encoding="utf-8", errors="ignore") as stdout_f, \
             stderr_path.open("w", encoding="utf-8", errors="ignore") as stderr_f:
            completed = subprocess.run(
                cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                env=env,
                cwd=str(Path(__file__).resolve().parent),
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        return_code = int(completed.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        return_code = -999
        timed_out = True
        with stderr_path.open("a", encoding="utf-8", errors="ignore") as stderr_f:
            stderr_f.write("\nSUBPROCESS TIMEOUT\n")
            stderr_f.write(str(exc))
            stderr_f.write("\n")

    status = "ok" if return_code == 0 else "failed"
    error = ""
    if timed_out:
        error = f"Subprocess timed out after {SUBPROCESS_TIMEOUT_SECONDS} seconds."
    elif return_code != 0:
        error = f"Subprocess exited with return code {return_code}. Check stderr/stdout logs."
        try:
            tail = stderr_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]
            if tail:
                error += "\n\nLast stderr lines:\n" + "\n".join(tail)
        except Exception:
            pass

    result = {
        "dr_day": pd.Timestamp(day_start).date().isoformat(),
        "status": status,
        "return_code": return_code,
        "log_dir": str(log_dir.resolve()),
        "plant_log": str((log_dir / "energyplus_plant_log.csv").resolve()),
        "action_log": str((log_dir / "mpc_action_log.csv").resolve()),
        "initial_temperature_log": str((log_dir / "selected_initial_zone_temperatures_wide.csv").resolve()),
        "stdout_log": str(stdout_path.resolve()),
        "stderr_log": str(stderr_path.resolve()),
        "error": error,
    }

    print(f"Finished isolated subprocess for {result['dr_day']} with status={status}, return_code={return_code}")
    if error:
        print(error.splitlines()[0])
    return result


def run_all_dr_days_mpc_night_setback_preference():
    event_windows = get_dr_event_windows()
    days = unique_dr_days(event_windows)
    if not days:
        raise RuntimeError("No DR days were found in the CSV.")

    print(f"Loaded {len(event_windows)} DR events from: {DR_EVENTS_CSV_PATH}")
    print(
        f"Running {len(days)} unique DR days with economic MPC, optimized "
        "preheating, and CSV initial temperatures:"
    )
    for day in days:
        print(f"  {day.date()}")

    if USE_CSV_INITIAL_ZONE_TEMPERATURES:
        # Fail early if the CSV is missing a day or a zone column, before launching
        # expensive EnergyPlus simulations.
        for day in days:
            get_initial_zone_temperatures_for_day(day)

    summary_rows = []

    if RUN_DAYS_IN_SUBPROCESSES:
        print(
            "Subprocess-isolated sequential mode enabled. "
            "Each DR day runs in a fresh Python process, so EnergyPlus/JAX state is not reused."
        )
        for day in days:
            result = run_energyplus_mpc_for_day_subprocess(pd.Timestamp(day).isoformat())
            summary_rows.append(result)
            if result.get("status") != "ok" and STOP_BATCH_ON_FIRST_FAILURE:
                print("Stopping batch because STOP_BATCH_ON_FIRST_FAILURE=True.")
                break

    elif RUN_DAYS_IN_PARALLEL:
        print(
            f"Parallel mode enabled with up to {MAX_PARALLEL_WORKERS} workers. "
            "Use process-based parallelism only; EnergyPlus Runtime API/JAX should not be shared across threads."
        )
        ctx = mp.get_context("spawn")
        day_inputs = [pd.Timestamp(day).isoformat() for day in days]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(MAX_PARALLEL_WORKERS),
            mp_context=ctx,
        ) as executor:
            future_to_day = {
                executor.submit(run_energyplus_mpc_for_day_summary, day_iso): day_iso
                for day_iso in day_inputs
            }
            for future in concurrent.futures.as_completed(future_to_day):
                result = future.result()
                summary_rows.append(result)
                print(f"Finished DR day {result['dr_day']} with status: {result['status']}")
    else:
        # Legacy in-process mode. This can be faster for one run, but it is less
        # robust for a long sequence because EnergyPlus Runtime API and JAX/XLA
        # native memory may not be fully released between DR days.
        for day in days:
            controller = run_energyplus_mpc_for_day(day)
            summary_rows.append({
                "dr_day": pd.Timestamp(day).date().isoformat(),
                "status": "ok",
                "return_code": 0,
                "log_dir": str(Path(LOG_DIR).resolve()),
                "plant_log": str((Path(LOG_DIR) / "energyplus_plant_log.csv").resolve()),
                "action_log": str((Path(LOG_DIR) / "mpc_action_log.csv").resolve()),
                "initial_temperature_log": str((Path(LOG_DIR) / "selected_initial_zone_temperatures_wide.csv").resolve()),
                "stdout_log": "",
                "stderr_log": "",
                "error": "",
            })
            # Do not keep controllers from previous days alive.
            del controller
            try:
                import gc
                gc.collect()
                jax.clear_caches()
            except Exception:
                pass

    summary_rows = sorted(summary_rows, key=lambda r: r["dr_day"])
    summary_path = Path(BASE_LOG_DIR) / "mpc_event_exit_target_batch_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Batch summary saved to: {summary_path}")

    failures = [r for r in summary_rows if r.get("status") != "ok"]
    if failures:
        print("One or more DR-day simulations failed. See the batch summary error/stdout/stderr columns.")
        for row in failures:
            first_error_line = row.get("error", "").splitlines()[0] if row.get("error") else "unknown error"
            print(f"  {row['dr_day']}: {first_error_line}")
        raise RuntimeError(f"{len(failures)} DR-day simulation(s) failed.")

    return summary_rows


def parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run EnergyPlus MPC DR-day simulations with optional subprocess isolation."
    )
    parser.add_argument(
        "--single-day",
        default=None,
        help=(
            "Run exactly one DR day in this process. "
            "Used internally by the subprocess-isolated batch runner. "
            "Example: --single-day 2025-12-03T00:00:00-05:00"
        ),
    )
    parser.add_argument(
        "--legacy-in-process-batch",
        action="store_true",
        help="Run all DR days in the current process instead of subprocess-isolated mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()

    if args.single_day is not None:
        # Child-process entry point. Run one day and exit so all native
        # EnergyPlus/JAX/XLA state is released by the operating system.
        controller = run_energyplus_mpc_for_day(pd.Timestamp(args.single_day))
        del controller
    else:
        if args.legacy_in_process_batch:
            RUN_DAYS_IN_SUBPROCESSES = False
        summary_rows = run_all_dr_days_mpc_night_setback_preference()
