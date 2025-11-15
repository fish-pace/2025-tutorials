import numpy as np
import xarray as xr
import pandas as pd

def time_series_split(
    data: xr.Dataset,
    num_var,
    cat_var=None,
    mask="ocean_mask",
    split_ratio=(0.7, 0.2, 0.1),
    seed=42,
    X_mean=None,
    X_std=None,
    y_var="y",
    years=None,               # select one year, a list of years, or a slice
    cast_float32=True,
    contiguous_splits=False,
    return_full=False,
    nan_max_frac_y=0.5, # what fraction of missing days allowed in y
    nan_max_frac_v=0.05, # what fraction of missing days allowed in numerical vars
    add_missingness=False,
    verbose=False
):
    """
    Pure-NumPy splitter/normalizer for xarray Dataset (NumPy-backed). 
    Splits time indices randomly into train/val/test.
    Normalizes numerical variables only, using either provided or training-set mean/std. 
    Replaces NaNs with 0s.
    Removes days with too many NaNs (> 
    
    Parameters: 
      data: xarray dataset with 'time' dimension 
      years: year(s) to use for training
      num_var: list of numerical variable names (to normalize) 
      cat_var: list of categorical variable names (no normalization) 
      y_var: name of response variable in data. 
      mask: name of the mask in the data. 0 = ignore; 1 = use; can be static or one for each time step (y)
      split_ratio: tuple (train, val, test), must sum to 1.0 
      seed: random seed 
      nan_max_frac_y: maximum percent missing values for response
      nan_max_frac_v: maximum percent missing values for explanatory variables
      X_mean, X_std: optional mean/std arrays for num_var only (shape = [n_num_vars]) 
      cast_float32 : If True, cast outputs to float32 (good for TF)
      verbose: print out info
      return_full: return X and y
      contiguous_splits: versus random splits
      
    Returns: 
      X, y: full input and response arrays (NumPy arrays) 
      X_train, y_train, X_val, y_val, X_test, y_test: split data X_mean, X_std: mean and std used for normalization
      If return_full=False, X and y are None.
    """
    if cat_var is None:
        cat_var = []
    input_var = list(num_var) + list(cat_var)

    # --- checks
    if "time" not in data.dims:
        raise ValueError("Dataset must contain a 'time' dimension.")
    if abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError("split_ratio must sum to 1.0")
    if "ocean_mask" not in data:
        raise KeyError("Dataset must contain 'ocean_mask' (1=ocean, 0=land).")

    # ---------- subset by year(s) ----------
    if years is not None:
        if isinstance(years, (str, int)):
            data = data.sel(time=str(years))
        elif isinstance(years, slice):
            data = data.sel(time=years)
        else:
            # assume iterable of years (ints/strs)
            ti = pd.DatetimeIndex(np.asarray(data["time"].values))
            yrs = set(int(y) for y in years)
            sel = xr.DataArray(np.isin(ti.year, list(yrs)), coords={"time": data["time"]}, dims=["time"])
            data = data.sel(time=sel)
    if data.sizes.get("time", 0) == 0:
        raise ValueError("No timesteps left after year filtering.")

    # create a template for broadcasting 2D -> 3D
    template = data[y_var]

    # NaN-based time filtering where mask = 1
    ocean = data["ocean_mask"].astype(bool)
    if "time" not in ocean.dims:
        ocean = ocean.expand_dims({"time": data["time"]}).broadcast_like(template)
    else:
        ocean = ocean.broadcast_like(template)

    spatial_dims = [d for d in ocean.dims if d != "time"]
    ocean_pix_per_t = ocean.sum(dim=spatial_dims)

    check_vars = input_var + [y_var]
    valid_times = xr.DataArray(np.ones(data.sizes["time"], dtype=bool), coords={"time": data["time"]}, dims=["time"])

    for v in check_vars:
        if v not in data:
            raise KeyError(f"Variable '{v}' not found in dataset.")
        arr = data[v]
        if "time" not in arr.dims:
            arr = arr.expand_dims({"time": data["time"]}).broadcast_like(template)
        else:
            arr = arr.broadcast_like(template)
        frac = nan_max_frac_y if v == y_var else nan_max_frac_v
        nan_thresh = frac * ocean_pix_per_t                     # (time,)
        v_nan = xr.apply_ufunc(np.isnan, arr) & ocean
        v_nan_count = v_nan.sum(dim=spatial_dims)
        # Remove days with too many NaNs
        valid_times = valid_times & (v_nan_count < nan_thresh)

    before = int(data.sizes["time"])
    data = data.sel(time=valid_times)
    ocean = ocean.sel(time=valid_times)
    after = int(data.sizes["time"])
    if after == 0:
        raise ValueError("No timesteps left after NaN filtering.")
    if verbose:
        yrs_msg = f" (years={years})" if years is not None else ""
        print(f"[NaN filter]{yrs_msg} kept {after}/{before} days "
              f"(≤ {nan_max_frac*100:.1f}% NaNs over ocean per variable).")
    
        # --- days-per-month report ---
        t = pd.to_datetime(data["time"].values)
    
        # group by year-month (works for one or many years)
        per_month = (
            pd.Series(1, index=pd.Index(t, name="time"))
            .groupby([t.year, t.month])
            .sum()
            .astype(int)
        )
        # prettify as "YYYY-MM"
        per_month.index = [f"{y:04d}-{m:02d}" for y, m in per_month.index]
        print("Days kept per month:")
        for ym, cnt in per_month.items():
            print(f"    {ym}: {cnt}")
    
        # compact 12-month line when only a single year is present
        if len(pd.unique(t.year)) == 1:
            counts = (
                pd.Series(1, index=t)
                .groupby(t.month)
                .sum()
                .reindex(range(1, 13), fill_value=0)
                .astype(int)
            )
            print("By month (Jan..Dec):", " ".join(f"{c:2d}" for c in counts.values))
    
    # ---------- split indices ----------
    time_len = data.sizes["time"]
    rng = np.random.default_rng(seed)
    all_indices = rng.choice(time_len, size=time_len, replace=False)

    # Compute indices for splitting data into train, validate, and test
    train_end = int(split_ratio[0] * time_len)
    val_end = int((split_ratio[0] + split_ratio[1]) * time_len)
    train_idx = np.sort(all_indices[:train_end])
    val_idx = np.sort(all_indices[train_end:val_end])
    test_idx = np.sort(all_indices[val_end:])


    # ---------- helpers ----------
    def fetch(var):
        tmpl = data[y_var]                    # current (post-filter) template
        arr  = data[var]
        if "time" not in arr.dims:
            arr = arr.expand_dims({"time": data["time"]}).broadcast_like(tmpl)
        else:
            # ensure identical order & coords; avoid resurrecting dropped times
            arr = arr.transpose("time", ...).reindex_like(tmpl)
        out = arr.values.astype("float32", copy=False) if cast_float32 else arr.values
        return out

    # stats from training; compute before imputation (with median)
    if num_var:
        if X_mean is None or X_std is None:
            means, stds = [], []
            for v in num_var:
                a = fetch(v)
                a_tr = a[train_idx]
                means.append(np.nanmean(a_tr, axis=(0, 1, 2)))
                stds.append( np.nanstd( a_tr, axis=(0, 1, 2)))
            X_mean = np.asarray(means, dtype="float32" if cast_float32 else a.dtype)
            X_std  = np.asarray(stds,  dtype="float32" if cast_float32 else a.dtype)
        X_std_safe = np.where(X_std == 0, 1.0, X_std)
    else:
        X_mean = np.array([], dtype="float32" if cast_float32 else float)
        X_std  = np.array([], dtype="float32" if cast_float32 else float)
        X_std_safe = X_std

    # ---- precompute per-pixel medians for num_var using ONLY training data
    ocean_np = ocean.transpose("time","lat","lon").values
        
    medians = []
    for v in num_var:
        a      = fetch(v)                 # (T, H, W)
        a_tr   = a[train_idx]             # (t, H, W)
        oce_tr = ocean_np[train_idx]      # (t, H, W) boolean
    
        # Mask: land OR invalid values
        masked = np.ma.array(a_tr, mask=(~oce_tr) | (~np.isfinite(a_tr)))
    
        # Per-pixel median across time (returns masked result if all masked)
        med_ma = np.ma.median(masked, axis=0)        # (H, W) masked array
        med    = med_ma.filled(np.nan)               # fill all-masked pixels with NaN
    
        # Fallback for pixels with no finite ocean values
        cnt = np.isfinite(masked.filled(np.nan)).sum(axis=0)
        if masked.count() > 0:
            global_med = float(np.ma.median(masked))
        else:
            global_med = 0.0
        med = np.where(cnt > 0, med, global_med).astype('float32', copy=False)
    
        medians.append(med)     
        
    def build_split(idx):
        chans = []
    
        # numeric (normalize, impute NaNs with per-pixel medians; optional missingness)
        for k, v in enumerate(num_var):
            a = fetch(v)             # (T, H, W)
            a = a[idx]               # (t, H, W)
            oce_t = ocean_np[idx]  # (t, H, W)
            miss = (~np.isfinite(a)) & oce_t
            # impute with per-pixel median
            a = np.where(miss, medians[k], a)
            # normalize
            a = (a - X_mean[k]) / X_std_safe[k]
            a = np.where(oce_t, a, 0.0)  # set values over land to 0
            chans.append(a.astype("float32", copy=False))
            if add_missingness:
                # add a 0/1 channel indicating originally-missing inputs
                chans.append(miss.astype("float32", copy=False))
    
        # categorical (just fill NaNs with 0 or a benign default)
        for v in cat_var:
            a = fetch(v)             # (T, H, W)
            a = a[idx]
            a = np.nan_to_num(a)     # okay for categorical/auxiliary
            chans.append(a.astype("float32", copy=False))
    
        if not chans:
            raise ValueError("No input variables provided.")
        # stack channels last → (t, H, W, C)
        return np.stack(chans, axis=-1)
    
    # IMPORTANT: keep NaNs in y so your masked loss can ignore cloudy/land pixels!
    y_full = data[y_var].transpose("time", ...).values
    if cast_float32:
        y_full = y_full.astype("float32", copy=False)
    
    def take_y(idx):
        y_s = y_full[idx]        # DO NOT nan to 0 here
        return y_s

    # ---------- build splits ----------
    X_train = build_split(train_idx); y_train = take_y(train_idx)
    X_val   = build_split(val_idx);   y_val   = take_y(val_idx)
    X_test  = build_split(test_idx);  y_test  = take_y(test_idx)

    if return_full:
      X = build_split(slice(0, time_len))
      y = take_y(slice(0, time_len))
    else:
      X = None
      y = None

    return X, y, X_train, y_train, X_val, y_val, X_test, y_test, X_mean, X_std

# Save and load model bundle
import json, zipfile, tempfile, io
from pathlib import Path
import numpy as np
from keras.models import load_model

# Optional dependency (only needed if you pass custom_objects)
try:
    import cloudpickle
except Exception:  # pragma: no cover
    cloudpickle = None

# Save and Load fitted model
import json, zipfile, tempfile
from pathlib import Path
import numpy as np
import tensorflow as tf

def save_cnn_bundle(zip_path, model, X_mean, X_std, meta=None):
    """
    Create a single zip containing:
      - model.keras
      - stats.npz  (X_mean, X_std)
      - meta.json  (optional dict)
    """
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1) Save model in Keras native format
        model_path = tmp / "model.keras"
        model.save(model_path)
        # 2) Save stats
        np.savez(tmp / "stats.npz", X_mean=X_mean, X_std=X_std)
        # 3) Save meta
        (tmp / "meta.json").write_text(json.dumps(meta or {}))

        # 4) Zip it up
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(model_path, arcname="model.keras")
            z.write(tmp / "stats.npz", arcname="stats.npz")
            z.write(tmp / "meta.json", arcname="meta.json")

    return str(zip_path)


def load_cnn_bundle(zip_path, compile=False):
    """
    Load a bundle produced by save_inference_zip().
    Returns: (model, X_mean, X_std, meta_dict)
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Extract all files
        z.extract("model.keras", path=tmp)
        z.extract("stats.npz",  path=tmp)
        # meta.json might be missing in older bundles; handle gracefully
        meta = {}
        if "meta.json" in z.namelist():
            z.extract("meta.json", path=tmp)
            meta = json.loads((tmp / "meta.json").read_text())

        # Load model & stats
        model = tf.keras.models.load_model(tmp / "model.keras", compile=compile)
        stats = np.load(tmp / "stats.npz")
        X_mean, X_std = stats["X_mean"], stats["X_std"]

    return model, X_mean, X_std, meta


## PLOTTING

import numpy as np
import matplotlib.pyplot as plt

def predict_and_plot_date(
    data_xr,
    date,                          # "YYYY-MM-DD" or np.datetime64
    model,
    num_var,                       # list of vars to normalize
    cat_var,                       # list of vars not normalized (e.g., ocean_mask, sin/cos time)
    X_mean, X_std,                 # per-channel stats for num_var (shape [len(num_var)])
    y_var="y",
    mask_var="ocean_mask",
    model_type="cnn",              # "cnn" or "tabular"
    cast_float32=True,
    use_percentiles=False, p_lo=5, p_hi=95,
    cmap="viridis"
):
    """
    Build one-sample input from dataset for a specific date, predict, and plot True vs Pred.
    Works with CNN (map→map) and tabular models (flattened pixels).
    """
    # ---- resolve date index
    date64 = np.datetime64(str(date))
    times = np.asarray(data_xr["time"].values)
    idxs = np.where(times == date64)[0]
    if idxs.size == 0:
        raise ValueError(f"Date {date} not found in dataset time coord.")
    t = int(idxs[0])

    # ---- helper to fetch a variable as (H,W) for that date; broadcast 2D to 3D if needed
    # choose a spatial template (first available among inputs or y)
    tmpl_name = (num_var + cat_var + [y_var])[0]
    tmpl = data_xr[tmpl_name]
    def fetch_2d(varname):
        arr = data_xr[varname]
        if "time" in arr.dims:
            arr_t = arr.isel(time=t)
        else:
            arr_t = arr
        arr_t = arr_t.broadcast_like(tmpl.isel(time=t))  # ensure same H,W
        a = arr_t.values
        if cast_float32:
            a = a.astype("float32", copy=False)
        return a  # (H,W)

    # ---- build channels for this date
    num_chans = []
    for k, vn in enumerate(num_var):
        a = fetch_2d(vn)
        # normalize with training stats, then fill NaNs with 0
        a = (a - X_mean[k]) / (1.0 if X_std[k] == 0 else X_std[k])
        a = np.nan_to_num(a)
        num_chans.append(a)

    cat_chans = []
    for vn in cat_var:
        a = fetch_2d(vn)
        a = np.nan_to_num(a)
        cat_chans.append(a)

    if not (num_chans or cat_chans):
        raise ValueError("No input variables provided.")

    # stack to (H,W,C)
    X_map = np.stack(num_chans + cat_chans, axis=-1)
    H, W, C = X_map.shape

    # ---- ground truth map
    y_true = fetch_2d(y_var)

    # ---- predict
    if model_type == "cnn":
        y_pred = model.predict(X_map[np.newaxis, ...], verbose=0)[0]
        if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
            y_pred = y_pred[..., 0]
    elif model_type == "tabular":
        y_pred = model.predict(X_map.reshape(-1, C)).reshape(H, W)
    else:
        raise ValueError("model_type must be 'cnn' or 'tabular'.")

    # ---- mask land to NaN (mask_var==0 → land)
    land = (fetch_2d(mask_var) == 0.0)
    y_true = np.where(land, np.nan, y_true)
    y_pred = np.where(land, np.nan, y_pred)

    # ---- color limits
    if use_percentiles:
        stack = np.concatenate([y_true[~np.isnan(y_true)], y_pred[~np.isnan(y_pred)]]) if np.isfinite(y_true).any() and np.isfinite(y_pred).any() else np.array([])
        vmin, vmax = (np.percentile(stack, p_lo), np.percentile(stack, p_hi)) if stack.size else (None, None)
    else:
        vmin = np.nanmin([y_true, y_pred]); vmax = np.nanmax([y_true, y_pred])

    # --- ensure North is up
    lat = np.array(data_xr.lat.values)
    flip_lat = lat[0] > lat[-1]   # True if lat is descending

    if flip_lat:
        y_true = np.flipud(y_true)
        y_pred = np.flipud(y_pred)

    # extent must be (xmin, xmax, ymin, ymax) with increasing y
    lon_min, lon_max = float(data_xr.lon.min()), float(data_xr.lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    extent = [lon_min, lon_max, lat_min, lat_max]

    # ---- plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(y_true, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title(f"True {y_var} — {np.datetime_as_string(date64)}"); axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title(f"Predicted ({model_type.upper()})"); axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.show()
    return y_true, y_pred

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_true_vs_predicted_year_multi(
    data, year,
    models,                 # list of models
    X_mean, X_std,          # per-channel stats for num_var only
    num_var, cat_var,       # lists of variable names
    y_var="y",
    model_types="cnn",      # list like ['cnn','tabular', ...] same length as models
    model_names=None,       # optional names for column titles
    cmap='viridis',
    day=1,
    use_percentiles=True, p_lo=5, p_hi=95
):
    assert len(models) == len(model_types), "models and model_types must have same length"
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]

    ds = data.sel(time=year)

    # use day-th day of each month
    dates = pd.to_datetime(ds.time.values)
    df = pd.DataFrame({'date': dates, 'dom': dates.day, 'y': dates.year, 'm': dates.month})
    monthly_dates = pd.DatetimeIndex(
        df.groupby(['y','m']).apply(
            lambda g: g.loc[g.dom>=day, 'date'].min() if (g.dom>=day).any() else g['date'].max()
        ).sort_values().values
    )
    n_months = len(monthly_dates)

    lat = ds.lat.values
    lon = ds.lon.values
    flip_lat = lat[0] > lat[-1]
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    land_mask = (ds["ocean_mask"].values == 0.0)

    # helper: fetch a 2D array for var at given date; broadcast if var has no time dim
    def fetch_2d(var, date):
        arr = ds[var]
        arr_t = arr.sel(time=date) if "time" in arr.dims else arr
        arr_t = arr_t.broadcast_like(ds[y_var].sel(time=date))
        a = arr_t.values.astype("float32", copy=False)
        return a

    # figure: True + one column per model
    ncols = 1 + len(models)
    fig, axs = plt.subplots(n_months, ncols, figsize=(3.2*ncols, 2.2*n_months), constrained_layout=True)
    if n_months == 1:
        axs = np.atleast_2d(axs)  # ensure 2D indexing

    for i, date in enumerate(monthly_dates):
        # Build (H,W,C) input for this date
        chans = []
        for k, v in enumerate(num_var):
            a = fetch_2d(v, date)
            denom = 1.0 if X_std[k] == 0 else X_std[k]
            a = (a - X_mean[k]) / denom
            chans.append(np.nan_to_num(a))
        for v in cat_var:
            a = fetch_2d(v, date)
            chans.append(np.nan_to_num(a))
        X_map = np.stack(chans, axis=-1)
        H, W, C = X_map.shape

        # Truth
        truth = fetch_2d(y_var, date)

        # Predict with each model
        preds = []
        for mdl, mtype in zip(models, model_types):
            if mtype == "cnn":
                yhat = mdl.predict(X_map[np.newaxis, ...], verbose=0)[0]
                if yhat.ndim == 3 and yhat.shape[-1] == 1:
                    yhat = yhat[..., 0]
            elif mtype == "tabular":
                yhat = mdl.predict(X_map.reshape(-1, C)).reshape(H, W)
            else:
                raise ValueError("model_type must be 'cnn' or 'tabular'.")
            preds.append(yhat)

        # Apply ocean mask and optional north-up flip
        truth_m = np.where(land_mask, np.nan, truth)
        preds_m = [np.where(land_mask, np.nan, p) for p in preds]
        if flip_lat:
            truth_m = np.flipud(truth_m)
            preds_m = [np.flipud(p) for p in preds_m]

        # Shared color limits per row
        all_maps = [truth_m] + preds_m
        if use_percentiles:
            stack = np.concatenate([m[np.isfinite(m)] for m in all_maps if np.isfinite(m).any()]) if any(np.isfinite(m).any() for m in all_maps) else np.array([])
            vmin, vmax = (np.percentile(stack, p_lo), np.percentile(stack, p_hi)) if stack.size else (None, None)
        else:
            vmin = np.nanmin(all_maps); vmax = np.nanmax(all_maps)

        # True panel
        ax = axs[i, 0]
        im = ax.imshow(truth_m, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        ax.set_title(f"{date.strftime('%Y-%m-%d')} — True", fontsize=9)
        ax.axis('off')

        # Prediction panels with metrics
        for j, (pmap, name) in enumerate(zip(preds_m, model_names), start=1):
            axp = axs[i, j]
            im = axp.imshow(pmap, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
            axp.axis('off')

            # metrics (mask NaNs)
            m = np.isfinite(truth_m) & np.isfinite(pmap)
            if m.any():
                r2 = r2_score(truth_m[m].ravel(), pmap[m].ravel())
                rmse = np.sqrt(np.mean((truth_m[m] - pmap[m])**2))
                axp.set_title(f"{name}\n$R^2$={r2:.2f}, RMSE={rmse:.2f}", fontsize=9)
            else:
                axp.set_title(f"{name}\nno valid pixels", fontsize=9)

    # one colorbar for the last column
    # cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    # fig.colorbar(im, cax=cax, label=y_var)
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import calendar

def plot_metric_by_month(
    data, years, model, X_mean, X_std, num_var, cat_var,
    training_year=None, metric='r2',
    y_name='y', mask_var='ocean_mask',
    ssim_win_size=None, ssim_sigma=None,
    ymin=None, ymax=None
):
    assert metric in ['r2', 'rmse', 'mae', 'bias', 'ssim']

    def fetch_2d(ds, var, date, like_var):
        arr = ds[var]
        arr_t = arr.sel(time=date) if 'time' in arr.dims else arr
        arr_t = arr_t.broadcast_like(ds[like_var].sel(time=date))
        # ensure spatial order is (lat, lon)
        if tuple(d for d in arr_t.dims if d != 'time') != ('lat','lon'):
            arr_t = arr_t.transpose(..., 'lat', 'lon') if 'time' in arr_t.dims else arr_t.transpose('lat','lon')
        return arr_t.values.astype('float32', copy=False)
        
    metric_by_year_month = {}

    for year in years:
        ds = data.sel(time=year)
        dates = pd.to_datetime(ds.time.values)
        monthly_dates = (
            pd.Series(dates).groupby([dates.year, dates.month]).min().sort_values()
        )

        scores = []
        for date in monthly_dates:
            # build (H,W,C) input for this date
            chans = []
            for k, v in enumerate(num_var):
                a = fetch_2d(ds, v, date, y_name)
                denom = 1.0 if X_std[k] == 0 else X_std[k]
                a = (a - X_mean[k]) / denom
                chans.append(np.nan_to_num(a))
            for v in cat_var:
                a = fetch_2d(ds, v, date, y_name)
                chans.append(np.nan_to_num(a))
            X_map = np.stack(chans, axis=-1)

            # predict
            pred = model.predict(X_map[np.newaxis, ...], verbose=0)[0]
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred[..., 0]

            # truth & mask
            truth = fetch_2d(ds, y_name, date, y_name)
            land = (fetch_2d(ds, mask_var, date, y_name) == 0.0)
            pred  = np.where(land, np.nan, pred)
            truth = np.where(land, np.nan, truth)

            # metric
            if metric == 'ssim':
                # fill NaNs for SSIM computation
                t = np.nan_to_num(truth, nan=(np.nanmean(truth) if np.isfinite(truth).any() else 0.0))
                p = np.nan_to_num(pred,  nan=(np.nanmean(pred)  if np.isfinite(pred).any()  else 0.0))
                # robust data_range
                dr = np.nanmax(truth) - np.nanmin(truth)
                if not np.isfinite(dr) or dr == 0:
                    dr = (np.nanmax(t) - np.nanmin(t)) or 1.0
                # build kwargs safely (don’t pass sigma=None)
                ssim_kwargs = {"data_range": dr}
                if ssim_win_size is not None:
                    ssim_kwargs["win_size"] = int(ssim_win_size)  # must be odd
                if ssim_sigma is not None:
                    ssim_kwargs["gaussian_weights"] = True
                    ssim_kwargs["sigma"] = float(ssim_sigma)
                score = ssim(t.astype(np.float64), p.astype(np.float64), **ssim_kwargs)
            else:
                m = ~np.isnan(truth) & ~np.isnan(pred)
                if not m.any():
                    score = np.nan
                elif metric == 'r2':
                    score = r2_score(truth[m].ravel(), pred[m].ravel())
                elif metric == 'rmse':
                    score = float(np.sqrt(np.mean((truth[m] - pred[m])**2)))
                elif metric == 'mae':
                    score = float(mean_absolute_error(truth[m], pred[m]))
                elif metric == 'bias':
                    score = float(np.mean(pred[m] - truth[m]))

            scores.append(score)

        metric_by_year_month[year] = (monthly_dates.dt.month.values, scores)

    # plot
    plt.figure(figsize=(10,5))
    for year, (months, scores) in metric_by_year_month.items():
        label = f"{year} (train)" if year == training_year else year
        style = "--" if year == training_year else "-"
        plt.plot(months, scores, style, marker='o', label=label)

    plt.xlabel("Month")
    plt.ylabel({'r2':"$R^2$",'rmse':"RMSE",'mae':"MAE",'bias':"Bias",'ssim':"SSIM"}[metric])
    plt.title(f"Monthly {metric.upper()} by Year")
    plt.xticks(np.arange(1,13), calendar.month_abbr[1:13])
    plt.legend(); plt.grid(True); plt.tight_layout(); 
    plt.ylim(ymin, ymax)
    
    plt.show()

def plot_4metric_by_month(
    data, years, model, X_mean, X_std, num_var, cat_var,
    training_year=None,
    y_name='y', mask_var='ocean_mask',
    ssim_win_size=None, ssim_sigma=None,
    ymin=None, ymax=None,
    model_type="cnn",
):
    """
    Compute and plot monthly performance metrics (R², bias, MAE, SSIM) over
    selected days in each month (1, 7, 14, 28) for either a CNN or a BRT model.

    For each year in ``years``:
      1. Select timesteps from ``data`` within that year.
      2. Group timesteps by (year, month).
      3. Within each month, select dates whose day-of-month is in {1, 7, 14, 28}.
      4. For each selected date, build an (H, W, C) feature map using numerical
         variables (``num_var``) and categorical variables (``cat_var``). If
         ``X_mean``/``X_std`` are provided, numerical variables are normalized.
      5. Run the model to obtain predictions for each (H, W) field:
           - If ``model_type == "cnn"``:
               * Stack daily inputs into (B, H, W, C) and call
                 ``model.predict(X_batch)``.
               * Output is expected as (B, H, W) or (B, H, W, 1).
           - If ``model_type == "brt"``:
               * For each day, reshape (H, W, C) → (H*W, C),
                 call ``model.predict(X_flat)``, then reshape predictions
                 back to (H, W).
      6. For each day, mask land using ``mask_var``, drop NaNs and compute:
           * R²
           * Bias (pred - truth)
           * MAE
           * SSIM (with optional window size and Gaussian sigma)
      7. Average daily metrics within each month (via ``np.nanmean``) to obtain
         a monthly value.
      8. Produce a 2×2 panel plot of monthly metrics across all years, with an
         optional highlight for ``training_year``.

    Months that do not contain any of the target days {1, 7, 14, 28} are skipped.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing the target and predictor variables.
        Must have a ``time`` dimension and at least the variables in
        ``num_var``, ``cat_var``, ``y_name`` and ``mask_var``.
    years : sequence of int
        Years to evaluate (used with ``data.sel(time=year)``).
    model : object
        - If ``model_type == "cnn"``: a tf.keras.Model (or compatible) that
          accepts (B, H, W, C) and returns (B, H, W) or (B, H, W, 1).
        - If ``model_type == "brt"``: a scikit-learn-like regressor with
          ``predict(X_2d)`` where X_2d has shape (n_samples, n_features).
    X_mean : array-like of float or None
        Per-channel means for numerical variables in ``num_var``.
        Length must match ``len(num_var)`` if not None. If None, no mean/std
        normalization is applied.
    X_std : array-like of float or None
        Per-channel standard deviations for numerical variables in ``num_var``.
        Length must match ``len(num_var)`` if not None. If a std is zero, that
        channel is left unscaled. If None, no mean/std normalization is applied.
    num_var : list of str
        Names of numerical predictor variables in ``data``. Each is fetched,
        broadcast to the target grid, optionally normalized, and stacked as a
        channel.
    cat_var : list of str
        Names of categorical / non-normalized predictor variables in ``data``.
        Each is fetched, broadcast to the target grid, and stacked as a channel
        (no mean/std normalization).
    training_year : int, optional
        Year used for training. If provided, that year's line is dashed and
        labeled "(train)".
    y_name : str, default "y"
        Name of the target variable in ``data``.
    mask_var : str, default "ocean_mask"
        Name of the land/ocean mask variable. Values equal to 0.0 are treated
        as land and masked out.
    ssim_win_size : int, optional
        Window size for SSIM. Must be odd if provided.
    ssim_sigma : float, optional
        Standard deviation for Gaussian-weighted SSIM. If provided, Gaussian
        weights are used.
    ymin, ymax : float, optional
        Common y-limits for all metric subplots. If None, matplotlib defaults.
    model_type : {"cnn", "brt"}, default "cnn"
        Type of model:
          - "cnn": use batched (B, H, W, C) predictions.
          - "brt": predict on flattened features per day and reshape back.

    Notes
    -----
    - SSIM is computed on fields where land is masked and NaNs are filled with
      the mean of valid values for that day (or 0 if none).
    - Daily metric values within a month are averaged via ``np.nanmean``.
    - The resulting figure shows four panels (R², Bias, MAE, SSIM) with month
      on the x-axis (1–12) and one line per year.
    """
    # helper
    def fetch_2d(ds, var, date, like_var):
        arr = ds[var]
        arr_t = arr.sel(time=date) if 'time' in arr.dims else arr
        arr_t = arr_t.broadcast_like(ds[like_var].sel(time=date))
        # ensure spatial order is (lat, lon)
        if tuple(d for d in arr_t.dims if d != 'time') != ('lat', 'lon'):
            arr_t = (
                arr_t.transpose(..., 'lat', 'lon')
                if 'time' in arr_t.dims
                else arr_t.transpose('lat', 'lon')
            )
        return arr_t.values.astype('float32', copy=False)

    # target days within each month
    target_days = {1, 7, 14, 28}

    metrics = ['r2', 'bias', 'mae', 'ssim']
    metric_by_year_month = {m: {} for m in metrics}

    for year in years:
        ds = data.sel(time=year)
        dates = pd.to_datetime(ds.time.values)

        gb = pd.Series(dates).groupby([dates.year, dates.month])

        scores_dict = {m: [] for m in metrics}
        months_list = []

        for (yy, mm), group in gb:
            month_dates = list(group)
            pick = [d for d in month_dates if d.day in target_days]
            if len(pick) == 0:
                continue

            chans_list = []
            truths = []
            lands = []
            pred_list = []

            for date in pick:
                chans = []
                for k, v in enumerate(num_var):
                    a = fetch_2d(ds, v, date, y_name)
                    if X_mean is not None and X_std is not None:
                        denom = 1.0 if X_std[k] == 0 else X_std[k]
                        a = (a - X_mean[k]) / denom
                    chans.append(np.nan_to_num(a))
                for v in cat_var:
                    a = fetch_2d(ds, v, date, y_name)
                    chans.append(np.nan_to_num(a))

                X_map = np.stack(chans, axis=-1)  # (H, W, C)
                truths.append(fetch_2d(ds, y_name, date, y_name))
                lands.append(fetch_2d(ds, mask_var, date, y_name) == 0.0)

                if model_type == "cnn":
                    chans_list.append(X_map)
                elif model_type == "brt":
                    H, W, C = X_map.shape
                    X_flat = X_map.reshape(-1, C)
                    pred_flat = model.predict(X_flat)
                    pred_list.append(pred_flat.reshape(H, W))
                else:
                    raise ValueError(f"Unknown model_type: {model_type!r}")

            # ---- predict batch
            if model_type == "cnn":
                X_batch = np.stack(chans_list, axis=0)  # (B, H, W, C)
                pred_batch = model.predict(X_batch, verbose=0)
                if pred_batch.ndim == 4 and pred_batch.shape[-1] == 1:
                    pred_batch = pred_batch[..., 0]  # (B, H, W)
            else:  # brt
                pred_batch = np.stack(pred_list, axis=0)  # (B, H, W)

            # ---- metrics
            r2_vals, bias_vals, mae_vals, ssim_vals = [], [], [], []
            for b, date in enumerate(pick):
                truth = np.where(lands[b], np.nan, truths[b])
                pred = np.where(lands[b], np.nan, pred_batch[b])

                m = ~np.isnan(truth) & ~np.isnan(pred)

                if m.any():
                    r2_vals.append(r2_score(truth[m].ravel(), pred[m].ravel()))
                    mae_vals.append(float(mean_absolute_error(truth[m], pred[m])))
                    bias_vals.append(float(np.mean(pred[m] - truth[m])))
                else:
                    r2_vals.append(np.nan)
                    mae_vals.append(np.nan)
                    bias_vals.append(np.nan)

                t = np.nan_to_num(
                    truth,
                    nan=(np.nanmean(truth) if np.isfinite(truth).any() else 0.0),
                )
                p = np.nan_to_num(
                    pred,
                    nan=(np.nanmean(pred) if np.isfinite(pred).any() else 0.0),
                )
                dr = np.nanmax(truth) - np.nanmin(truth)
                if not np.isfinite(dr) or dr == 0:
                    dr = (np.nanmax(t) - np.nanmin(t)) or 1.0
                ssim_kwargs = {"data_range": dr}
                if ssim_win_size is not None:
                    ssim_kwargs["win_size"] = int(ssim_win_size)
                if ssim_sigma is not None:
                    ssim_kwargs["gaussian_weights"] = True
                    ssim_kwargs["sigma"] = float(ssim_sigma)
                ssim_vals.append(
                    ssim(t.astype(np.float64), p.astype(np.float64), **ssim_kwargs)
                )

            months_list.append(mm)
            scores_dict['r2'].append(np.nanmean(r2_vals))
            scores_dict['bias'].append(np.nanmean(bias_vals))
            scores_dict['mae'].append(np.nanmean(mae_vals))
            scores_dict['ssim'].append(np.nanmean(ssim_vals))

        months = np.array(months_list, dtype=int)
        for m in metrics:
            metric_by_year_month[m][year] = (months, scores_dict[m])

    # ---- plotting: 2x2 panel ----
    titles = {'r2': "$R^2$", 'bias': "Bias", 'mae': "MAE", 'ssim': "SSIM"}
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = dict(zip(metrics, axs.ravel()))

    for m in metrics:
        ax = axes[m]
        for year, (months, scores) in metric_by_year_month[m].items():
            label = f"{year} (train)" if year == training_year else year
            style = "--" if year == training_year else "-"
            ax.plot(months, scores, style, marker='o', label=label)
        ax.set_title(f"Monthly {titles[m]}")
        ax.set_xlabel("Month")
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(calendar.month_abbr[1:13])
        ax.grid(True)
        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin, ymax)

    handles, labels = axes['r2'].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(len(labels), 6),
            frameon=False,
        )
    plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from skimage.metrics import structural_similarity as ssim

def evaluate_year_batched(
    data, year, model, X_mean, X_std, num_var, cat_var,
    y_name='y', mask_var='ocean_mask',
    ssim_win_size=None, ssim_sigma=None,
    batch_size=16, model_type='cnn'  # 'cnn' or 'tabular'
):
    """
    Predicts all days in `year` in batches and returns a DataFrame of per-day metrics
    plus a monthly-aggregated summary.

    Returns
    -------
    daily_df : pd.DataFrame with columns ['date','year','month','r2','mae','bias','ssim']
    monthly_df : pd.DataFrame with index (year,month) and mean metrics over days
    """
    ds = data.sel(time=str(year))
    dates = pd.to_datetime(ds.time.values)

    # --- helper to fetch (H,W) at a date; broadcast if var has no time
    def fetch_2d(var, date):
        arr = ds[var]
        arr_t = arr.sel(time=date) if 'time' in arr.dims else arr
        arr_t = arr_t.broadcast_like(ds[y_name].sel(time=date))
        # ensure spatial order is (lat, lon)
        if tuple(d for d in arr_t.dims if d != 'time') != ('lat','lon'):
            arr_t = arr_t.transpose(..., 'lat', 'lon') if 'time' in arr_t.dims else arr_t.transpose('lat','lon')
        return arr_t.values.astype('float32', copy=False)

    # --- build all-day inputs (but predict in mini-batches)
    H, W = int(ds.sizes['lat']), int(ds.sizes['lon'])
    Cn, Cc = len(num_var), len(cat_var)
    C = Cn + Cc

    # We’ll lazily fill a list of batches: [(X_batch, truth_batch, land_batch), ...]
    # to avoid storing the entire year in one big array.
    daily_records = []  # store (date, r2, mae, bias, ssim)
    idx = 0
    while idx < len(dates):
        j = min(idx + batch_size, len(dates))
        batch_dates = dates[idx:j]

        # Build this batch
        X_batch = np.empty((len(batch_dates), H, W, C), dtype='float32')
        truth_batch = np.empty((len(batch_dates), H, W), dtype='float32')
        land_batch = np.empty((len(batch_dates), H, W), dtype=bool)

        for b, date in enumerate(batch_dates):
            chans = []
            # numeric (normalize, fill NaNs with 0 for inputs)
            for k, v in enumerate(num_var):
                a = fetch_2d(v, date)
                denom = 1.0 if X_std[k] == 0 else X_std[k]
                a = (a - X_mean[k]) / denom
                chans.append(np.nan_to_num(a))
            # categorical (just fill NaNs with 0)
            for v in cat_var:
                a = fetch_2d(v, date)
                chans.append(np.nan_to_num(a))
            X_batch[b] = np.stack(chans, axis=-1)

            truth_batch[b] = fetch_2d(y_name, date)
            land_batch[b]  = (fetch_2d(mask_var, date) == 0.0)

        # Predict this batch
        if model_type == 'cnn':
            pred_batch = model.predict(X_batch, verbose=0)
            if pred_batch.ndim == 4 and pred_batch.shape[-1] == 1:
                pred_batch = pred_batch[..., 0]
        else:  # tabular (e.g., BRT)
            pred_batch = np.empty((len(batch_dates), H, W), dtype='float32')
            for b in range(len(batch_dates)):
                pred_batch[b] = model.predict(X_batch[b][np.newaxis, ...], verbose=0)[0][..., 0]

        # Compute metrics per day
        for b, date in enumerate(batch_dates):
            truth = np.where(land_batch[b], np.nan, truth_batch[b])
            pred  = np.where(land_batch[b], np.nan, pred_batch[b])

            m = ~np.isnan(truth) & ~np.isnan(pred)
            if m.any():
                r2   = r2_score(truth[m].ravel(), pred[m].ravel())
                mae  = float(mean_absolute_error(truth[m], pred[m]))
                bias = float(np.mean(pred[m] - truth[m]))
            else:
                r2 = mae = bias = np.nan

            # SSIM (fill NaNs for SSIM only)
            t = np.nan_to_num(truth, nan=(np.nanmean(truth) if np.isfinite(truth).any() else 0.0))
            p = np.nan_to_num(pred,  nan=(np.nanmean(pred)  if np.isfinite(pred).any()  else 0.0))
            dr = np.nanmax(truth) - np.nanmin(truth)
            if not np.isfinite(dr) or dr == 0:
                dr = (np.nanmax(t) - np.nanmin(t)) or 1.0
            ssim_kwargs = {"data_range": dr}
            if ssim_win_size is not None:
                ssim_kwargs["win_size"] = int(ssim_win_size)
            if ssim_sigma is not None:
                ssim_kwargs.update(gaussian_weights=True, sigma=float(ssim_sigma))
            ssim_val = ssim(t.astype(np.float64), p.astype(np.float64), **ssim_kwargs)

            daily_records.append((date, date.year, date.month, r2, mae, bias, ssim_val))

        idx = j  # next batch

    daily_df = pd.DataFrame(daily_records, columns=["date","year","month","r2","mae","bias","ssim"])
    monthly_df = (daily_df
                  .groupby(["year","month"], as_index=True)[["r2","mae","bias","ssim"]]
                  .mean())

    return daily_df, monthly_df


## MODEL FITTING

from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

def train_brt_from_splits(X_train, y_train, feature_names, ocean_channel='ocean_mask',
                          max_samples=300_000, random_state=42,
                          max_depth=6, learning_rate=0.05, max_iter=400,
                          l2_regularization=0.0,
                          grid_shape=None):   # <-- NEW
    T, H, W, C = X_train.shape
    if grid_shape is None:
        grid_shape = (H, W)                  # fallback to training grid

    X2 = X_train.reshape(-1, C)
    y2 = y_train.reshape(-1)

    oce_idx = feature_names.index(ocean_channel)
    valid = (X2[:, oce_idx] > 0.5) & np.isfinite(y2) & np.all(np.isfinite(X2), axis=1)

    X2v, y2v = X2[valid], y2[valid]
    if X2v.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(X2v.shape[0], size=max_samples, replace=False)
        X2v, y2v = X2v[sel], y2v[sel]

    brt = HistGradientBoostingRegressor(
        max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter,
        l2_regularization=l2_regularization,
        random_state=random_state, validation_fraction=0.1, early_stopping=True
    ).fit(X2v, y2v)

    class BRTWrapper:
        def __init__(self, base, grid_shape):
            self.base = base
            self.grid_shape = tuple(grid_shape)  # (H, W)
        def predict(self, X4, verbose=0):
            X3 = X4[0] if (hasattr(X4, "ndim") and X4.ndim == 4) else X4
            C  = X3.shape[-1]
            flat = self.base.predict(X3.reshape(-1, C))
            H, W = self.grid_shape
            return flat.reshape(1, H, W, 1)  # <-- add batch axis like Keras
        
    return brt, BRTWrapper(brt, grid_shape)


# UTILITIES

def add_latlon_2d(ds):
    import numpy as np
    lat2d, lon2d = np.meshgrid(ds.lat.values, ds.lon.values, indexing='ij')
    return ds.assign(
        lat2d=(('lat','lon'), lat2d.astype('float32')),
        lon2d=(('lat','lon'), lon2d.astype('float32')),
    )

def add_sin_coords(ds):
    import numpy as np, xarray as xr
    lat2d, lon2d = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    lonr = np.deg2rad(lon2d); latr = np.deg2rad(lat2d)
    return ds.assign(
        lon_sin=(('lat','lon'), np.sin(lonr).astype('float32')),
        lon_cos=(('lat','lon'), np.cos(lonr).astype('float32')),
        lat_sin=(('lat','lon'), np.sin(latr).astype('float32')),
    )

import xarray as xr
import numpy as np

def add_spherical_coords(ds, lat="lat", lon="lon"):
    """
    Add 3D unit-sphere coordinates (x_geo, y_geo, z_geo) computed from lat/lon.

    Works with Dask-backed or in-memory xarray Datasets.
    No .values used; stays lazy when possible.

    Returns
    -------
    xarray.Dataset
        Original ds plus float32 x_geo, y_geo, z_geo on (lat, lon).
    """
    # Make 2D lat/lon without materializing into NumPy arrays
    lat2d, lon2d = xr.broadcast(ds[lat], ds[lon])          # (lat, lon), (lat, lon)

    # radians (lazy if dask)
    psi = xr.apply_ufunc(np.deg2rad, lat2d, dask="parallelized")
    lam = xr.apply_ufunc(np.deg2rad, lon2d, dask="parallelized")

    x_geo = xr.apply_ufunc(np.cos, psi, dask="parallelized") * xr.apply_ufunc(np.cos, lam, dask="parallelized")
    y_geo = xr.apply_ufunc(np.cos, psi, dask="parallelized") * xr.apply_ufunc(np.sin, lam, dask="parallelized")
    z_geo = xr.apply_ufunc(np.sin, psi, dask="parallelized")

    return ds.assign(
        x_geo=x_geo.astype("float32"),
        y_geo=y_geo.astype("float32"),
        z_geo=z_geo.astype("float32"),
    )
    
import xarray as xr
import numpy as np

def add_seasonal_time_features(dataset, ref_var="sst"):
    """
    Add sin_time and cos_time (seasonal cycle) on (time, lat, lon).

    - Uses xarray's datetime accessor (works with numpy/cftime)
    - Broadcasts across space using xarray, preserving Dask chunking when present
    - If `ref_var` isn't in ds, tries to infer a 3D (time, lat, lon) variable

    Parameters
    ----------
    dataset : xr.Dataset
    ref_var : str
        Name of a 3D variable to define the (time, lat, lon) shape for broadcasting.

    Returns
    -------
    xr.Dataset
        With sin_time and cos_time (float32).
    """
    if "time" not in dataset.coords:
        raise ValueError("Dataset has no 'time' coordinate.")

    if ref_var not in dataset.data_vars:
        # try to infer a suitable 3D var
        candidates = [v for v, da in dataset.data_vars.items() if set(da.dims) >= {"time", "lat", "lon"}]
        if not candidates:
            raise ValueError("Could not find a 3D (time, lat, lon) variable to broadcast across. "
                             "Pass ref_var='<your_var>' explicitly.")
        ref_var = candidates[0]

    # day-of-year (1..366). Works with numpy/cftime calendars.
    doy = dataset["time"].dt.dayofyear

    # Use a smooth year length; change to 365 or 366 if you prefer strict calendars
    rad = 2 * np.pi * (doy / 365.25)

    sin_t = xr.apply_ufunc(np.sin, rad, dask="parallelized").astype("float32")
    cos_t = xr.apply_ufunc(np.cos, rad, dask="parallelized").astype("float32")

    # Broadcast time-only arrays to (time, lat, lon) using xarray
    sin_3d, _ = xr.broadcast(sin_t, dataset[ref_var])
    cos_3d, _ = xr.broadcast(cos_t, dataset[ref_var])

    return dataset.assign(
        sin_time=sin_3d,
        cos_time=cos_3d,
    )
    
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

def add_distance_to_coast(ds: xr.Dataset,
                          mask_var="ocean_mask",
                          out_name="dist2coast_km") -> xr.Dataset:
    """
    Add distance-to-coast (km) for ocean pixels; 0 on land.
    Assumes lat/lon on a regular grid. Uses EDT with anisotropic sampling.
    """
    if mask_var not in ds:
        raise KeyError(f"{mask_var} not found in dataset.")
    if not {"lat","lon"} <= set(ds.coords):
        raise KeyError("Dataset must have 'lat' and 'lon' coordinates.")

    # 2D boolean masks (lat, lon)
    ocean = (ds[mask_var].astype(bool))
    if "time" in ocean.dims:
        ocean2d = ocean.isel(time=0)  # mask is time-invariant in your data
    else:
        ocean2d = ocean

    # Build per-axis sampling (km per pixel). Use mean spacing + mean latitude.
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    dlat = np.abs(np.diff(lat_vals)).mean() if lat_vals.size > 1 else 0.0
    dlon = np.abs(np.diff(lon_vals)).mean() if lon_vals.size > 1 else 0.0
    lat0  = float(np.mean(lat_vals))  # for lon scaling
    km_per_deg = 111.0
    dy_km = dlat * km_per_deg
    dx_km = dlon * km_per_deg * max(np.cos(np.deg2rad(lat0)), 1e-6)  # avoid 0 at poles

    # Distance from ocean -> nearest land (0 on land)
    # EDT expects True for the "non-zero" region to measure distance *to zero* pixels.
    # We want distance to land, so zero = land, one = ocean.
    land2d = ~ocean2d.values
    dist_km = distance_transform_edt(~land2d, sampling=(dy_km, dx_km))  # ~land2d == ocean
    dist_km[land2d] = 0.0  # exactly zero on land

    # Wrap into DataArray and attach
    dist_da = xr.DataArray(dist_km.astype("float32"),
                           dims=("lat","lon"),
                           coords={"lat": ds["lat"], "lon": ds["lon"]},
                           name=out_name,
                           attrs={"units":"km", "long_name":"distance to coast (over ocean)"})
    return ds.assign({out_name: dist_da})

import numpy as np
import pandas as pd
import xarray as xr

def count_valid_days_by_month(
    ds: xr.Dataset,
    year,
    vars_to_check=("y",),          # str or iterable of str
    mask_var="ocean_mask",         # 1=ocean, 0=land
    nan_max_frac=0.05,             # allow ≤ 5% NaNs over ocean
):
    """
    Return a Series with counts of days per month that pass the NaN filter.
    A day passes iff, for every variable in `vars_to_check`, the number of
    NaNs over the ocean is <= nan_max_frac * ocean_pixels_on_that_day.
    """
    # subset to the year
    dsy = ds.sel(time=str(year))

    # tidy inputs
    if isinstance(vars_to_check, str):
        vars_to_check = [vars_to_check]

    # spatial mask, broadcast to (time, lat, lon)
    ocean = dsy[mask_var].astype(bool)
    if "time" not in ocean.dims:
        ocean = ocean.expand_dims(time=dsy["time"])
    # broadcast to the target grid (uses first var as template)
    template = dsy[vars_to_check[0]] if vars_to_check else dsy["y"]
    ocean = ocean.broadcast_like(template)

    # how many ocean pixels each day?
    spatial_dims = [d for d in ocean.dims if d != "time"]
    ocean_count = ocean.sum(dim=spatial_dims)  # (time,)
    nan_thresh_t = nan_max_frac * ocean_count  # per time-step threshold

    # build per-var validity (True if NaN count over ocean <= threshold)
    valid_per_var = []
    for v in vars_to_check:
        if v not in dsy:
            raise KeyError(f"Variable '{v}' not in dataset.")
        arr = dsy[v]
        if "time" not in arr.dims:
            arr = arr.expand_dims(time=dsy["time"])
        arr = arr.broadcast_like(ocean)
        v_nan = xr.apply_ufunc(np.isnan, arr) & ocean
        v_nan_count = v_nan.sum(dim=spatial_dims)  # (time,)
        valid_per_var.append(v_nan_count <= nan_thresh_t)

    # day is valid if all vars pass
    valid_all = xr.concat(valid_per_var, dim="var").all(dim="var")  # (time,) bool

    # group by month and count
    month_idx = pd.to_datetime(valid_all["time"].values).month
    counts = pd.Series(valid_all.values, index=month_idx).groupby(level=0).sum().astype(int)
    # ensure all months present
    counts = counts.reindex(range(1,13), fill_value=0)
    counts.index.name = "month"
    counts.name = f"valid_days_{year}"
    return counts

import xarray as xr
import numpy as np
import pandas as pd

def pct_missing_by_day_year(
    ds: xr.Dataset,
    year: int,
    var: str = "y",
    mask_var: str = "ocean_mask",   # 1=ocean, 0=land
) -> pd.Series:
    """
    Return a Series with the percent of ocean pixels that are NaN
    for each day in the given year.

    Each value is:
        (# NaN pixels over ocean) / (# ocean pixels) * 100
    for the variable `var`.
    """
    # subset to year explicitly
    dsy = ds.sel(time=ds["time"].dt.year == year)
    if dsy.sizes.get("time", 0) == 0:
        raise ValueError(f"No data found for year {year}.")

    # ocean mask -> bool, broadcast to (time, lat, lon, ...)
    ocean = dsy[mask_var].astype(bool)
    if "time" not in ocean.dims:
        ocean = ocean.expand_dims(time=dsy["time"])

    arr = dsy[var]
    if "time" not in arr.dims:
        arr = arr.expand_dims(time=dsy["time"])

    ocean = ocean.broadcast_like(arr)

    # spatial dims (everything except time)
    spatial_dims = [d for d in ocean.dims if d != "time"]

    # counts per day
    ocean_count = ocean.sum(dim=spatial_dims)          # (time,)
    nan_mask = arr.isnull() & ocean
    nan_count = nan_mask.sum(dim=spatial_dims)         # (time,)

    frac = nan_count / ocean_count                     # (time,), 0–1
    pct = (frac * 100).to_series()
    pct.name = f"pct_missing_{var}_{year}"

    return pct