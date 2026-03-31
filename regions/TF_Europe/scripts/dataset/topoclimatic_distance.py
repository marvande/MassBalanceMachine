import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler


# def to_feature_matrix(
#     df: pd.DataFrame,
#     feature_cols: list[str],
#     dropna: bool = True,
# ) -> np.ndarray:
#     """Return X (n_samples, n_features) as float64."""
#     missing = [c for c in feature_cols if c not in df.columns]
#     if missing:
#         raise KeyError(f"Missing feature columns in df: {missing}")

#     X = df[feature_cols].to_numpy(dtype=np.float64, copy=True)

#     if dropna:
#         mask = np.isfinite(X).all(axis=1)
#         X = X[mask]

#     return X


# def wasserstein_distance_multivariate(
#     X: np.ndarray,
#     Y: np.ndarray,
# ) -> float:
#     """
#     Mean 1D Wasserstein distance across features.
#     Assumes features are already standardized.
#     """
#     if X.size == 0 or Y.size == 0:
#         return np.nan

#     dists = [wasserstein_distance(X[:, j], Y[:, j]) for j in range(X.shape[1])]
#     return float(np.mean(dists))


# def compute_topoclimatic_distances(
#     df_src: pd.DataFrame,
#     df_pool: pd.DataFrame,
#     df_holdout: pd.DataFrame,
#     feature_cols: list[str],
#     scaler: StandardScaler,
#     seed: int,
#     energy_max_samples: int = 4000,
#     exclude_cols: list[str] | None = None,
#     topo_cols: list[str] | None = None,
#     climate_cols: list[str] | None = None,
# ) -> dict:
#     """
#     Compute topoclimatic distances in a globally standardized feature space.

#     Distances returned:
#       - src vs pool
#       - src vs holdout
#       - pool vs holdout
#       - src vs tgt_all (pool ∪ holdout)
#       - optionally: topo-only and climate-only subsets of the above

#     Parameters
#     ----------
#     df_src, df_pool, df_holdout : pd.DataFrame
#         Source / pool / holdout dataframes.
#     feature_cols : list[str]
#         Features used for distance computation.
#     scaler : StandardScaler
#         Pre-fitted global scaler, fitted once across all regions/settings.
#     seed : int
#         Random seed for energy-distance subsampling.
#     energy_max_samples : int
#         Max number of samples per set for energy distance.
#     exclude_cols : list[str] | None
#         Features to exclude from feature_cols before computation.
#     topo_cols : list[str] | None
#         Subset of feature_cols for topo-only distances. If None, skipped.
#     climate_cols : list[str] | None
#         Subset of feature_cols for climate-only distances. If None, skipped.

#     Returns
#     -------
#     dict
#     """
#     effective_cols = [f for f in feature_cols if f not in (exclude_cols or [])]

#     df_tgt_all = pd.concat([df_pool, df_holdout], axis=0, ignore_index=True)

#     Xs = to_feature_matrix(df_src, effective_cols, dropna=True)
#     Xp = to_feature_matrix(df_pool, effective_cols, dropna=True)
#     Xh = to_feature_matrix(df_holdout, effective_cols, dropna=True)
#     Xt = to_feature_matrix(df_tgt_all, effective_cols, dropna=True)

#     if Xs.size == 0 or Xp.size == 0 or Xh.size == 0 or Xt.size == 0:
#         return {
#             "n_src": int(Xs.shape[0]),
#             "n_pool": int(Xp.shape[0]),
#             "n_holdout": int(Xh.shape[0]),
#             "n_tgt_all": int(Xt.shape[0]),
#             "D_energy_src_pool": np.nan,
#             "D_energy_src_holdout": np.nan,
#             "D_energy_pool_holdout": np.nan,
#             "D_energy_src_tgt_all": np.nan,
#             "D_wass_src_pool": np.nan,
#             "D_wass_src_holdout": np.nan,
#             "D_wass_pool_holdout": np.nan,
#             "D_wass_src_tgt_all": np.nan,
#         }

#     # Transform with one shared global scaler
#     Xs_z = scaler.transform(Xs)
#     Xp_z = scaler.transform(Xp)
#     Xh_z = scaler.transform(Xh)
#     Xt_z = scaler.transform(Xt)

#     # Energy distance
#     d_en_src_pool = energy_distance(Xs_z,
#                                     Xp_z,
#                                     max_samples=energy_max_samples,
#                                     seed=seed + 1)
#     d_en_src_hold = energy_distance(Xs_z,
#                                     Xh_z,
#                                     max_samples=energy_max_samples,
#                                     seed=seed + 2)
#     d_en_pool_hold = energy_distance(Xp_z,
#                                      Xh_z,
#                                      max_samples=energy_max_samples,
#                                      seed=seed + 3)
#     d_en_src_tgt = energy_distance(Xs_z,
#                                    Xt_z,
#                                    max_samples=energy_max_samples,
#                                    seed=seed + 4)

#     # Wasserstein distance
#     d_wass_src_pool = wasserstein_distance_multivariate(Xs_z, Xp_z)
#     d_wass_src_hold = wasserstein_distance_multivariate(Xs_z, Xh_z)
#     d_wass_pool_hold = wasserstein_distance_multivariate(Xp_z, Xh_z)
#     d_wass_src_tgt = wasserstein_distance_multivariate(Xs_z, Xt_z)

#     out = {
#         "n_src": int(Xs.shape[0]),
#         "n_pool": int(Xp.shape[0]),
#         "n_holdout": int(Xh.shape[0]),
#         "n_tgt_all": int(Xt.shape[0]),
#         "D_energy_src_pool": float(d_en_src_pool),
#         "D_energy_src_holdout": float(d_en_src_hold),
#         "D_energy_pool_holdout": float(d_en_pool_hold),
#         "D_energy_src_tgt_all": float(d_en_src_tgt),
#         "D_wass_src_pool": float(d_wass_src_pool),
#         "D_wass_src_holdout": float(d_wass_src_hold),
#         "D_wass_pool_holdout": float(d_wass_pool_hold),
#         "D_wass_src_tgt_all": float(d_wass_src_tgt),
#     }

#     # --- topo-only distances ---
#     if topo_cols:
#         topo_idx = [
#             effective_cols.index(f) for f in topo_cols if f in effective_cols
#         ]
#         if topo_idx:
#             out["D_energy_src_pool_topo"] = float(
#                 energy_distance(
#                     Xs_z[:, topo_idx],
#                     Xp_z[:, topo_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 10,
#                 ))
#             out["D_energy_src_holdout_topo"] = float(
#                 energy_distance(
#                     Xs_z[:, topo_idx],
#                     Xh_z[:, topo_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 11,
#                 ))
#             out["D_energy_pool_holdout_topo"] = float(
#                 energy_distance(
#                     Xp_z[:, topo_idx],
#                     Xh_z[:, topo_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 12,
#                 ))
#             out["D_energy_src_tgt_all_topo"] = float(
#                 energy_distance(
#                     Xs_z[:, topo_idx],
#                     Xt_z[:, topo_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 13,
#                 ))
#             out["D_wass_src_pool_topo"] = float(
#                 wasserstein_distance_multivariate(Xs_z[:, topo_idx],
#                                                   Xp_z[:, topo_idx]))
#             out["D_wass_src_holdout_topo"] = float(
#                 wasserstein_distance_multivariate(Xs_z[:, topo_idx],
#                                                   Xh_z[:, topo_idx]))
#             out["D_wass_pool_holdout_topo"] = float(
#                 wasserstein_distance_multivariate(Xp_z[:, topo_idx],
#                                                   Xh_z[:, topo_idx]))
#             out["D_wass_src_tgt_all_topo"] = float(
#                 wasserstein_distance_multivariate(Xs_z[:, topo_idx],
#                                                   Xt_z[:, topo_idx]))

#     # --- climate-only distances ---
#     if climate_cols:
#         clim_idx = [
#             effective_cols.index(f) for f in climate_cols
#             if f in effective_cols
#         ]
#         if clim_idx:
#             out["D_energy_src_pool_clim"] = float(
#                 energy_distance(
#                     Xs_z[:, clim_idx],
#                     Xp_z[:, clim_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 20,
#                 ))
#             out["D_energy_src_holdout_clim"] = float(
#                 energy_distance(
#                     Xs_z[:, clim_idx],
#                     Xh_z[:, clim_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 21,
#                 ))
#             out["D_energy_pool_holdout_clim"] = float(
#                 energy_distance(
#                     Xp_z[:, clim_idx],
#                     Xh_z[:, clim_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 22,
#                 ))
#             out["D_energy_src_tgt_all_clim"] = float(
#                 energy_distance(
#                     Xs_z[:, clim_idx],
#                     Xt_z[:, clim_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 23,
#                 ))
#             out["D_wass_src_pool_clim"] = float(
#                 wasserstein_distance_multivariate(Xs_z[:, clim_idx],
#                                                   Xp_z[:, clim_idx]))
#             out["D_wass_src_holdout_clim"] = float(
#                 wasserstein_distance_multivariate(Xs_z[:, clim_idx],
#                                                   Xh_z[:, clim_idx]))
#             out["D_wass_pool_holdout_clim"] = float(
#                 wasserstein_distance_multivariate(Xp_z[:, clim_idx],
#                                                   Xh_z[:, clim_idx]))
#             out["D_wass_src_tgt_all_clim"] = float(
#                 wasserstein_distance_multivariate(Xs_z[:, clim_idx],
#                                                   Xt_z[:, clim_idx]))

#     return out


# def compute_topoclimatic_distances_sets(
#     df_src: pd.DataFrame,
#     df_ft: pd.DataFrame,
#     feature_cols: list[str],
#     scaler: StandardScaler,
#     seed: int,
#     df_holdout: pd.DataFrame | None = None,
#     energy_max_samples: int = 4000,
#     exclude_cols: list[str] | None = None,
#     topo_cols: list[str] | None = None,
#     climate_cols: list[str] | None = None,
# ) -> dict:
#     """
#     Compute multivariate topoclimatic distances between source, fine-tuning,
#     and optionally hold-out distributions.

#     Parameters
#     ----------
#     df_src : pd.DataFrame
#         Source-region samples.
#     df_ft : pd.DataFrame
#         Fine-tuning / monitoring samples.
#     feature_cols : list[str]
#         Feature columns used to build the multivariate distribution.
#     scaler : StandardScaler
#         Fitted scaler used to standardize all feature matrices in the same space.
#         This should normally be fitted once on a reference dataset and reused.
#     seed : int
#         Random seed used for subsampling in the energy distance computation.
#     df_holdout : pd.DataFrame | None, optional
#         Hold-out / evaluation samples. If provided, distances involving the
#         hold-out set are also computed.
#     energy_max_samples : int, default=4000
#         Maximum number of samples used inside the energy distance computation.
#     exclude_cols : list[str] | None
#         Features to exclude from feature_cols before computation.
#     topo_cols : list[str] | None
#         Subset of feature_cols for topo-only distances. If None, skipped.
#     climate_cols : list[str] | None
#         Subset of feature_cols for climate-only distances. If None, skipped.

#     Returns
#     -------
#     dict
#         Dictionary containing sample counts and pairwise distances.
#     """
#     effective_cols = [f for f in feature_cols if f not in (exclude_cols or [])]

#     Xs = to_feature_matrix(df_src, effective_cols, dropna=True)
#     Xf = to_feature_matrix(df_ft, effective_cols, dropna=True)

#     Xh = None
#     if df_holdout is not None:
#         Xh = to_feature_matrix(df_holdout, effective_cols, dropna=True)

#     out = {
#         "n_src": int(Xs.shape[0]),
#         "n_ft": int(Xf.shape[0]),
#         "n_holdout": int(Xh.shape[0]) if Xh is not None else np.nan,
#         "D_energy_src_ft": np.nan,
#         "D_wass_src_ft": np.nan,
#         "D_energy_ft_holdout": np.nan,
#         "D_wass_ft_holdout": np.nan,
#         "D_energy_src_holdout": np.nan,
#         "D_wass_src_holdout": np.nan,
#     }

#     if Xs.size == 0 or Xf.size == 0:
#         return out

#     Xs_z = scaler.transform(Xs)
#     Xf_z = scaler.transform(Xf)

#     out["D_energy_src_ft"] = float(
#         energy_distance(Xs_z,
#                         Xf_z,
#                         max_samples=energy_max_samples,
#                         seed=seed + 101))
#     out["D_wass_src_ft"] = float(wasserstein_distance_multivariate(Xs_z, Xf_z))

#     Xh_z = None
#     if Xh is not None and Xh.size > 0:
#         Xh_z = scaler.transform(Xh)

#         out["D_energy_ft_holdout"] = float(
#             energy_distance(Xf_z,
#                             Xh_z,
#                             max_samples=energy_max_samples,
#                             seed=seed + 202))
#         out["D_wass_ft_holdout"] = float(
#             wasserstein_distance_multivariate(Xf_z, Xh_z))
#         out["D_energy_src_holdout"] = float(
#             energy_distance(Xs_z,
#                             Xh_z,
#                             max_samples=energy_max_samples,
#                             seed=seed + 303))
#         out["D_wass_src_holdout"] = float(
#             wasserstein_distance_multivariate(Xs_z, Xh_z))

#     # --- topo-only distances ---
#     if topo_cols:
#         topo_idx = [
#             effective_cols.index(f) for f in topo_cols if f in effective_cols
#         ]
#         if topo_idx:
#             out["D_energy_src_ft_topo"] = float(
#                 energy_distance(
#                     Xs_z[:, topo_idx],
#                     Xf_z[:, topo_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 110,
#                 ))
#             out["D_wass_src_ft_topo"] = float(
#                 wasserstein_distance_multivariate(Xs_z[:, topo_idx],
#                                                   Xf_z[:, topo_idx]))
#             if Xh_z is not None:
#                 out["D_energy_ft_holdout_topo"] = float(
#                     energy_distance(
#                         Xf_z[:, topo_idx],
#                         Xh_z[:, topo_idx],
#                         max_samples=energy_max_samples,
#                         seed=seed + 211,
#                     ))
#                 out["D_wass_ft_holdout_topo"] = float(
#                     wasserstein_distance_multivariate(Xf_z[:, topo_idx],
#                                                       Xh_z[:, topo_idx]))
#                 out["D_energy_src_holdout_topo"] = float(
#                     energy_distance(
#                         Xs_z[:, topo_idx],
#                         Xh_z[:, topo_idx],
#                         max_samples=energy_max_samples,
#                         seed=seed + 312,
#                     ))
#                 out["D_wass_src_holdout_topo"] = float(
#                     wasserstein_distance_multivariate(Xs_z[:, topo_idx],
#                                                       Xh_z[:, topo_idx]))

#     # --- climate-only distances ---
#     if climate_cols:
#         clim_idx = [
#             effective_cols.index(f) for f in climate_cols
#             if f in effective_cols
#         ]
#         if clim_idx:
#             out["D_energy_src_ft_clim"] = float(
#                 energy_distance(
#                     Xs_z[:, clim_idx],
#                     Xf_z[:, clim_idx],
#                     max_samples=energy_max_samples,
#                     seed=seed + 120,
#                 ))
#             out["D_wass_src_ft_clim"] = float(
#                 wasserstein_distance_multivariate(Xs_z[:, clim_idx],
#                                                   Xf_z[:, clim_idx]))
#             if Xh_z is not None:
#                 out["D_energy_ft_holdout_clim"] = float(
#                     energy_distance(
#                         Xf_z[:, clim_idx],
#                         Xh_z[:, clim_idx],
#                         max_samples=energy_max_samples,
#                         seed=seed + 221,
#                     ))
#                 out["D_wass_ft_holdout_clim"] = float(
#                     wasserstein_distance_multivariate(Xf_z[:, clim_idx],
#                                                       Xh_z[:, clim_idx]))
#                 out["D_energy_src_holdout_clim"] = float(
#                     energy_distance(
#                         Xs_z[:, clim_idx],
#                         Xh_z[:, clim_idx],
#                         max_samples=energy_max_samples,
#                         seed=seed + 322,
#                     ))
#                 out["D_wass_src_holdout_clim"] = float(
#                     wasserstein_distance_multivariate(Xs_z[:, clim_idx],
#                                                       Xh_z[:, clim_idx]))

#     return out


def compute_wasserstein_per_var(df_src, df_tgt, cols, id_col="ID"):
    result = {}
    for col in cols:
        # CURRENT: x = df_src.groupby(id_col)[col].mean().dropna().values
        # FIX:
        x = df_src[col].dropna().values
        y = df_tgt[col].dropna().values
        mu, sd = np.concatenate([x, y]).mean(), np.concatenate([x, y]).std()
        result[col] = wasserstein_distance((x - mu) / sd, (y - mu) / sd)
    return result


def energy_distance(
    X: np.ndarray,
    Y: np.ndarray,
    max_samples: int = 5000,
    seed: int = 0,
) -> float:
    """
    Energy distance between two multivariate samples.

    Returns sqrt(ed2), where:
        ed2 = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||
    """
    if X.size == 0 or Y.size == 0:
        return np.nan

    rng = np.random.default_rng(seed)

    def _subsample(A: np.ndarray) -> np.ndarray:
        if A.shape[0] <= max_samples:
            return A
        idx = rng.choice(A.shape[0], size=max_samples, replace=False)
        return A[idx]

    Xs = _subsample(X)
    Ys = _subsample(Y)

    def _mean_pairwise_dist(A: np.ndarray, B: np.ndarray) -> float:
        a2 = np.sum(A * A, axis=1, keepdims=True)
        b2 = np.sum(B * B, axis=1, keepdims=True).T
        d2 = a2 + b2 - 2.0 * (A @ B.T)
        d2 = np.maximum(d2, 0.0)
        d = np.sqrt(d2)
        return float(d.mean())

    EXY = _mean_pairwise_dist(Xs, Ys)
    EXX = _mean_pairwise_dist(Xs, Xs)
    EYY = _mean_pairwise_dist(Ys, Ys)

    ed2 = 2.0 * EXY - EXX - EYY
    return float(np.sqrt(max(ed2, 0.0)))


def mmd_squared_unbiased(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidths: list[float] | None = None,
    max_samples: int = 5000,
    seed: int = 0,
) -> float:
    """
    Unbiased estimator of MMD^2 using a mixture of RBF kernels (MK-MMD).

    MMD^2(P,Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    If bandwidths is None, uses the median heuristic on the pooled sample
    and adds two extra scales (0.5x and 2x) for robustness.
    """
    if X.size == 0 or Y.size == 0:
        return np.nan

    rng = np.random.default_rng(seed)

    def _subsample(A: np.ndarray) -> np.ndarray:
        if A.shape[0] <= max_samples:
            return A
        return A[rng.choice(A.shape[0], max_samples, replace=False)]

    X, Y = _subsample(X), _subsample(Y)
    m, n = X.shape[0], Y.shape[0]

    if bandwidths is None:
        # Median heuristic on pooled sample
        Z = np.vstack([X, Y])
        if Z.shape[0] > 2000:
            Z = Z[rng.choice(Z.shape[0], 2000, replace=False)]
        z2 = np.sum(Z * Z, axis=1, keepdims=True)
        D2 = np.maximum(z2 + z2.T - 2.0 * (Z @ Z.T), 0.0)
        np.fill_diagonal(D2, 0.0)
        median_d2 = np.median(D2[D2 > 0])
        sig = float(np.sqrt(0.5 * median_d2))
        bandwidths = [sig * 0.5, sig, sig * 2.0]

    def _rbf(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
        a2 = np.sum(A * A, axis=1, keepdims=True)
        b2 = np.sum(B * B, axis=1, keepdims=True).T
        return np.exp(-np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0) / (2.0 * sigma**2))

    mmd2_per_kernel = []
    for sigma in bandwidths:
        Kxx = _rbf(X, X, sigma)
        np.fill_diagonal(Kxx, 0.0)
        Kyy = _rbf(Y, Y, sigma)
        np.fill_diagonal(Kyy, 0.0)
        Kxy = _rbf(X, Y, sigma)
        mmd2_per_kernel.append(
            Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2.0 * Kxy.mean()
        )

    return float(np.mean(mmd2_per_kernel))


# NOTE: CHANGE SCALER FOR MULTIPLE REGIONS; WORKS FOR NOW
def compute_domain_shift(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    monthly_cols: list[str],
    static_cols: list[str],
    id_col: str = "ID",
    glacier_col: str = "GLACIER",
    seed: int = 0,
) -> dict:
    """
    Assess input-space domain shift between source and target.

    Climate variables: computed at row (monthly) level to preserve seasonal
    distribution — no aggregation.

    Static/topographic variables: computed at glacier level (one row per
    glacier) since they are fixed properties repeated across all months,
    avoiding pseudoreplication.

    Joint distance: weighted average of climate and topo MMD²/energy,
    rather than concatenation, to avoid static cols being overrepresented
    due to repetition.
    """

    # --- climate: row-level ---
    Xm_src = df_src[monthly_cols].to_numpy(dtype=np.float64)
    Xm_tgt = df_tgt[monthly_cols].to_numpy(dtype=np.float64)

    # --- topo: glacier-level (deduplicated) ---
    Xs_src = df_src.groupby(glacier_col)[static_cols].first().to_numpy(dtype=np.float64)
    Xs_tgt = df_tgt.groupby(glacier_col)[static_cols].first().to_numpy(dtype=np.float64)

    # --- scalers fitted on pooled source + target ---
    scaler_m = StandardScaler().fit(np.vstack([Xm_src, Xm_tgt]))
    scaler_s = StandardScaler().fit(np.vstack([Xs_src, Xs_tgt]))

    Xm_src_z = scaler_m.transform(Xm_src)
    Xm_tgt_z = scaler_m.transform(Xm_tgt)
    Xs_src_z = scaler_s.transform(Xs_src)
    Xs_tgt_z = scaler_s.transform(Xs_tgt)

    # --- distances ---
    D_mmd2_climate = mmd_squared_unbiased(Xm_src_z, Xm_tgt_z, seed=seed + 2)
    D_energy_climate = energy_distance(Xm_src_z, Xm_tgt_z, seed=seed + 3)
    D_mmd2_topo = mmd_squared_unbiased(Xs_src_z, Xs_tgt_z, seed=seed + 4)
    D_energy_topo = energy_distance(Xs_src_z, Xs_tgt_z, seed=seed + 5)

    out = {
        "n_src_rows": len(Xm_src),
        "n_tgt_rows": len(Xm_tgt),
        "n_src_glaciers": len(Xs_src),
        "n_tgt_glaciers": len(Xs_tgt),
        # --- joint: weighted average of climate + topo ---
        "D_mmd2_joint": 0.5 * D_mmd2_climate + 0.5 * D_mmd2_topo,
        "D_energy_joint": 0.5 * D_energy_climate + 0.5 * D_energy_topo,
        # --- climate only ---
        "D_mmd2_climate": D_mmd2_climate,
        "D_energy_climate": D_energy_climate,
        # --- topo only ---
        "D_mmd2_topo": D_mmd2_topo,
        "D_energy_topo": D_energy_topo,
    }

    # --- per-variable marginal distances ---
    # climate vars: row-level
    for j, col in enumerate(monthly_cols):
        out[f"D_mmd2_{col}"] = float(
            mmd_squared_unbiased(
                Xm_src_z[:, j : j + 1], Xm_tgt_z[:, j : j + 1], seed=seed + 10 + j
            )
        )
        out[f"D_energy_{col}"] = float(
            energy_distance(
                Xm_src_z[:, j : j + 1], Xm_tgt_z[:, j : j + 1], seed=seed + 100 + j
            )
        )

    # topo vars: glacier-level
    for j, col in enumerate(static_cols):
        out[f"D_mmd2_{col}"] = float(
            mmd_squared_unbiased(
                Xs_src_z[:, j : j + 1],
                Xs_tgt_z[:, j : j + 1],
                seed=seed + 10 + len(monthly_cols) + j,
            )
        )
        out[f"D_energy_{col}"] = float(
            energy_distance(
                Xs_src_z[:, j : j + 1],
                Xs_tgt_z[:, j : j + 1],
                seed=seed + 100 + len(monthly_cols) + j,
            )
        )

    return out


def split_pool_holdout(
    df_region: pd.DataFrame,
    monthly_cols: list[str],
    static_cols: list[str],
    glacier_col: str = "GLACIER",
    id_col: str = "ID",
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> dict:

    def _glacier_features(df):
        meas_m = df.groupby(id_col)[monthly_cols].mean()
        meas_s = df.groupby(id_col)[[glacier_col] + static_cols].first()
        meas = meas_m.join(meas_s)
        grp = meas.groupby(glacier_col)
        X = np.hstack(
            [
                grp[monthly_cols].mean().to_numpy(dtype=np.float64),
                grp[static_cols].mean().to_numpy(dtype=np.float64),
            ]
        )
        names = grp[monthly_cols].mean().index.tolist()
        n_meas = grp[monthly_cols].count().iloc[:, 0].to_numpy(dtype=int)
        return X, names, n_meas

    X, glaciers, n_meas = _glacier_features(df_region)
    n_total_meas = n_meas.sum()
    target_holdout_meas = int(np.round(holdout_frac * n_total_meas))

    scaler = StandardScaler().fit(X)
    X_z = scaler.transform(X)

    print(f"  Total measurements   : {n_total_meas}")
    print(f"  Target holdout meas  : {target_holdout_meas} ({holdout_frac:.0%})")
    print(f"  Total glaciers       : {len(glaciers)}")

    # Return 0 for empty sets (not inf) — empty side is neutral, not broken
    def _mmd2(idxs):
        if len(idxs) < 2:
            return 0.0  # can't estimate MMD² with <2 samples, treat as 0
        return float(mmd_squared_unbiased(X_z[idxs], X_z, seed=seed))

    holdout_idxs = []
    pool_idxs = []
    holdout_meas_count = 0

    rng = np.random.default_rng(seed)
    order = list(range(len(glaciers)))
    rng.shuffle(order)

    for glacier_idx in order:
        glacier_meas = n_meas[glacier_idx]
        holdout_full = holdout_meas_count + glacier_meas > target_holdout_meas * 1.5

        if holdout_full:
            pool_idxs.append(glacier_idx)
            continue

        trial_holdout = holdout_idxs + [glacier_idx]
        trial_pool = pool_idxs + [glacier_idx]

        # Both sides evaluated independently vs Region_all — no cross-contamination
        combined_if_holdout = _mmd2(trial_holdout) + _mmd2(pool_idxs)
        combined_if_pool = _mmd2(holdout_idxs) + _mmd2(trial_pool)

        if combined_if_holdout <= combined_if_pool:
            holdout_idxs.append(glacier_idx)
            holdout_meas_count += glacier_meas
        else:
            pool_idxs.append(glacier_idx)

    mmd2_holdout = _mmd2(holdout_idxs)
    mmd2_pool = _mmd2(pool_idxs)
    actual_frac = holdout_meas_count / n_total_meas

    print(
        f"\n  Holdout : {len(holdout_idxs)} glaciers, "
        f"{holdout_meas_count} measurements ({actual_frac:.1%})"
    )
    print(
        f"  Pool    : {len(pool_idxs)} glaciers, "
        f"{n_total_meas - holdout_meas_count} measurements ({1-actual_frac:.1%})"
    )
    print(f"  MMD²(holdout, Region_all) = {mmd2_holdout:.4f}")
    print(f"  MMD²(pool,    Region_all) = {mmd2_pool:.4f}")

    return {
        "holdout_glaciers": [glaciers[i] for i in holdout_idxs],
        "pool_glaciers": [glaciers[i] for i in pool_idxs],
        "n_meas_holdout": int(holdout_meas_count),
        "n_meas_pool": int(n_total_meas - holdout_meas_count),
        "actual_holdout_frac": float(actual_frac),
        "mmd2_holdout_vs_region": float(mmd2_holdout),
        "mmd2_pool_vs_region": float(mmd2_pool),
    }
