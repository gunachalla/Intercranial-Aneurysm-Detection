# ============================================
# vessel_orientation_solver.py
# Estimate (perm, signs) from 3-class vessel segmentation (24 rotations; LR symmetry)
# ============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import torch
import itertools


# ---------- 24 rotation table (det=+1) ----------
# Generated to match orientation_2d_dataset spec (row=new axis, col=old axis;
# perm maps new_row <- old_axis). Equivalent to ORIENTATION_TABLE_ROT.
def _build_rot_table_24() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    entries: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            mat = np.zeros((3, 3), dtype=np.int8)
            for row, axis in enumerate(perm):
                mat[row, axis] = signs[row]
            if round(np.linalg.det(mat)) == 1:
                entries.append((tuple(int(x) for x in perm), tuple(int(x) for x in signs)))
    if len(entries) != 24:
        raise RuntimeError(f"rotation table build failed: got {len(entries)} (expected 24)")
    return entries


ROT_TABLE_24: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = _build_rot_table_24()


# ---------- Utilities ----------
def _to_numpy_int(seg: Any) -> np.ndarray:
    if isinstance(seg, torch.Tensor):
        arr = seg.detach().cpu().numpy()
    else:
        arr = np.asarray(seg)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"seg must be (Z,Y,X), got {arr.shape}")
    return arr.astype(np.int16, copy=False)


def _ensure_spacing(spacing_zyx: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if spacing_zyx is None:  # fallback to 1 mm isotropic
        return (1.0, 1.0, 1.0)
    sz, sy, sx = spacing_zyx
    return (float(max(sz, 1e-6)), float(max(sy, 1e-6)), float(max(sx, 1e-6)))


def _transform_vec(v: np.ndarray, perm: Tuple[int, int, int], signs: Tuple[int, int, int]) -> np.ndarray:
    # Map v=(z,y,x) to candidate: (s0*v[p0], s1*v[p1], s2*v[p2])
    return np.array([signs[0] * v[perm[0]], signs[1] * v[perm[1]], signs[2] * v[perm[2]]], dtype=np.float64)


def _transform_var(var: np.ndarray, perm: Tuple[int, int, int]) -> np.ndarray:
    # Variance is sign-invariant; only permute axes
    return np.array([var[perm[0]], var[perm[1]], var[perm[2]]], dtype=np.float64)


def _two_means_3d(
    pts: np.ndarray,
    axis_vals: np.ndarray,
    iters: int = 8,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """3D k=2-means. Returns (c_low, c_high, separation)."""
    n = pts.shape[0]
    if n < 4:
        return None

    q25, q75 = np.percentile(axis_vals, [25.0, 75.0])
    thr = float(0.5 * (q25 + q75))
    split = axis_vals <= thr
    if split.sum() == 0 or split.sum() == n:
        thr = float(np.median(axis_vals))
        split = axis_vals <= thr
    if split.sum() == 0 or split.sum() == n:
        amin = float(axis_vals.min())
        amax = float(axis_vals.max())
        if amin == amax:
            return None
        thr = 0.5 * (amin + amax)
        split = axis_vals <= thr
        if split.sum() == 0 or split.sum() == n:
            order = np.argsort(axis_vals)
            half = max(1, n // 2)
            split = np.zeros(n, dtype=bool)
            split[order[:half]] = True
            if split.sum() == 0 or split.sum() == n:
                return None

    c1 = pts[split].mean(axis=0)
    c2 = pts[~split].mean(axis=0)

    for _ in range(iters):
        diff1 = pts - c1
        diff2 = pts - c2
        dist1 = np.sum(diff1 * diff1, axis=1)
        dist2 = np.sum(diff2 * diff2, axis=1)
        assign = dist1 <= dist2
        cnt1 = int(assign.sum())
        cnt2 = n - cnt1
        if cnt1 == 0 or cnt2 == 0:
            break
        new_c1 = pts[assign].mean(axis=0)
        new_c2 = pts[~assign].mean(axis=0)
        shift = np.max(np.abs(new_c1 - c1)) + np.max(np.abs(new_c2 - c2))
        c1, c2 = new_c1, new_c2
        if shift < 1e-6:
            break

    if c1[2] > c2[2]:
        c1, c2 = c2, c1
    sep = float(c2[2] - c1[2])
    if sep < 1e-9:
        sep = float(np.linalg.norm(c2 - c1))
    return c1, c2, sep


@dataclass
class VesselStats:
    # Counts
    nP: int
    nM: int
    nO: int
    nA: int
    nU: int
    # Centroids (physical coordinates)
    cP: np.ndarray
    cM: np.ndarray
    cO: np.ndarray
    cA: np.ndarray
    # Variance (physical coordinates)
    varU: np.ndarray
    varP: np.ndarray
    varA: np.ndarray
    varO: np.ndarray
    varM: np.ndarray
    # M voxel coordinates (physical; only when needed)
    coordsM: Optional[np.ndarray]
    # First principal component of U (all vessels)
    e_SI: Optional[np.ndarray]


def _precompute_stats(seg_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> VesselStats:
    sz, sy, sx = spacing_zyx
    Z, Y, X = seg_zyx.shape
    # Masks
    P = seg_zyx == 1  # Posterior_Circulation_and_Basilar
    M = seg_zyx == 2  # Middle_Cerebral_Arteries
    O = seg_zyx == 3  # Other_Locations
    A = np.logical_or(M, O)
    U = np.logical_or(P, A)

    def _centroid_and_var(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        idx = np.nonzero(mask)
        n = int(idx[0].size)
        if n == 0:
            return (
                np.array([np.nan, np.nan, np.nan], dtype=np.float64),
                np.array([np.nan, np.nan, np.nan], dtype=np.float64),
                0,
            )
        z = idx[0].astype(np.float64) * sz
        y = idx[1].astype(np.float64) * sy
        x = idx[2].astype(np.float64) * sx
        cz, cy, cx = float(z.mean()), float(y.mean()), float(x.mean())
        vz = float(z.var())  # population variance (stable)
        vy = float(y.var())
        vx = float(x.var())
        return np.array([cz, cy, cx], dtype=np.float64), np.array([vz, vy, vx], dtype=np.float64), n

    cP, varP, nP = _centroid_and_var(P)
    cM, varM, nM = _centroid_and_var(M)
    cO, varO, nO = _centroid_and_var(O)
    cA, varA, nA = _centroid_and_var(A)
    _, varU, nU = _centroid_and_var(U)

    coordsM = None
    if nM >= 50:  # light threshold (store coords only when needed)
        iz, iy, ix = np.nonzero(M)
        coordsM = np.stack([iz * sz, iy * sy, ix * sx], axis=1).astype(np.float32)

    # PCA of U (use 1st component to support SI)
    e_SI = None
    if nU >= 200:
        iz, iy, ix = np.nonzero(U)
        pts = np.stack([iz * sz, iy * sy, ix * sx], axis=1).astype(np.float64)
        pts -= pts.mean(axis=0, keepdims=True)
        cov = (pts.T @ pts) / max(1, pts.shape[0])
        w, v = np.linalg.eigh(cov)  # columns are eigenvectors (ascending)
        e_SI = v[:, 2] / (np.linalg.norm(v[:, 2]) + 1e-12)

    return VesselStats(
        nP=nP,
        nM=nM,
        nO=nO,
        nA=nA,
        nU=nU,
        cP=cP,
        cM=cM,
        cO=cO,
        cA=cA,
        varU=varU,
        varP=varP,
        varA=varA,
        varM=varM,
        varO=varO,
        coordsM=coordsM,
        e_SI=e_SI,
    )


@dataclass
class OrientationEstimate:
    perm: Tuple[int, int, int]
    signs: Tuple[int, int, int]
    score: float
    score_terms: Dict[str, float]
    reliability: Dict[str, float]
    chosen_index_24: int  # index on ROT_TABLE_24


def estimate_perm_signs_from_vessel3(
    seg: Any,
    spacing_zyx: Optional[Tuple[float, float, float]] = None,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    eps: float = 1e-6,
) -> OrientationEstimate:
    """
    Determine (perm, signs) from a 3-class vessel segmentation (24 rotations; LR symmetric).

    Args:
        seg: (Z,Y,X) int array/tensor. {0,1,2,3} = {bg, P, M, O}
        spacing_zyx: (sz, sy, sx) in mm
        weights: Composite weights (w_AP, w_LR, w_SI)
    Returns:
        OrientationEstimate
    """
    seg_np = _to_numpy_int(seg)
    spacing = _ensure_spacing(spacing_zyx)
    st = _precompute_stats(seg_np, spacing)

    wAP, wLR_base, wSI = [float(x) for x in weights]
    # Down-weight terms based on availability
    has_AP = st.nP >= 50 and st.nA >= 50 and np.isfinite(st.cP[1]) and np.isfinite(st.cA[1])
    has_LR = st.nM >= 100 and st.coordsM is not None and np.isfinite(st.varM[2])
    has_SI = st.nU >= 200 and np.isfinite(st.varU[0]) and np.isfinite(st.varU[1]) and np.isfinite(st.varU[2])

    if not has_AP:
        wAP = 0.05
    wLR = wLR_base if has_LR else 0.05
    if not has_SI:
        wSI = 0.05

    best_idx, best_score = 0, -1e19
    best_perm, best_signs = (0, 1, 2), (1, 1, 1)
    best_terms = {}
    best_rel = {}

    # Precompute: variances for AP (depend only on perm)
    varP = st.varP
    varA = st.varA
    varU = st.varU
    varM = st.varM
    varO = st.varO
    cP = st.cP
    cM = st.cM
    cA = st.cA
    cO = st.cO

    # M coordinates (only when available)
    coordsM = st.coordsM  # (N,3) physical coordinates

    for idx, (perm, signs) in enumerate(ROT_TABLE_24):
        # --- AP (anterior-posterior) ---
        if has_AP:
            cPp = _transform_vec(cP, perm, signs)
            cAp = _transform_vec(cA, perm, signs)
            varPp = _transform_var(varP, perm)
            varAp = _transform_var(varA, perm)
            Sap = (cAp[1] - cPp[1]) / np.sqrt(abs(varAp[1]) + abs(varPp[1]) + eps)
        else:
            Sap = 0.0

        # --- LR (left-right; MCA bimodality) ---
        Slr = 0.0
        lr_rel = 0.0
        if has_LR:
            # Split LR clusters with 3D 2-means
            coordsMp = coordsM[:, perm].astype(np.float64)
            coordsMp *= np.asarray(signs, dtype=np.float64)
            xin = coordsMp[:, 2]
            km = _two_means_3d(coordsMp, xin, iters=8)
            if km is not None:
                c1, c2, d = km
                varMp = _transform_var(varM, perm)
                Slr = d / (np.sqrt(abs(varMp[2])) + eps)
                # Reliability (separation/IQR)
                q25, q75 = np.percentile(xin, [25.0, 75.0])
                iqr = max(q75 - q25, eps)
                lr_rel = float(d / iqr)

        # --- SI (superior-inferior elongation) ---
        if has_SI:
            cOp = _transform_vec(cO, perm, signs)
            cMp = _transform_vec(cM, perm, signs)
            varOp = _transform_var(varO, perm)
            varMp = _transform_var(varM, perm)
            Ssi = (cMp[0] - cOp[0]) / np.sqrt(abs(varMp[0]) + abs(varOp[0]) + eps)
        else:
            Ssi = 0.0

        score = wAP * Sap + wLR * Slr + wSI * Ssi
        # print("---")
        # print(f"{perm}, {signs}")
        # print(f"Sap: {Sap}, Slr: {Slr}, Ssi: {Ssi}")
        # print(f"score: {score}")

        if (score > best_score) or (abs(score - best_score) < 1e-9 and idx < best_idx):
            best_idx, best_score = idx, score
            best_perm, best_signs = perm, signs
            best_terms = {"AP": float(Sap), "LR": float(Slr), "SI": float(Ssi)}
            best_rel = {
                "has_AP": float(has_AP),
                "has_LR": float(has_LR),
                "has_SI": float(has_SI),
                "LR_separation_IQR": float(lr_rel),
            }

    return OrientationEstimate(
        perm=best_perm,
        signs=best_signs,
        score=float(best_score),
        score_terms=best_terms,
        reliability=best_rel,
        chosen_index_24=int(best_idx),
    )
