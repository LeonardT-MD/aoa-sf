from __future__ import annotations

import numpy as np
from typing import Tuple

EPS = 1e-12


def as_xyz(p) -> np.ndarray:
    return np.asarray(p, dtype=float).reshape(3,)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("Zero-length vector; cannot normalize.")
    return v / n


def rescale_point_along_ray(point: np.ndarray, pivot: np.ndarray, new_distance: float) -> np.ndarray:
    """
    Ray-preserving rescale: move `point` along the original pivot->point direction
    so that ||point' - pivot|| == new_distance.

    This preserves all AoA angles, by construction.
    """
    point = as_xyz(point)
    pivot = as_xyz(pivot)
    d = float(new_distance)
    if d <= 0:
        raise ValueError("new_distance must be > 0")

    v = point - pivot
    return pivot + unit(v) * d


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """
    Angle (degrees) between two vectors using dot-product.
    """
    u = np.asarray(u, dtype=float).reshape(3,)
    v = np.asarray(v, dtype=float).reshape(3,)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < EPS or nv < EPS:
        raise ValueError("Cannot compute angle with near-zero vector.")
    c = float(np.dot(u, v) / (nu * nv))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def best_fit_plane_basis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Least-squares plane via SVD/PCA on Nx3 points.
    Returns (origin, e1, e2, n):
      - origin: centroid of points
      - e1,e2: orthonormal in-plane basis
      - n: unit normal (orthogonal to plane)
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        raise ValueError("points must be (N,3) with N>=3")

    origin = P.mean(axis=0)
    X = P - origin
    _, _, Vt = np.linalg.svd(X, full_matrices=False)

    # Vt[0], Vt[1] are principal directions in-plane, Vt[-1] is normal
    e1 = Vt[0]
    n = Vt[-1]
    e2 = np.cross(n, e1)

    e1 = e1 / (np.linalg.norm(e1) + EPS)
    e2 = e2 / (np.linalg.norm(e2) + EPS)
    n = n / (np.linalg.norm(n) + EPS)

    return origin, e1, e2, n


def project_point_to_plane(point: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection of a point onto a plane (minimal displacement).
    """
    point = as_xyz(point)
    o = as_xyz(plane_origin)
    n = as_xyz(plane_normal)
    n = n / (np.linalg.norm(n) + EPS)
    v = point - o
    signed = float(np.dot(v, n))
    return point - signed * n


def project_to_plane_2d(points: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    Project Nx3 points to Nx2 coordinates in the (e1,e2) basis.
    """
    P = np.asarray(points, dtype=float)
    X = P - origin
    x = X @ e1
    y = X @ e2
    return np.c_[x, y]


def lift_from_plane_2d(pt2: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    Lift 2D plane coordinates back to 3D.
    """
    pt2 = np.asarray(pt2, dtype=float).reshape(2,)
    return origin + e1 * pt2[0] + e2 * pt2[1]


def order_points_clockwise_2d(pts2: np.ndarray) -> np.ndarray:
    """
    Returns indices that order 2D points clockwise around centroid.
    """
    P = np.asarray(pts2, dtype=float)
    c = P.mean(axis=0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    return np.argsort(ang)


def polygon_area_centroid_2d(pts2_ordered: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Shoelace area and polygon centroid (area-weighted).
    pts2_ordered must be ordered (CW or CCW).
    Returns (area>=0, centroid_xy).
    """
    P = np.asarray(pts2_ordered, dtype=float)
    x = P[:, 0]
    y = P[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y

    A = 0.5 * np.sum(cross)
    if abs(A) < EPS:
        return 0.0, P.mean(axis=0)

    Cx = (1.0 / (6.0 * A)) * np.sum((x + x2) * cross)
    Cy = (1.0 / (6.0 * A)) * np.sum((y + y2) * cross)
    return float(abs(A)), np.array([Cx, Cy], dtype=float)
