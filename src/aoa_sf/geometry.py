from __future__ import annotations
import numpy as np
from typing import Tuple

EPS = 1e-12


def _as_xyz(p) -> np.ndarray:
    return np.asarray(p, dtype=float).reshape(3,)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def rescale_point_from_pivot(point: np.ndarray, pivot: np.ndarray, radius: float) -> np.ndarray:
    """
    Move `point` along the ray pivot->point so that ||point - pivot|| == radius.
    """
    v = point - pivot
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("Point coincides with pivot; cannot rescale.")
    return pivot + (v / n) * float(radius)


def triangle_sides(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, float, float]:
    """
    Sides (a,b,c) for triangle defined by points (a,b,c) as:
      side_a = |b-c|
      side_b = |a-c|
      side_c = |a-b|
    """
    side_a = distance(b, c)
    side_b = distance(a, c)
    side_c = distance(a, b)
    return side_a, side_b, side_c


def triangle_angles_law_of_cosines(side_a: float, side_b: float, side_c: float) -> Tuple[float, float, float]:
    """
    Returns (alpha, beta, gamma) in degrees given sides (a,b,c).
    Uses clipping for numerical stability.
    """
    def _acos_clip(x: float) -> float:
        return float(np.arccos(np.clip(x, -1.0, 1.0)))

    alpha = _acos_clip((side_b**2 + side_c**2 - side_a**2) / (2 * side_b * side_c))
    beta  = _acos_clip((side_a**2 + side_c**2 - side_b**2) / (2 * side_a * side_c))
    gamma = _acos_clip((side_a**2 + side_b**2 - side_c**2) / (2 * side_a * side_b))
    return float(np.degrees(alpha)), float(np.degrees(beta)), float(np.degrees(gamma))


def pca_plane_basis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Best-fit plane basis for Nx3 points using SVD.
    Returns (origin, e1, e2, n) where e1,e2 are in-plane orthonormal axes and n is unit normal.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        raise ValueError("points must be (N,3) with N>=3")

    origin = P.mean(axis=0)
    X = P - origin
    _, _, Vt = np.linalg.svd(X, full_matrices=False)

    # Vt[0], Vt[1] = principal directions in the plane; Vt[-1] = normal
    e1 = Vt[0]
    n = Vt[-1]
    e2 = np.cross(n, e1)

    e1 = e1 / (np.linalg.norm(e1) + EPS)
    e2 = e2 / (np.linalg.norm(e2) + EPS)
    n = n / (np.linalg.norm(n) + EPS)

    return origin, e1, e2, n


def project_to_2d(points: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    Map Nx3 -> Nx2 coordinates in the (e1,e2) plane basis.
    """
    X = points - origin
    x = X @ e1
    y = X @ e2
    return np.c_[x, y]


def lift_from_2d(pt2: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    Map 2D plane coords back to 3D.
    """
    pt2 = np.asarray(pt2, dtype=float).reshape(2,)
    return origin + e1 * pt2[0] + e2 * pt2[1]


def project_point_to_plane(point: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection of a 3D point onto a plane defined by (plane_origin, plane_normal).
    """
    point = _as_xyz(point)
    plane_origin = _as_xyz(plane_origin)
    plane_normal = _as_xyz(plane_normal)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + EPS)
    v = point - plane_origin
    dist_signed = float(v @ plane_normal)
    return point - dist_signed * plane_normal


def order_points_clockwise_2d(pts2: np.ndarray) -> np.ndarray:
    """
    Returns indices sorting points clockwise around their mean.
    """
    pts2 = np.asarray(pts2, dtype=float)
    c = pts2.mean(axis=0)
    ang = np.arctan2(pts2[:, 1] - c[1], pts2[:, 0] - c[0])
    return np.argsort(ang)


def polygon_area_centroid_2d(pts2_ordered: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Shoelace area + polygon centroid (area-weighted) for a simple polygon.
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


def rescale_points_on_fitted_plane_equal_radius(
    points4: np.ndarray,
    pivot: np.ndarray,
    radius: float
) -> np.ndarray:
    """
    Absolute rescale for SF *on the fitted plane* with equal distance constraint:

    - Fit plane to the 4 points (S/I/M/L).
    - Compute pivot projection p0 onto that plane.
    - The set of points on the plane at distance `radius` from pivot is a circle
      (sphere-plane intersection) centered at p0 with in-plane radius:
          r_plane = sqrt(radius^2 - d^2)
      where d = distance(pivot, p0).

    - For each original vertex, take its projection onto the plane to define an
      in-plane direction from p0; then place the rescaled point on the circle
      along that direction.

    Returns 4 rescaled points (N=4,3) on the fitted plane, each with ||p - pivot|| == radius.
    """
    P = np.asarray(points4, dtype=float)
    if P.shape != (4, 3):
        raise ValueError("points4 must be shape (4,3)")

    pivot = _as_xyz(pivot)
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be > 0")

    origin, e1, e2, n = pca_plane_basis(P)
    p0 = project_point_to_plane(pivot, origin, n)
    d = float(np.linalg.norm(pivot - p0))
    if radius <= d + 1e-9:
        raise ValueError(
            f"Cannot place points on plane at distance {radius:.3f} from pivot: "
            f"pivot-to-plane distance is {d:.3f}. Choose radius > {d:.3f}."
        )

    r_plane = float(np.sqrt(max(radius**2 - d**2, 0.0)))

    # Directions: from p0 to each point projected on plane
    P_proj = np.vstack([project_point_to_plane(p, origin, n) for p in P])

    out = []
    for pp in P_proj:
        v = pp - p0
        nv = float(np.linalg.norm(v))
        if nv < EPS:
            # Degenerate: pick a stable in-plane direction (e1)
            v = e1
            nv = float(np.linalg.norm(v))
        v = v / (nv + EPS)
        out.append(p0 + v * r_plane)

    return np.vstack(out)
