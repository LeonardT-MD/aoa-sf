from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .geometry import (
    as_xyz,
    norm,
    angle_between,
    rescale_point_along_ray,
    best_fit_plane_basis,
    project_point_to_plane,
    project_to_plane_2d,
    order_points_clockwise_2d,
    polygon_area_centroid_2d,
    lift_from_plane_2d,
)


@dataclass(frozen=True)
class TriangleReport:
    """
    Minimal report to keep parity with your old output.
    'AoA' here is the angle at pivot between the two rays.
    """
    angles_deg: Dict[str, float]


@dataclass(frozen=True)
class AoSFOutput:
    pivot: np.ndarray

    # AoA at pivot (ray-based)
    aoa_vertical_deg: float       # cranial–caudal
    aoa_horizontal_deg: float     # medial–lateral
    vertical_report: TriangleReport
    horizontal_report: TriangleReport

    # Surgical Freedom (plane-based)
    sf_area: float
    sf_centroid: np.ndarray
    centroid_to_pivot: float

    # Effective (ray-preserved) points used for AoA (and for rays in plot)
    cranial: np.ndarray
    caudal: np.ndarray
    medial: np.ndarray
    lateral: np.ndarray

    # Plane-projected polygon for SF plotting (ordered, non-self-crossing)
    sf_polygon_projected_ordered_3d: np.ndarray  # (4,3)


def compute_aosf(
    pivot,
    superior,
    inferior,
    medial,
    lateral,
    rescale: Optional[Dict[str, Any]] = None,
) -> AoSFOutput:
    """
    Spec-accurate behavior:

    1) Rescaling is performed along the original pivot->point ray (alignment preserved),
       using a NEW DISTANCE from pivot (absolute) or a multiplicative factor (relative).
       => AoA angles MUST NOT change under rescaling.

    2) AoA (vertical/horizontal) computed purely from ray vectors in 3D:
         Vertical  = angle between (cranial-pivot) and (caudal-pivot)
         Horizontal= angle between (medial-pivot)  and (lateral-pivot)

    3) SF computed on best-fit plane of the 4 points with minimal displacement:
       - fit plane to the 4 effective points
       - orthogonally project points to plane
       - compute polygon area in plane 2D coords (shoelace) after CW ordering
       - centroid returned in 3D on the plane

    rescale dict:
      {"mode":"none"} (default)
      {"mode":"absolute","distance": float}  # new distance from pivot for ALL 4 points
      {"mode":"relative","factor": float}    # multiply each original distance by factor
    """
    pivot = as_xyz(pivot)
    cranial = as_xyz(superior)
    caudal = as_xyz(inferior)
    medial_p = as_xyz(medial)
    lateral_p = as_xyz(lateral)

    rescale = rescale or {"mode": "none"}
    mode = str(rescale.get("mode", "none")).lower()

    # --- 1) Ray-preserving rescale (angles invariant) ---
    if mode == "absolute":
        new_d = float(rescale["distance"])
        cranial = rescale_point_along_ray(cranial, pivot, new_d)
        caudal = rescale_point_along_ray(caudal, pivot, new_d)
        medial_p = rescale_point_along_ray(medial_p, pivot, new_d)
        lateral_p = rescale_point_along_ray(lateral_p, pivot, new_d)

    elif mode == "relative":
        f = float(rescale["factor"])
        if f <= 0:
            raise ValueError("factor must be > 0")
        cranial = rescale_point_along_ray(cranial, pivot, norm(cranial - pivot) * f)
        caudal = rescale_point_along_ray(caudal, pivot, norm(caudal - pivot) * f)
        medial_p = rescale_point_along_ray(medial_p, pivot, norm(medial_p - pivot) * f)
        lateral_p = rescale_point_along_ray(lateral_p, pivot, norm(lateral_p - pivot) * f)

    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown rescale mode: {mode}")

    # --- 2) AoA angles in 3D (ray-based) ---
    v_cr = cranial - pivot
    v_ca = caudal - pivot
    v_me = medial_p - pivot
    v_la = lateral_p - pivot

    aoa_vertical = angle_between(v_cr, v_ca)
    aoa_horizontal = angle_between(v_me, v_la)

    vertical_report = TriangleReport(angles_deg={"AoA": float(aoa_vertical)})
    horizontal_report = TriangleReport(angles_deg={"AoA": float(aoa_horizontal)})

    # --- 3) SF on best-fit plane with minimal displacement ---
    P = np.vstack([cranial, caudal, medial_p, lateral_p])
    origin, e1, e2, n = best_fit_plane_basis(P)

    # orthogonal projection = minimal displacement
    Pproj = np.vstack([project_point_to_plane(p, origin, n) for p in P])

    # plane coordinates
    P2 = project_to_plane_2d(Pproj, origin, e1, e2)
    order = order_points_clockwise_2d(P2)
    P2o = P2[order]

    # ordered projected polygon in 3D (for plotting SF)
    P3o = np.vstack([lift_from_plane_2d(pt, origin, e1, e2) for pt in P2o])

    sf_area, centroid2 = polygon_area_centroid_2d(P2o)
    centroid3 = lift_from_plane_2d(centroid2, origin, e1, e2)
    c2p = float(np.linalg.norm(centroid3 - pivot))

    return AoSFOutput(
        pivot=pivot,
        aoa_vertical_deg=float(aoa_vertical),
        aoa_horizontal_deg=float(aoa_horizontal),
        vertical_report=vertical_report,
        horizontal_report=horizontal_report,
        sf_area=float(sf_area),
        sf_centroid=centroid3,
        centroid_to_pivot=c2p,
        cranial=cranial,
        caudal=caudal,
        medial=medial_p,
        lateral=lateral_p,
        sf_polygon_projected_ordered_3d=P3o,
    )
