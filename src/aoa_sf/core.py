from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .geometry import (
    _as_xyz,
    distance,
    rescale_point_from_pivot,
    triangle_sides,
    triangle_angles_law_of_cosines,
    pca_plane_basis,
    project_to_2d,
    order_points_clockwise_2d,
    polygon_area_centroid_2d,
    lift_from_2d,
)


@dataclass(frozen=True)
class TriangleReport:
    sides: Dict[str, float]
    angles_deg: Dict[str, float]   # includes "AoA" at pivot


@dataclass(frozen=True)
class AoSFOutput:
    pivot: np.ndarray

    # AoA at pivot
    aoa_si_deg: float
    aoa_ml_deg: float
    si_report: TriangleReport
    ml_report: TriangleReport

    # Surgical Freedom
    sf_area: float
    sf_centroid: np.ndarray
    centroid_to_pivot: float

    # Effective points used
    superior: np.ndarray
    inferior: np.ndarray
    medial: np.ndarray
    lateral: np.ndarray


def _aoa_at_pivot(p_a: np.ndarray, p_b: np.ndarray, pivot: np.ndarray):
    """
    AoA at pivot between rays pivot->p_a and pivot->p_b.
    Implemented via triangle sides + law of cosines.
    """
    side_a, side_b, side_c = triangle_sides(p_a, p_b, pivot)
    alpha, beta, gamma = triangle_angles_law_of_cosines(side_a, side_b, side_c)

    rep = TriangleReport(
        sides={"a": side_a, "b": side_b, "c": side_c},
        angles_deg={"alpha": alpha, "beta": beta, "AoA": gamma},
    )
    return float(gamma), rep


def compute_aosf(
    pivot,
    superior,
    inferior,
    medial,
    lateral,
    rescale: Optional[Dict[str, Any]] = None,
) -> AoSFOutput:
    """
    Compute:
      - AoA_SI at pivot (superior–inferior pair)
      - AoA_ML at pivot (medial–lateral pair)
      - Surgical freedom area (SF) and centroid from the 4-point polygon

    Rescale options:
      {"mode":"none"} (default)
      {"mode":"absolute","radius": float}  -> all points set to fixed radius from pivot
      {"mode":"relative","factor": float}  -> each point radius multiplied by factor
    """
    pivot = _as_xyz(pivot)
    s = _as_xyz(superior)
    i = _as_xyz(inferior)
    m = _as_xyz(medial)
    l = _as_xyz(lateral)

    rescale = rescale or {"mode": "none"}
    mode = str(rescale.get("mode", "none")).lower()

    if mode == "absolute":
        r = float(rescale["radius"])
        s = rescale_point_from_pivot(s, pivot, r)
        i = rescale_point_from_pivot(i, pivot, r)
        m = rescale_point_from_pivot(m, pivot, r)
        l = rescale_point_from_pivot(l, pivot, r)
    elif mode == "relative":
        f = float(rescale["factor"])
        s = rescale_point_from_pivot(s, pivot, distance(s, pivot) * f)
        i = rescale_point_from_pivot(i, pivot, distance(i, pivot) * f)
        m = rescale_point_from_pivot(m, pivot, distance(m, pivot) * f)
        l = rescale_point_from_pivot(l, pivot, distance(l, pivot) * f)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown rescale mode: {mode}")

    # AoA at pivot
    aoa_si, si_rep = _aoa_at_pivot(s, i, pivot)
    aoa_ml, ml_rep = _aoa_at_pivot(m, l, pivot)

    # SF polygon area in best-fit plane
    poly3 = np.vstack([s, i, m, l])
    origin, e1, e2, _n = pca_plane_basis(poly3)
    poly2 = project_to_2d(poly3, origin, e1, e2)

    order = order_points_clockwise_2d(poly2)
    poly2o = poly2[order]

    sf_area, centroid2 = polygon_area_centroid_2d(poly2o)
    centroid3 = lift_from_2d(centroid2, origin, e1, e2)

    c2p = float(np.linalg.norm(centroid3 - pivot))

    return AoSFOutput(
        pivot=pivot,
        aoa_si_deg=aoa_si,
        aoa_ml_deg=aoa_ml,
        si_report=si_rep,
        ml_report=ml_rep,
        sf_area=float(sf_area),
        sf_centroid=centroid3,
        centroid_to_pivot=c2p,
        superior=s, inferior=i, medial=m, lateral=l,
    )
