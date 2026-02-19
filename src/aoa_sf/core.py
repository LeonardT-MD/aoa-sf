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

    # Human-readable textual output
    text_report: str


def format_report(
    *,
    coord_system: str,
    units: str,
    rescale_mode: str,
    rescale_distance: Optional[float],
    out: "AoSFOutput",
) -> str:
    """
    Produce a publication-friendly textual report that can be printed, pasted
    into notes, or saved in logs/supplementary material.
    """
    lines = []
    lines.append("AoA–SF results")
    lines.append(f"Coordinate system: {coord_system}")
    lines.append(f"Units: {units}")
    lines.append(f"Rescale mode: {rescale_mode}")
    if rescale_mode == "absolute":
        lines.append(f"Rescale distance from pivot: {rescale_distance:.6f} {units}")
    lines.append("")
    lines.append("Angles of Attack (deg)")
    lines.append(f"  Vertical (Cranial–Caudal):   {out.aoa_vertical_deg:.6f}")
    lines.append(f"  Horizontal (Medial–Lateral): {out.aoa_horizontal_deg:.6f}")
    lines.append("")
    lines.append(f"Surgical Freedom area: {out.sf_area:.6f} {units}^2")
    lines.append(f"SF centroid (x, y, z): {out.sf_centroid[0]:.6f}, {out.sf_centroid[1]:.6f}, {out.sf_centroid[2]:.6f}")
    lines.append(f"Centroid → Pivot: {out.centroid_to_pivot:.6f} {units}")
    lines.append("")
    lines.append("Effective points used (x, y, z)")
    lines.append(f"  Pivot:   {out.pivot[0]:.6f}, {out.pivot[1]:.6f}, {out.pivot[2]:.6f}")
    lines.append(f"  Cranial: {out.cranial[0]:.6f}, {out.cranial[1]:.6f}, {out.cranial[2]:.6f}")
    lines.append(f"  Caudal:  {out.caudal[0]:.6f}, {out.caudal[1]:.6f}, {out.caudal[2]:.6f}")
    lines.append(f"  Medial:  {out.medial[0]:.6f}, {out.medial[1]:.6f}, {out.medial[2]:.6f}")
    lines.append(f"  Lateral: {out.lateral[0]:.6f}, {out.lateral[1]:.6f}, {out.lateral[2]:.6f}")
    return "\n".join(lines)


def compute_aosf(
    pivot,
    superior,
    inferior,
    medial,
    lateral,
    rescale: Optional[Dict[str, Any]] = None,
    *,
    coord_system: str = "LPS",
    units: str = "mm",
) -> AoSFOutput:
    """
    Spec-accurate behavior:

    1) Optional rescaling is ray-preserving:
       each point is moved along its original pivot->point direction to a new
       distance from pivot. This preserves AoA angles (invariance).

    2) AoA computed purely from ray vectors in 3D:
         Vertical  = angle between (cranial-pivot) and (caudal-pivot)
         Horizontal= angle between (medial-pivot)  and (lateral-pivot)

    3) SF computed on best-fit plane of the 4 points with minimal displacement:
       - fit plane to the 4 effective points
       - orthogonally project points to plane
       - compute polygon area in plane 2D coords (shoelace) after CW ordering
       - centroid returned in 3D on the plane
