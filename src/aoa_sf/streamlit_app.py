from __future__ import annotations

import io
import csv
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from aoa_sf import compute_aosf


st.set_page_config(page_title="AoA & Surgical Freedom", layout="wide")


def parse_xyz(s: str) -> np.ndarray:
    """
    Parse 'x, y, z' or 'x y z' into np.array([x,y,z]).
    """
    s = s.strip().replace(";", ",")
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    else:
        parts = [p.strip() for p in s.split() if p.strip() != ""]
    if len(parts) != 3:
        raise ValueError("Expected 3 numbers (x, y, z).")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def make_plot(out):
    pivot = out.pivot
    cranial = out.superior      # formerly "superior"
    caudal = out.inferior       # formerly "inferior"
    medial = out.medial
    lateral = out.lateral
    C = out.sf_centroid

    fig = go.Figure()

    # --- Points + labels ---
    labels = ["Pivot", "Cranial", "Caudal", "Medial", "Lateral", "SF centroid"]
    pts = np.vstack([pivot, cranial, caudal, medial, lateral, C])

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=6),
            name="Landmarks",
        )
    )

    # --- Rays: pivot -> each point ---
    for p, name in [
        (cranial, "Pivot→Cranial"),
        (caudal, "Pivot→Caudal"),
        (medial, "Pivot→Medial"),
        (lateral, "Pivot→Lateral"),
    ]:
        seg = np.vstack([pivot, p])
        fig.add_trace(
            go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                name=name,
            )
        )

    # --- Vertical AoA triangle (Cranial–Caudal–Pivot) ---
    tri_v = np.vstack([cranial, caudal, pivot, cranial])
    fig.add_trace(
        go.Scatter3d(
            x=tri_v[:, 0], y=tri_v[:, 1], z=tri_v[:, 2],
            mode="lines",
            name="Vertical AoA triangle",
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=[cranial[0], caudal[0], pivot[0]],
            y=[cranial[1], caudal[1], pivot[1]],
            z=[cranial[2], caudal[2], pivot[2]],
            opacity=0.20,
            name="Vertical AoA surface",
        )
    )

    # --- Horizontal AoA triangle (Medial–Lateral–Pivot) ---
    tri_h = np.vstack([medial, lateral, pivot, medial])
    fig.add_trace(
        go.Scatter3d(
            x=tri_h[:, 0], y=tri_h[:, 1], z=tri_h[:, 2],
            mode="lines",
            name="Horizontal AoA triangle",
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=[medial[0], lateral[0], pivot[0]],
            y=[medial[1], lateral[1], pivot[1]],
            z=[medial[2], lateral[2], pivot[2]],
            opacity=0.20,
            name="Horizontal AoA surface",
        )
    )

    # --- SF polygon (semi-transparent) on best-fit plane ---
    # The 4 cardinal points are the ones used to fit the best-fit plane in compute_aosf().
    # We fill the quad (cranial, caudal, medial, lateral) as two triangles.
    fig.add_trace(
        go.Mesh3d(
            x=[cranial[0], caudal[0], medial[0], cranial[0], medial[0], lateral[0]],
            y=[cranial[1], caudal[1], medial[1], cranial[1], medial[1], lateral[1]],
            z=[cranial[2], caudal[2], medial[2], cranial[2], medial[2], lateral[2]],
            i=[0, 3],
            j=[1, 4],
            k=[2, 5],
            opacity=0.18,
            name="SF area (best-fit plane)",
        )
    )

    # Outline SF polygon edges for clarity (not necessarily the same ordering used in SF computation,
    # but visually helpful; SF is computed from plane-projected ordered polygon in the backend).
    sf_outline = np.vstack([cranial, caudal, medial, lateral, cranial])
    fig.add_trace(
        go.Scatter3d(
            x=sf_outline[:, 0], y=sf_outline[:, 1], z=sf_outline[:, 2],
            mode="lines",
            name="SF outline",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h"),
    )
    return fig


def build_csv_single_row(
    out,
    unit_label: str,
    coord_system: str,
    rescale_mode: str,
    rescale_value: float | None,
) -> str:
    """
    One-row CSV for manuscript/supplementary export.
    Includes labels for units and coordinate system.
    Uses cranial/caudal terminology in headers.
    """
    headers = [
        "coordinate_system",
        "units",
        "rescale_mode",
        "rescale_value",
        "Vertical_AoA_deg",          # cranial–caudal
        "Horizontal_AoA_deg",        # medial–lateral
        "SF_area_units2",
        "SF_centroid_x",
        "SF_centroid_y",
        "SF_centroid_z",
        "Centroid_to_pivot_units",
        "pivot_x", "pivot_y", "pivot_z",
        "cranial_x", "cranial_y", "cranial_z",
        "caudal_x", "caudal_y", "caudal_z",
        "medial_x", "medial_y", "medial_z",
        "lateral_x", "lateral_y", "lateral_z",
    ]

    row = [
        coord_system,
        unit_label,
        rescale_mode,
        "" if rescale_value is None else f"{rescale_value}",
        f"{out.aoa_si_deg:.6f}",
        f"{out.aoa_ml_deg:.6f}",
        f"{out.sf_area:.6f}",
        f"{out.sf_centroid[0]:.6f}",
        f"{out.sf_centroid[1]:.6f}",
        f"{out.sf_centroid[2]:.6f}",
        f"{out.centroid_to_pivot:.6f}",
        f"{out.pivot[0]:.6f}", f"{out.pivot[1]:.6f}", f"{out.pivot[2]:.6f}",
        f"{out.superior[0]:.6f}", f"{out.superior[1]:.6f}", f"{out.superior[2]:.6f}",
        f"{out.inferior[0]:.6f}", f"{out.inferior[1]:.6f}", f"{out.inferior[2]:.6f}",
        f"{out.medial[0]:.6f}", f"{out.medial[1]:.6f}", f"{out.medial[2]:.6f}",
        f"{out.lateral[0]:.6f}", f"{out.lateral[1]:.6f}", f"{out.lateral[2]:.6f}",
    ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerow(row)
    return buf.getvalue()


def build_csv_points_long(out, unit_label: str, coord_system: str) -> str:
    """
    Long-format CSV with one row per point.
    Uses cranial/caudal terminology in labels.
    """
    headers = ["coordinate_system", "units", "label", "x", "y", "z"]
    rows = [
        [coord_system, unit_label, "pivot", *out.pivot.tolist()],
        [coord_system, unit_label, "cranial", *out.superior.tolist()],
        [coord_system, unit_label, "caudal", *out.inferior.tolist()],
        [coord_system, unit_label, "medial", *out.medial.tolist()],
        [coord_system, unit_label, "lateral", *out.lateral.tolist()],
        [coord_system, unit_label, "SF_centroid", *out.sf_centroid.tolist()],
    ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


st.title("AoA (Vertical / Horizontal) + Surgical Freedom (SF)")

with st.sidebar:
    st.header("Metadata")
    unit_label = st.text_input("Units label", value="mm", help="e.g., mm, cm, vox, m")
    coord_system = st.selectbox(
        "Coordinate system",
        ["LPS", "RAS", "Scanner", "Custom"],
        index=0,
        help="Label the coordinate convention used to define X/Y/Z.",
    )

    if coord_system == "Custom":
        coord_system_custom = st.text_input("Custom coordinate system label", value="Custom")
        coord_system_effective = coord_system_custom.strip() or "Custom"
    else:
        coord_system_effective = coord_system

    st.divider()
    st.header("Input points (x, y, z)")
    st.caption("Use `x, y, z` or `x y z` format.")

    pivot_s = st.text_input("Pivot", value="125, 95, 145")
    sup_s = st.text_input("Cranial", value="126.3, 101, 151")
    inf_s = st.text_input("Caudal", value="122.9, 93.8, 150.1")
    med_s = st.text_input("Medial", value="121.2, 98.4, 147.6")
    lat_s = st.text_input("Lateral", value="130.1, 92.5, 149.0")

    st.divider()
    st.header("Rescaling")

    rescale_mode = st.selectbox("Rescale mode", ["none", "absolute", "relative"], index=0)

    radius = None
    factor = None
    if rescale_mode == "absolute":
        radius = st.number_input(
            f"Absolute radius ({unit_label})",
            min_value=0.0,
            value=20.0,
            step=1.0
        )
    elif rescale_mode == "relative":
        factor = st.number_input(
            "Relative factor (e.g., 0.8 or 1.2)",
            min_value=0.0,
            value=1.0,
            step=0.05
        )

    compute_btn = st.button("Compute", type="primary", use_container_width=True)

# Auto-compute on first load too
if "computed_once" not in st.session_state:
    st.session_state["computed_once"] = False

# Store last output for export
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "last_export_meta" not in st.session_state:
    st.session_state["last_export_meta"] = None

if compute_btn or (not st.session_state["computed_once"]):
    try:
        pivot = parse_xyz(pivot_s)
        cranial = parse_xyz(sup_s)
        caudal = parse_xyz(inf_s)
        medial = parse_xyz(med_s)
        lateral = parse_xyz(lat_s)

        if rescale_mode == "absolute":
            rescale = {"mode": "absolute", "radius": float(radius)}
            rescale_value = float(radius)
        elif rescale_mode == "relative":
            rescale = {"mode": "relative", "factor": float(factor)}
            rescale_value = float(factor)
        else:
            rescale = {"mode": "none"}
            rescale_value = None

        out = compute_aosf(
            pivot=pivot,
            superior=cranial,   # backend uses superior/inferior naming
            inferior=caudal,
            medial=medial,
            lateral=lateral,
            rescale=rescale,
        )

        st.session_state["computed_once"] = True
        st.session_state["last_out"] = out
        st.session_state["last_export_meta"] = {
            "unit_label": unit_label.strip() or "units",
            "coord_system": coord_system_effective,
            "rescale_mode": rescale_mode,
            "rescale_value": rescale_value,
        }

    except Exception as e:
        st.error(f"Input error: {e}")

out = st.session_state.get("last_out", None)
meta = st.session_state.get("last_export_meta", None)

if out is not None and meta is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Results")
        st.caption(f"Coordinate system: **{meta['coord_system']}** · Units: **{meta['unit_label']}**")

        st.metric("Vertical AoA (Cranial–Caudal) (deg)", f"{out.aoa_si_deg:.3f}")
        st.metric("Horizontal AoA (Medial–Lateral) (deg)", f"{out.aoa_ml_deg:.3f}")
        st.metric(f"SF area ({meta['unit_label']}²)", f"{out.sf_area:.3f}")
        st.metric(f"Centroid → Pivot ({meta['unit_label']})", f"{out.centroid_to_pivot:.3f}")

        st.write("**SF centroid (x, y, z):**")
        st.code(f"{out.sf_centroid[0]:.6f}, {out.sf_centroid[1]:.6f}, {out.sf_centroid[2]:.6f}")

        with st.expander("Triangle reports (sides + angles)"):
            st.write("**Vertical triangle (Cranial–Caudal)** (AoA at pivot)")
            st.json(out.si_report.sides)
            st.json(out.si_report.angles_deg)

            st.write("**Horizontal triangle (Medial–Lateral)** (AoA at pivot)")
            st.json(out.ml_report.sides)
            st.json(out.ml_report.angles_deg)

        st.divider()
        st.subheader("Export")

        csv_single = build_csv_single_row(
            out=out,
            unit_label=meta["unit_label"],
            coord_system=meta["coord_system"],
            rescale_mode=meta["rescale_mode"],
            rescale_value=meta["rescale_value"],
        )

        csv_long = build_csv_points_long(
            out=out,
            unit_label=meta["unit_label"],
            coord_system=meta["coord_system"],
        )

        st.download_button(
            label="Download CSV (single row: metrics + points)",
            data=csv_single,
            file_name="aoa_sf_metrics_single_row.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            label="Download CSV (long format: one row per point)",
            data=csv_long,
            file_name="aoa_sf_points_long.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.subheader("Interactive 3D plot")
        st.caption("Rotate/zoom to inspect the geometry.")
        fig = make_plot(out)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter points and click **Compute** to see results and enable export.")
