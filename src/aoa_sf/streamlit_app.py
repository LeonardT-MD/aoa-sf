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
    cranial = out.superior
    caudal = out.inferior
    medial = out.medial
    lateral = out.lateral
    centroid = out.sf_centroid

    # Ordered SF polygon vertices on the fitted plane (non-self-crossing)
    sf_poly = out.sf_polygon_ordered_3d  # (4,3)
    sf_outline = np.vstack([sf_poly, sf_poly[0]])

    fig = go.Figure()

    # --- Points + labels ---
    labels = ["Pivot", "Cranial", "Caudal", "Medial", "Lateral", "SF centroid"]
    pts = np.vstack([pivot, cranial, caudal, medial, lateral, centroid])

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=6),
            name="Points",
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

    # --- SF area only (semi-transparent) ---
    # Triangulate quad as two triangles: (0,1,2) and (0,2,3)
    fig.add_trace(
        go.Mesh3d(
            x=sf_poly[:, 0],
            y=sf_poly[:, 1],
            z=sf_poly[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            opacity=0.20,
            name="SF area (fitted plane)",
        )
    )

    # SF outline
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
            superior=cranial,   # backend naming remains superior/inferior
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
        st.caption("Points + pivot rays + SF area (semi-transparent).")
        fig = make_plot(out)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter points and click **Compute** to see results and enable export.")
