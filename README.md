
# aoa-sf

Compute **Angle of Attack** (AoA) and **Surgical Freedom** (SF) from:
- a **pivot** (target) point
- four **cardinal** points: **superior, inferior, medial, lateral**

Outputs:
- **AoA_SI** (superior–inferior) at the pivot
- **AoA_ML** (medial–lateral) at the pivot
- **Surgical Freedom area (SF)** as polygon area of the four points in their best-fit plane (SVD/PCA projection + shoelace)
- **SF centroid (3D)** and **centroid-to-pivot distance**

Optional normalization:
- Rescale all 4 cardinal points to a **fixed radius** from the pivot (`absolute`)
- Rescale each point radius by a **multiplicative factor** (`relative`)

The GUI includes:
- units label (e.g., mm)
- coordinate system label (LPS/RAS/Scanner/Custom)
- export panel to download CSV outputs (single-row metrics+points; long-format points)

---

## Install (local)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -U pip
pip install -e .
```

---

## CLI usage

Input JSON must include: pivot, superior, inferior, medial, lateral.

```bash
aoa_sf --in examples/example_points.json
aoa_sf --in examples/example_points.json --rescale absolute --radius 20 --plot demo.png
aoa_sf --in examples/example_points.json --rescale relative --factor 0.8
```

---

## Python usage

```python
import json
from aoa_sf import compute_aosf
from aoa_sf.plotting import plot_aosf

with open("examples/example_points.json") as f:
    d = json.load(f)

out = compute_aosf(
    pivot=d["pivot"],
    superior=d["superior"],
    inferior=d["inferior"],
    medial=d["medial"],
    lateral=d["lateral"],
    rescale={"mode": "none"}
)

print(out.aoa_si_deg, out.aoa_ml_deg, out.sf_area)
plot_aosf(out, savepath="demo.png")
```

---

## GUI (interactive point entry + 3D plot + CSV export)

Install with UI extras:

```bash
pip install -e '.[ui]'
```

Launch:

```bash
aoa_sf_gui
```

The GUI provides:
- interactive point entry
- rescaling options (none / absolute / relative)
- interactive 3D visualization (Plotly)
- export panel:
  - single-row CSV (metrics + points + metadata)
  - long-format CSV (one row per point)

---

## Notes on geometry

- AoA is computed at the pivot using the law of cosines on the triangle defined by (pivot, pointA, pointB).
- SF area is computed from the 4 cardinal points after projecting them to their best-fit plane (SVD/PCA),
  ordering them around the centroid in 2D, then applying the shoelace formula.
=======

