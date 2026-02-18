# AoA–SF: Angle of Attack and Surgical Freedom Geometry Toolkit

AoA–SF is a Python package and interactive graphical interface for quantitative geometric analysis of:

- Vertical Angle of Attack (AoA)  
- Horizontal Angle of Attack (AoA)  
- Surgical Freedom (SF) area  

The package computes these metrics from a pivot point and four cardinal three-dimensional points (Cranial, Caudal, Medial, Lateral). It is designed for reproducible neuroanatomical and surgical corridor analysis in research and simulation settings.

---

## 1. Conceptual Framework

### Input Geometry

The model requires:

- A pivot point (target point)
- Four surrounding cardinal points:
  - Cranial
  - Caudal
  - Medial
  - Lateral

All coordinates are defined in 3D space using any consistent coordinate system (e.g., LPS, RAS, scanner coordinates).

---

## 2. Angle of Attack (AoA)

### 2.1 Vertical AoA

The Vertical Angle of Attack is defined as the angle at the pivot between the two vectors:

- Cranial − Pivot  
- Caudal − Pivot  

It quantifies the superior–inferior opening around the pivot.

### 2.2 Horizontal AoA

The Horizontal Angle of Attack is defined as the angle at the pivot between:

- Medial − Pivot  
- Lateral − Pivot  

It quantifies the medial–lateral opening around the pivot.

### 2.3 Mathematical Definition

Both angles are computed directly in 3D using the dot product:

\[
\theta = \arccos \left( \frac{u \cdot v}{\|u\|\|v\|} \right)
\]

This ensures:

- Independence from any plane fitting
- Rotational invariance
- Numerical stability (with clipping to [-1, 1])

Angles are expressed in degrees.

---

## 3. Surgical Freedom (SF)

Surgical Freedom is defined as the area of the polygon formed by the four cardinal points.

### 3.1 Plane Fitting

Because the four points are generally not perfectly coplanar in real anatomical data, SF is computed on:

The best-fit least-squares plane of the four cardinal points.

Plane extraction is performed via Singular Value Decomposition (SVD) on the centered point cloud.

### 3.2 Minimal Displacement Projection

Each of the four points is orthogonally projected onto the fitted plane. This ensures:

- Minimal geometric distortion
- Minimal squared orthogonal displacement
- No dependency on global XY alignment

### 3.3 Area Computation

After projection:

1. Points are expressed in 2D coordinates within the plane basis.
2. Points are ordered clockwise around their centroid.
3. The polygon area is computed using the Shoelace formula.

The centroid of the polygon is computed in plane coordinates and then lifted back into 3D space.

SF is expressed in squared units corresponding to the coordinate units (e.g., mm²).

---

## 4. Rescaling Logic

Rescaling is strictly ray-preserving.

Each point is moved along its original pivot-to-point trajectory without altering directional geometry.

For a given point \( p \):

\[
p' = pivot + \frac{(p - pivot)}{\|p - pivot\|} \cdot d
\]

Where:

- \( d \) is the new distance from the pivot.

### 4.1 Rescale Modes

| Mode       | Description |
|------------|------------|
| none       | No rescaling |
| absolute   | All four points are moved to the same new distance from pivot |
| relative   | Each original pivot distance is multiplied by a scaling factor |

### 4.2 Invariance Property

Because directions are preserved, both Vertical and Horizontal AoA remain invariant under rescaling.

Only SF area changes as a function of scaling.

---

## 5. Graphical User Interface

The package includes a Streamlit-based graphical interface.

### 5.1 Launch

After installation:

```
aoa_sf_gui
```

or alternatively:

```
streamlit run src/aoa_sf/streamlit_app.py
```

### 5.2 GUI Features

- Manual entry of pivot and four cardinal points
- Unit labeling (e.g., mm, cm, voxels)
- Coordinate system labeling (LPS, RAS, Scanner, Custom)
- Ray-preserving rescaling controls
- Interactive 3D visualization including:
  - Pivot
  - Cardinal points
  - Pivot-to-point rays
  - Semi-transparent SF area projected onto best-fit plane
- CSV export:
  - Single-row metrics output
  - Long-format point coordinate output

---

## 6. Installation

### 6.1 Clone Repository

```
git clone https://github.com/LeonardT-MD/aoa-sf.git
cd aoa-sf
```

### 6.2 Create Virtual Environment (Recommended)

```
python3 -m venv .venv
source .venv/bin/activate
```

### 6.3 Install Package

```
pip install -U pip setuptools wheel
pip install -e '.[ui]'
```

---

## 7. Command-Line Interface

Example usage:

```
aoa_sf --in examples/example_points.json
```

Optional figure output:

```
aoa_sf --in examples/example_points.json --plot output.png
```

---

## 8. Programmatic Usage

```
from aoa_sf import compute_aosf
```

Returns an object containing:

- aoa_vertical_deg
- aoa_horizontal_deg
- sf_area
- sf_centroid
- centroid_to_pivot
- Ray-preserved cranial, caudal, medial, lateral points
- Ordered plane-projected SF polygon for visualization

---

## 9. Verification of AoA Invariance

Example:

```
out0 = compute_aosf(..., rescale={"mode":"none"})
out1 = compute_aosf(..., rescale={"mode":"absolute","distance":80})

abs(out0.aoa_vertical_deg - out1.aoa_vertical_deg)
```

The difference should be on the order of floating-point precision (approximately 1e-12 to 1e-9).

---

## 10. Repository Structure

```
aoa-sf/
├── src/
│   └── aoa_sf/
│       ├── geometry.py
│       ├── core.py
│       ├── streamlit_app.py
│       ├── cli.py
│       └── __init__.py
├── examples/
├── pyproject.toml
└── README.md
```

---

## 11. Intended Applications

- Skull base approach quantification
- Surgical corridor modeling
- Neuroanatomical simulation
- Photogrammetric or cadaveric measurements
- Navigation-based geometric analysis

---

## 12. License

MIT License.

---

## 13. Citation

If this toolkit is used in academic work, please cite the repository and associated publications:

Tariciotti et al.  
AoA–SF: Quantitative Angle of Attack and Surgical Freedom Analysis Toolkit.  
GitHub repository.
