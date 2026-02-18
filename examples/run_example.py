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
    rescale={"mode": "none"},
)

print("AoA_SI:", out.aoa_si_deg)
print("AoA_ML:", out.aoa_ml_deg)
print("SF_area:", out.sf_area)
print("SF_centroid:", out.sf_centroid)
print("Centroid_to_pivot:", out.centroid_to_pivot)

plot_aosf(out, savepath="example_output.png")
print("Saved example_output.png")
