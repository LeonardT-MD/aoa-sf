from __future__ import annotations
import json
import argparse

from .core import compute_aosf
from .plotting import plot_aosf


def main():
    ap = argparse.ArgumentParser(
        prog="aoa_sf",
        description="Compute AoA (S–I and M–L) and Surgical Freedom (SF) from pivot + superior/inferior/medial/lateral points."
    )
    ap.add_argument("--in", dest="infile", required=True, help="JSON file with pivot, superior, inferior, medial, lateral arrays.")
    ap.add_argument("--rescale", choices=["none", "absolute", "relative"], default="none")
    ap.add_argument("--radius", type=float, default=None, help="Used if --rescale absolute")
    ap.add_argument("--factor", type=float, default=None, help="Used if --rescale relative")
    ap.add_argument("--plot", default=None, help="Optional path to save a PNG figure (matplotlib).")
    args = ap.parse_args()

    with open(args.infile, "r") as f:
        d = json.load(f)

    if args.rescale == "absolute":
        if args.radius is None:
            raise SystemExit("--radius is required for --rescale absolute")
        rescale = {"mode": "absolute", "radius": args.radius}
    elif args.rescale == "relative":
        if args.factor is None:
            raise SystemExit("--factor is required for --rescale relative")
        rescale = {"mode": "relative", "factor": args.factor}
    else:
        rescale = {"mode": "none"}

    out = compute_aosf(
        pivot=d["pivot"],
        superior=d["superior"],
        inferior=d["inferior"],
        medial=d["medial"],
        lateral=d["lateral"],
        rescale=rescale,
    )

    print(f"AoA_SI_deg: {out.aoa_si_deg:.6f}")
    print(f"AoA_ML_deg: {out.aoa_ml_deg:.6f}")
    print(f"SF_area: {out.sf_area:.6f}")
    print(f"SF_centroid_xyz: {out.sf_centroid.tolist()}")
    print(f"Centroid_to_pivot: {out.centroid_to_pivot:.6f}")

    if args.plot:
        plot_aosf(out, savepath=args.plot)


if __name__ == "__main__":
    main()
