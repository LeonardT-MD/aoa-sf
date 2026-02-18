from __future__ import annotations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .core import AoSFOutput


def plot_aosf(out: AoSFOutput, savepath: str | None = None, show: bool = False):
    """
    Quick 3D visualization: SI triangle + ML triangle + SF polygon, pivot and centroid.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    pivot = out.pivot
    s, i, m, l = out.superior, out.inferior, out.medial, out.lateral

    tri_si = [s, i, pivot]
    tri_ml = [m, l, pivot]
    ax.add_collection3d(Poly3DCollection([tri_si], alpha=0.25))
    ax.add_collection3d(Poly3DCollection([tri_ml], alpha=0.25))

    quad = [s, i, m, l]
    ax.add_collection3d(Poly3DCollection([quad], alpha=0.12))

    ax.scatter(*pivot, s=70, label="Pivot")
    ax.scatter(*out.sf_centroid, s=70, label="SF centroid")

    for pt, lab in [(s, "S"), (i, "I"), (m, "M"), (l, "L")]:
        ax.scatter(*pt, s=35)
        ax.text(pt[0], pt[1], pt[2], lab)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    txt = (
        f"AoA_SI: {out.aoa_si_deg:.2f}° | AoA_ML: {out.aoa_ml_deg:.2f}°\n"
        f"SF area: {out.sf_area:.2f} | centroid-to-pivot: {out.centroid_to_pivot:.2f}"
    )
    ax.text2D(0.02, 0.02, txt, transform=ax.transAxes)

    ax.legend(loc="upper left")

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
