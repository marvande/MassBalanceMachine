from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_domain_shift(shift: dict, monthly_cols: list[str], static_cols: list[str]):

    var_keys = {f"D_mmd2_{col}": col for col in monthly_cols + static_cols}
    records_mmd2 = [
        (label, shift[key]) for key, label in var_keys.items() if key in shift
    ]
    records_mmd2.sort(key=lambda x: x[1], reverse=True)

    # Energy distance per variable
    var_keys_e = {f"D_energy_{col}": col for col in monthly_cols + static_cols}
    records_energy = {
        label: shift[key] for key, label in var_keys_e.items() if key in shift
    }

    labels, values_mmd2 = zip(*records_mmd2)
    values_energy = [records_energy.get(l, np.nan) for l in labels]

    radiation_vars = {"sshf", "slhf", "ssrd", "str", "fal"}
    topo_vars = {"slope", "svf", "aspect", "ELEVATION_DIFFERENCE"}

    def _color(label):
        if label in radiation_vars:
            return "#D85A30"
        if label in topo_vars:
            return "#534AB7"
        return "#888780"

    colors = [_color(l) for l in labels]

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 2.2, 2.2]}
    )

    # ── Left: summary bars ────────────────────────────────────────────────────
    ax_left = axes[0]
    summary_labels = ["joint", "climate", "topo"]
    summary_mmd2 = [
        shift["D_mmd2_joint"],
        shift["D_mmd2_climate"],
        shift["D_mmd2_topo"],
    ]
    summary_energy = [
        shift["D_energy_joint"],
        shift["D_energy_climate"],
        shift["D_energy_topo"],
    ]
    summary_colors = ["#3d3d3a", "#D85A30", "#534AB7"]

    x = np.arange(len(summary_labels))
    w = 0.35
    for i, (mv, ev, c) in enumerate(zip(summary_mmd2, summary_energy, summary_colors)):
        ax_left.barh(
            x[i] + w / 2, mv, height=w, color=c, label="MMD²" if i == 0 else ""
        )
        ax_left.barh(
            x[i] - w / 2,
            ev,
            height=w,
            color=c,
            alpha=0.45,
            label="Energy" if i == 0 else "",
        )
        ax_left.text(mv + 0.005, x[i] + w / 2, f"{mv:.3f}", va="center", fontsize=9)
        ax_left.text(ev + 0.005, x[i] - w / 2, f"{ev:.3f}", va="center", fontsize=9)

    ax_left.set_yticks(x)
    ax_left.set_yticklabels(summary_labels)
    ax_left.set_xlabel("Distance", fontsize=11)
    ax_left.set_title("Summary", fontsize=12, pad=10)
    ax_left.legend(frameon=False, fontsize=9)
    ax_left.spines[["top", "right"]].set_visible(False)
    ax_left.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax_left.set_axisbelow(True)

    # ── Middle: per-variable MMD² ─────────────────────────────────────────────
    ax_mid = axes[1]
    y = np.arange(len(labels))
    bars = ax_mid.barh(y, values_mmd2, color=colors, height=0.6)
    ax_mid.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax_mid.set_yticks(y)
    ax_mid.set_yticklabels(labels, fontsize=10)
    ax_mid.set_xlim(0, max(values_mmd2) * 1.25)
    ax_mid.set_xlabel("MMD²", fontsize=11)
    ax_mid.set_title("Per-variable MMD²  (Iceland → Switzerland)", fontsize=12, pad=10)
    ax_mid.invert_yaxis()
    ax_mid.spines[["top", "right"]].set_visible(False)
    ax_mid.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax_mid.set_axisbelow(True)

    # ── Right: per-variable Energy ────────────────────────────────────────────
    ax_right = axes[2]
    bars2 = ax_right.barh(y, values_energy, color=colors, height=0.6, alpha=0.7)
    ax_right.bar_label(bars2, fmt="%.3f", padding=4, fontsize=9)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels(labels, fontsize=10)
    ax_right.set_xlim(0, max(values_energy) * 1.25)
    ax_right.set_xlabel("Energy distance", fontsize=11)
    ax_right.set_title(
        "Per-variable Energy dist.  (Iceland → Switzerland)", fontsize=12, pad=10
    )
    ax_right.invert_yaxis()
    ax_right.spines[["top", "right"]].set_visible(False)
    ax_right.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax_right.set_axisbelow(True)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#D85A30", label="radiation / energy flux"),
        mpatches.Patch(color="#534AB7", label="topographic"),
        mpatches.Patch(color="#888780", label="temperature / precip"),
    ]
    ax_right.legend(
        handles=legend_handles, loc="lower right", frameon=False, fontsize=9
    )

    fig.tight_layout(w_pad=3)
    return fig
