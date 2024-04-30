import matplotlib.pyplot as plt
import numpy as np
import seaborn

FUEL_NAMES = [
    "solar",
    "onwind",
    "offwind_floating",
    "hydro",
    "geothermal",
    "nuclear",
    "biomass",
    "CCGT",
    "OCGT",
    "coal",
    "oil",
]

# rc=Dict(
#     "axes.titlesize" => 8,
#     "legend.fontsize" => 8,
# )

seaborn.set_theme(
    style="white",
    rc={
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    },
)


def total_capacity(capacities, fuels, fuel=None):
    if fuel is None:
        return {f: total_capacity(capacities, fuels, f) for f in np.unique(fuels)}
    else:
        return np.sum(np.multiply(capacities, fuels == fuel))


def make_bar(ax, p0, p1, key=None, order=None):
    if key is not None:
        init = np.sum(p0[key])
        final = np.sum(p1[key])
        x = [0]

    else:
        init = np.array([p0[k] for k in order])
        final = np.array([p1[k] for k in order])
        x = np.arange(init.size)

    ax.bar(x, init)
    ax.bar(x, final - init, bottom=init)

    ax.set_xlim(np.min(x) - 0.6, np.max(x) + 0.6)

    if key is not None:
        ax.set_xticks([])
        ax.set_xlabel(key)
    else:
        ax.set_xticks(x, [o[:7] for o in order])

    return ax


def capacity_plot(p0, p1, devices):
    # Make bar plot
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.5), width_ratios=[1, 1, 1, 12])

    # Plot transmission and storage expansion
    make_bar(axes[0], p0, p1, "ac_line")
    make_bar(axes[1], p0, p1, "dc_line")
    make_bar(axes[2], p0, p1, "battery")

    # Plot generation expansion
    generators = devices[0]
    fuels = generators.fuel_type.reshape(-1, 1)
    gen0 = total_capacity(p0["generator"], fuels)
    gen1 = total_capacity(p1["generator"], fuels)

    make_bar(axes[3], gen0, gen1, order=FUEL_NAMES)
    axes[3].set_xlabel("generator")
    axes[3].tick_params(axis="x", labelrotation=45, labelsize=8)

    # Labels / limits
    axes[0].set_ylabel("Capacity (GW)")
    axes[1].set_ylim(*axes[3].get_ylim())
    axes[2].set_ylim(*axes[3].get_ylim())

    # Finalize figure
    fig.align_labels()
    fig.tight_layout()
    return fig, axes


def stackplot(ax, p0, p1, layer):
    pass
