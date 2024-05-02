import matplotlib.pyplot as plt
import numpy as np
import seaborn

FUEL_NAMES = [
    "nuclear",
    "coal",
    "CCGT",
    "OCGT",
    "hydro",
    "geothermal",
    "onwind",
    "offwind_floating",
    "solar",
    "biomass",
    "oil",
]

FUEL_COLORS = {
    "solar": "yellow",
    "onwind": "lightblue",
    "offwind_floating": "turquoise",
    "hydro": "blue",
    "geothermal": "brown",
    "nuclear": "purple",
    "biomass": "green",
    "CCGT": "orange",
    "OCGT": "red",
    "coal": "gray",
    "oil": "black",
}

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
        return np.sum(capacities[fuels == fuel])


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
    axes[0].set_ylim(0, 1500)
    axes[1].set_ylim(0, 100)
    axes[2].set_ylim(0, 100)
    axes[3].set_ylim(0, 300)

    # Finalize figure
    fig.align_labels()
    fig.tight_layout()
    return fig, axes


def stackplot(p1, layer, y1=None):
    fig, ax = plt.subplots(figsize=(6.5, 3))

    if y1 is None:
        y1 = layer(**p1)

    devices = layer.devices

    # Plot total load
    loads = devices[1]
    total_load = -np.sum(loads.min_power * loads.nominal_capacity, axis=0)
    t = np.arange(total_load.size)
    ax.plot(t, total_load, color="black")

    # Stackplot generation
    gens = devices[0]
    gen_power = y1.power[0][0]
    fuels = gens.fuel_type

    gen_per_period = [np.sum(gen_power[fuels == f, :], axis=0) for f in FUEL_NAMES]

    ax.stackplot(
        t,
        gen_per_period,
        labels=[f[:7] for f in FUEL_NAMES],
        colors=[FUEL_COLORS[f] for f in FUEL_NAMES],
    )

    # Plot battery output
    bat_power = y1.power[-2][0]
    total_bat_power = np.sum(bat_power, axis=0)
    ax.fill_between(
        t,
        total_load,
        total_load - total_bat_power,
        color="pink",
        alpha=0.5,
        label="battery",
    )

    # Tune figure
    ax.legend(fontsize=8, bbox_to_anchor=(1.2, 0.5), loc="center right")
    ax.set_xlim(np.min(t), np.max(t))
    ax.set_ylim(0, 275)

    fig.tight_layout()

    return fig, ax
