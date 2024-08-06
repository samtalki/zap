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

FUELS_JOINED = {
    "nuclear": "Nuclear",
    "coal": "Coal",
    "CCGT": "Gas",
    "OCGT": "Gas",
    "hydro": "Hydro",
    "geothermal": "Geo",
    "onwind": "Wind",
    "offwind_floating": "Wind",
    "solar": "Solar",
}

JOINED_FUEL_NAMES = [
    "Nuclear",
    "Hydro",
    "Geo",
    "Wind",
    "Solar",
    "Coal",
    "Gas",
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
    "Nuclear": "purple",
    "Coal": "gray",
    "Gas": "brown",
    "Hydro": "blue",
    "Geo": "green",
    "Wind": "lightblue",
    "Solar": "yellow",
    "Biomass": "green",
    "Oil": "black",
}
for k in FUEL_COLORS:
    FUEL_COLORS[k] = "xkcd:" + FUEL_COLORS[k]

SEABORN_RC = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


seaborn.set_theme(style="white", rc=SEABORN_RC)


def total_capacity(capacities, fuels, fuel=None, group_fuels=False):
    if fuel is None:
        capacities = {f: total_capacity(capacities, fuels, f) for f in np.unique(fuels)}

        if group_fuels:
            # Group fuels according to FUELS_JOINED
            new_caps = {}
            for v in FUELS_JOINED.values():
                new_caps[v] = 0.0

            for f, cap in capacities.items():
                if f in FUELS_JOINED:
                    new_caps[FUELS_JOINED[f]] += cap

            return new_caps

        else:
            return capacities

    else:
        return np.sum(capacities[fuels == fuel])


def make_bar(
    ax,
    p0,
    p1,
    key=None,
    order=None,
    shift=0.0,
    base_color="C0",
    expansion_color="C1",
    width=0.8,
    label=None,
    label_base=True,
):
    if isinstance(p1, list):
        for i, pii in enumerate(p1):
            # Center shifts
            subshift = shift + (i + 1) * (1.0 / (len(p1) + 1)) - 0.5
            subwidth = width / (len(p1) + 1)
            make_bar(
                ax,
                p0,
                pii,
                key,
                order,
                subshift,
                expansion_color=f"C{i + 1}",
                width=subwidth,
                label_base=(i == 0),
                label=None if label is None else label[i],
            )
        return ax

    if key is not None:
        init = np.sum(p0[key])
        final = np.sum(p1[key])
        x = [0 + shift]

    else:
        init = np.array([p0[k] for k in order])
        final = np.array([p1[k] for k in order])
        x = np.arange(init.size) + shift

    base_label = "Initial" if label_base else None
    ax.bar(x, init, color=base_color, width=width, label=base_label)
    ax.bar(x, final - init, bottom=init, color=expansion_color, width=width, label=label)

    if key is not None:
        ax.set_xticks([])
        ax.set_xlabel(key)
    else:
        ax.set_xticks(range(len(x)), [o for o in order])

    return ax


def capacity_plot(p0, p1, devices, group_fuels=True, fig=None, axes=None, label=None):
    # Make bar plot
    if fig is None:
        fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.5), width_ratios=[1, 1, 1, 10])

    # Plot transmission and storage expansion
    make_bar(axes[0], p0, p1, "ac_line")
    make_bar(axes[1], p0, p1, "dc_line")
    make_bar(axes[2], p0, p1, "battery")

    # Plot generation expansion
    generators = devices[0]
    fuels = generators.fuel_type.reshape(-1, 1)
    gen0 = total_capacity(p0["generator"], fuels, group_fuels=group_fuels)

    if isinstance(p1, list):
        gen1 = [total_capacity(pii["generator"], fuels, group_fuels=group_fuels) for pii in p1]
    else:
        gen1 = total_capacity(p1["generator"], fuels, group_fuels=group_fuels)

    order = JOINED_FUEL_NAMES if group_fuels else FUEL_NAMES

    make_bar(axes[3], gen0, gen1, order=order, label=label)
    axes[3].tick_params(axis="x", labelrotation=0, labelsize=8)

    # Labels / limits
    axes[0].set_xlabel("AC Line")
    axes[1].set_xlabel("DC Line")
    axes[2].set_xlabel("Battery")
    axes[3].set_xlabel("Generator")
    axes[0].set_ylabel("Capacity (GW)")
    axes[0].set_ylim(0, 1500)
    axes[1].set_ylim(0, 60)
    axes[2].set_ylim(0, 60)
    axes[3].set_ylim(0, 250)

    axes[3].legend()

    # Finalize figure
    fig.align_labels()
    fig.tight_layout()
    return fig, axes


def stackplot(p1, layer, y1=None, fig=None, ax=None, legend=True, group_fuels=True):
    if fig is None:
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
    if group_fuels:
        new_gen_per_period = {f: np.zeros_like(gen_per_period[0]) for f in JOINED_FUEL_NAMES}
        for f, cap in zip(FUEL_NAMES, gen_per_period):
            if f in FUELS_JOINED:
                new_gen_per_period[FUELS_JOINED[f]] += cap

        gen_per_period = [new_gen_per_period[f] for f in JOINED_FUEL_NAMES]

    fuel_names = JOINED_FUEL_NAMES if group_fuels else FUEL_NAMES
    ax.stackplot(
        t,
        gen_per_period,
        labels=[f[:7] for f in fuel_names],
        colors=[FUEL_COLORS[f] for f in fuel_names],
    )

    # Plot battery output
    bat_power = y1.power[-2][0]
    total_bat_power = np.sum(bat_power, axis=0)
    ax.fill_between(
        t,
        total_load,
        total_load - total_bat_power,
        color="xkcd:pink",
        alpha=0.5,
        label="Battery",
    )

    # Tune figure
    if legend:
        ax.legend(fontsize=8, bbox_to_anchor=(1.2, 0.5), loc="center right")
    ax.set_xlim(np.min(t), np.max(t))
    ax.set_ylim(0, 200)

    fig.tight_layout()

    return fig, ax
