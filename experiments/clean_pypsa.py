import pypsa

NUM_NODES = range(100, stop=1100, step=100)

for num_nodes in NUM_NODES:
    for ext in ["", "_ec"]:
        pn = pypsa.Network(f"~/pypsa-usa/workflow/resources/western/elec_s_{num_nodes}{ext}.nc")
        pn.export_to_csv_folder(f"../data/pypsa/western/elec_s_{num_nodes}{ext}/")
