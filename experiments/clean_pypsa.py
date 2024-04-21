import os
import pypsa

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
NUM_NODES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

for num_nodes in NUM_NODES:
    for ext in ["", "_ec"]:
        pn = pypsa.Network(f"~/pypsa-usa/workflow/resources/western/elec_s_{num_nodes}{ext}.nc")
        csv_folder_path = os.path.join(
            FILE_PATH, "..", "data", "pypsa", "western", f"elec_s_{num_nodes}{ext}"
        )
        pn.export_to_csv_folder(csv_folder_path)
