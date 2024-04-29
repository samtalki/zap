# Example
# python experiments/pypsa/generate.py /Users/degleris/Data/pypsa-usa/

import os
import subprocess
import sys
import yaml
import zap
import pypsa
from pathlib import Path

pypsa_path = Path(sys.argv[1]).resolve() / "workflow"

ZAP_PATH = Path(zap.__file__).resolve().parent.parent
config_path = Path(__file__).resolve().parent / "config.yaml"
temp_config_path = Path(__file__).resolve().parent / "temp_config.yaml"

NUM_NODES = [42, 100, 200, 240, 500, 1000]


for efs_case in ["medium", "high"]:
    run_name = f"zap_{efs_case}"

    # Open config
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Change some settings
    config["run"]["name"] = run_name
    config["scenario"]["clusters"] = NUM_NODES
    config["scenario"]["planning_horizon"] = [2050]

    config["electricity"]["extendable_carriers"]["Generator"] += ["nuclear"]
    config["electricity"]["demand"]["scenario"]["efs_case"] = efs_case

    # Save the new config
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # Run the snakemake command
    snake_path = pypsa_path / "Snakefile"
    print("Firing up the Snakemake workflow...")
    command = (
        "snakemake --until add_extra_components --cores 4 "
        + f"--directory {pypsa_path} "
        + f"--snakefile {snake_path} "
        + f"--configfile {temp_config_path}"
    )

    print("Running command: ", command)
    print("\n\n\n\n\n\n")

    subprocess.run(
        f"""source ~/.zshrc
        micromamba activate pypsa-usa
        micromamba info
        {command}""",
        shell=True,
        executable="/bin/zsh",
    )

    print("\n\n\n\n\n\nWorkflow completed!")

    # Remove the temp config
    os.remove(temp_config_path)

    # Save data
    resource_path = pypsa_path / "resources" / run_name / "western"
    for num_nodes in NUM_NODES:
        for ext in ["", "_ec"]:
            # Import network
            network_path = resource_path / f"elec_s_{num_nodes}{ext}.nc"
            pn = pypsa.Network(network_path)

            # Export network
            output_path = ZAP_PATH / "data/pypsa/western"
            output_path = output_path / f"load_{efs_case}" / f"elec_s_{num_nodes}{ext}"
            output_path.mkdir(parents=True, exist_ok=True)
            pn.export_to_csv_folder(output_path)
