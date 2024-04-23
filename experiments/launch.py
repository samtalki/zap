import sys
import os
import runner


config_path = sys.argv[1]
config = runner.load_config(config_path)
system = config["system"]

# Configure file paths
output_file = runner.datadir("slurm", f"{config['name']}.out")
script_file = runner.datadir("scripts", f"{config['name']}.sh")

output_file.parent.mkdir(parents=True, exist_ok=True)
script_file.parent.mkdir(parents=True, exist_ok=True)

# Write slurm script
slurm_script = f"""#!/bin/bash
#SBATCH --job-name={config["name"]}
#SBATCH --output={output_file}
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={system["threads"]}
#SBATCH --time={system["runtime"]}

conda init bash
conda activate $ZAP_ENV
srun python -u experiments/runner.py {config_path}
"""

with open(script_file, "w") as f:
    f.write(slurm_script)

# Launch
print(f"Launching job {config['name']}")
os.system(f"sbatch {script_file}")
