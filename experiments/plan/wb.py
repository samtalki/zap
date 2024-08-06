import pandas as pd
import wandb


def get_wandb_data(api: wandb.Api):
    runs = api.runs("s3l/zap")

    wandb_data = {
        "name": [],
        "id": [],
        "index": [],
        "emissions_weight": [],
        "battery_cost_scale": [],
        "initial_state": [],
        "batch_strategy": [],
        "hash": [],
    }

    for r in runs:
        wandb_data["name"].append(r.name)
        wandb_data["id"].append(r.config.get("id", ""))
        wandb_data["index"].append(r.config.get("index", -1))
        wandb_data["hash"].append(r.id)

        wandb_data["emissions_weight"].append(float(r.config["problem"]["emissions_weight"]))
        wandb_data["battery_cost_scale"].append(
            float(r.config["data"].get("battery_cost_scale", 1.0))
        )

        wandb_data["initial_state"].append(r.config["optimizer"]["initial_state"])
        wandb_data["batch_strategy"].append(r.config["optimizer"].get("batch_strategy", ""))

    return pd.DataFrame(wandb_data)


def get_run_history(api, _hash, cache=None):
    if cache is None:
        cache = {}

    if _hash not in cache:
        run = api.run(f"s3l/zap/{_hash}")
        cache[_hash] = run.history()

    return cache[_hash]
