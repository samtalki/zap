import marimo

__generated_with = "0.4.2"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import yaml
    import importlib

    import zap
    return importlib, mo, plt, yaml, zap


@app.cell
def __(config, runner):
    data = runner.load_dataset(config)
    return data,


@app.cell
def __(importlib):
    from experiments import runner
    _ = importlib.reload(runner)
    return runner,


@app.cell
def __(yaml):
    with open("experiments/config/default.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config, f


@app.cell
def __(config, data, runner):
    problem = runner.setup_problem(data, config)
    return problem,


@app.cell
def __(config, problem, runner):
    relax = runner.solve_relaxed_problem(problem, config)
    return relax,


@app.cell
def __(problem, relax, result):
    _J = problem["problem"]

    print(_J(**_J.initialize_parameters(None)))
    print(_J(**relax["relaxed_parameters"]))
    print(_J(**result["parameters"]))

    print(relax["lower_bound"])
    return


@app.cell
def __(config, problem, relax, runner):
    result = runner.solve_problem(problem, relax, config)
    return result,


@app.cell
def __(result):
    result["history"]
    return


if __name__ == "__main__":
    app.run()
