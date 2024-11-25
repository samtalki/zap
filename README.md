# zap ⚡


## Documentation

See the [documentation page](https://degleris1.github.io/zap/).


## Installation

We recommend using `pip` to install directly from Github.

```zsh
python3 -m pip install "zap @ git+https://github.com/degleris1/zap.git"  # No PyPSA
```

Dependency on `pypsa` is optional.
This can be installed by running the following command (instead of the above).

```zsh
python3 -m pip install "zap[pypsa] @ git+https://github.com/degleris1/zap.git"  # With PyPSA
```

### Developer Setup

For developers, we recommend installing [poetry](https://python-poetry.org/docs/).
Then clone the repo and install depedencies:

```zsh
git clone https://github.com/degleris1/zap.git
cd zap
poetry install --all-extras --with experiment
poetry shell  # Start a shell in a local virtual environment
```

Then run Python from the subshell.




## Basic Usage

Please download the Marimo notebook [here](https://github.com/degleris1/zap/blob/main/demo/01_basics.html).




## Reproducing Experiments

To reproduce experiments from our published work, use the [developer setup](#developer-setup).
Some experiments require a [Mosek](https://www.mosek.com/) license.




## Citation

If you use this package in published work, we ask that you cite our paper ([preprint](https://arxiv.org/abs/2410.17203)):

```bibtex
@article{degleris2024gpu,
  title={GPU Accelerated Security Constrained Optimal Power Flow},
  author={Anthony Degleris and Abbas El Gamal and Ram Rajagopal},
  journal={arXiv preprint arXiv:2404.01255},
  year={2024},
}
```

If you also used the gradient-based planning methods, please also cite our paper on planning ([preprint](https://arxiv.org/abs/2404.01255)):

```bibtex
@article{degleris2024gradient,
  title={Gradient Methods for Scalable Multi-value Electricity Network Expansion Planning},
  author={Degleris, Anthony and Gamal, Abbas El and Rajagopal, Ram},
  journal={arXiv preprint arXiv:2404.01255},
  year={2024}
}
```