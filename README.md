# zap ⚡


## Setup

```zsh
mamba env create -n zap --file environment.yaml
```


## Environment Management

**Updating Packages.** Change the `environment.yaml` then sync (see below).

**Syncing Environment.** Run one of the following commands.

```zsh
mamba update --file environment.yaml  # Mamba
# or
mamba env update --file environment.yaml  # Conda with Mamba solver
```

I haven't figured out a good way to lock the environment yet (i.e., fix all the dependency versions),
but this seems to work fine for the time being.