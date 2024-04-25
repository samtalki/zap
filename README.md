# zap ⚡


## Setup

```zsh
mamba env create -n zap --file environment.yaml
```


## Environment Management

**Updating Packages.** Change the `environment.yaml` then sync (see below).

**Syncing Environment.** Run one of the following commands.

On Perlmutter:
```zsh
mamba env update --file environment.yaml --prefix $ZAP_ENV  # Conda with Mamba solver
```

On Macbook:
```zsh
mamba update --file environment.yaml  # Mamba
```

I haven't figured out a good way to lock the environment yet (i.e., fix all the dependency versions),
but this seems to work fine for the time being.