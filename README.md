# zap ⚡




## Environment Management

Install environment:

```zsh
mamba env create -n zap
mamba update -f environment-lock.yml
```

Update environment:

```zsh
mamba update -f environment.yml --prune
mamba env export --no-builds > environment-lock.yml
```
