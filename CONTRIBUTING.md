# Contributing

## Development setup

```bash
git clone https://github.com/<your-username>/clinical-curator.git
cd clinical-curator
conda env create -f environment.yml
conda activate clinical-curator
pip install -e ".[dev]"
```

## Code style

This project uses **ruff** for linting and formatting:

```bash
ruff check src/ scripts/
ruff format src/ scripts/
```

## Running tests

```bash
pytest tests/ -v
```

## Pull request checklist

- [ ] All new public functions have docstrings and type annotations
- [ ] `ruff check` passes with no errors
- [ ] Unit tests added for new functionality
- [ ] `README.md` updated if API or CLI arguments changed

## Adding a new model

1. Create `src/models/<name>.py` with the `nn.Module` class
2. Export it in `src/models/__init__.py`
3. Register it in `backend/main.py` under `MODELS`
4. Add a config file at `configs/models/<name>.yaml`
5. Add a training script at `scripts/train_<review>.py` or extend an existing one
