# SAFEL-DT

Secure Adaptive Federated Learning for Hierarchical Digital Twins.

## Setup (Linux)

```bash
cd sec_hfl_drl
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Quick check

```bash
pytest tests/unit -q
```

## Run a smoke simulation

```bash
python -m scripts.run_simulation --dataset synthetic --rounds 5 --clients 6 --fogs 2
```
