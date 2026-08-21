"""Shared pytest fixtures.

Includes the IoT-dataset auto-download fixtures so integration tests can
run on a clean checkout without manual setup. Datasets are cached under
``results/data/`` (see ``--data-dir``) and reused on subsequent runs.

The fixtures gracefully ``skip`` if the network is unreachable or if a
Kaggle-gated dataset hasn't been pre-downloaded.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "results" / "data"

# Make `scripts/` importable (it isn't a package but a few integration
# tests want to call `scripts.run_sweep.main` to exercise the CLI in
# the same process).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _has_internet(host: str = "archive.ics.uci.edu", port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Resolved cache directory for IoT datasets.

    Honours ``SAFEL_DT_DATA_DIR`` for users with disk constraints or
    shared mounts. Defaults to ``<repo>/results/data``.
    """
    env = os.environ.get("SAFEL_DT_DATA_DIR")
    root = Path(env).resolve() if env else _DEFAULT_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def nbaiot_root(data_dir: Path) -> Path:
    """Ensure N-BaIoT is cached locally; auto-download from UCI if missing.

    Tests that depend on this fixture must use::

        @pytest.mark.requires_nbaiot
    """
    from safel_dt.data.nbaiot import ensure_nbaiot

    cache = data_dir / "nbaiot"
    already_present = cache.exists() and any(cache.iterdir())
    if not already_present and not _has_internet():
        pytest.skip("N-BaIoT not cached and no network access; skipping.")
    try:
        return ensure_nbaiot(data_dir)
    except Exception as exc:
        pytest.skip(f"N-BaIoT download failed: {exc}")


@pytest.fixture(scope="session")
def edge_iiotset_csv(data_dir: Path) -> Path:
    """Resolve Edge-IIoTset CSV from the local cache.

    Kaggle-gated; tests that need it must use::

        @pytest.mark.requires_edge_iiotset
    """
    from safel_dt.data.edge_iiotset import EdgeIIoTsetMissing, _resolve_csv

    try:
        return _resolve_csv(data_dir)
    except EdgeIIoTsetMissing as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def toniot_csv(data_dir: Path) -> Path:
    """Resolve the TON_IoT network-flow CSV from the local cache.

    Tests that depend on this fixture must use::

        @pytest.mark.requires_toniot
    """
    from safel_dt.data.toniot import TonIotMissing, _resolve_csv

    try:
        return _resolve_csv(data_dir)
    except TonIotMissing as exc:
        pytest.skip(str(exc))
