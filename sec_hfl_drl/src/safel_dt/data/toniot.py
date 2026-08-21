"""TON_IoT loader (Moustafa et al., UNSW Canberra).

Dataset: https://research.unsw.edu.au/projects/toniot-datasets

We use the **network-flow** slice (``train_test_network.csv``), which is
the most cited variant for FL+IDS comparisons in the 2023-2026
literature.  Schema (44 columns):

* ``label`` (int 0/1) -- binary IDS target.
* ``type`` (str) -- 10-class multi target: ``normal`` plus 9 attack
  families (``backdoor``, ``ddos``, ``dos``, ``injection``, ``password``,
  ``ransomware``, ``scanning``, ``xss``, ``mitm``).
* 42 raw network features: a mix of numerics (bytes / packets /
  duration), low-cardinality categoricals (``proto``, ``service``,
  ``conn_state``, ...) and high-cardinality strings (IPs, URIs, query
  bodies). We drop the high-cardinality strings (they leak labels and
  overfit), one-hot the low-cardinality columns, keep the rest as
  numeric, and replace missing / ``-`` placeholders with 0.

The loader splits the 211k flows uniformly at random across
``num_clients`` (default 9, matching the N-BaIoT topology). Dirichlet
skew is available through `data.partition.dirichlet_partition` if a
stronger non-IID setting is wanted.

The dataset is small enough (~30 MB) that we read it eagerly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

TonIotMode = Literal["binary", "multi"]


_DROP_COLUMNS: Final[tuple[str, ...]] = (
    "src_ip",
    "dst_ip",
    "dns_query",
    "http_uri",
    "ssl_subject",
    "ssl_issuer",
    "http_user_agent",
    "weird_name",
    "weird_addl",
    "http_orig_mime_types",
    "http_resp_mime_types",
)

_ONEHOT_COLUMNS: Final[tuple[str, ...]] = (
    "proto",
    "service",
    "conn_state",
    "http_method",
    "http_version",
    "ssl_version",
    "ssl_cipher",
    "weird_notice",
)

_DEFAULT_FILENAMES: Final[tuple[str, ...]] = (
    "train_test_network.csv",
    "Train_Test_Network.csv",
    "ToN_IoT_Train_Test_Network.csv",
)


class TonIotMissing(FileNotFoundError):
    """Raised when the TON_IoT cache is absent."""


def _resolve_csv(data_dir: Path) -> Path:
    """Locate the TON_IoT network-flow CSV under ``data_dir``.

    Accepts any of: ``<data_dir>/toniot/`` (canonical), ``<data_dir>/TONIOT/``
    (the casing used in the UNSW download), or ``<data_dir>/`` itself.
    """
    candidates: list[Path] = [data_dir]
    if data_dir.is_dir():
        for child in data_dir.iterdir():
            if child.is_dir() and child.name.lower() in {"toniot", "ton_iot", "ton-iot"}:
                candidates.append(child)
    for c in candidates:
        if not c.is_dir():
            continue
        for fname in _DEFAULT_FILENAMES:
            for hit in c.rglob(fname):
                return hit
    raise TonIotMissing(
        f"TON_IoT network-flow CSV not found under {data_dir}. "
        f"Expected one of {list(_DEFAULT_FILENAMES)}; "
        f"download from https://research.unsw.edu.au/projects/toniot-datasets ."
    )


def _preprocess_dataframe(df: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Drop ID columns, one-hot small categoricals, keep numerics."""
    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    df.columns = [c.strip() for c in df.columns]
    if "label" not in df.columns or "type" not in df.columns:
        raise ValueError(
            f"TON_IoT CSV missing 'label' / 'type' columns; got {list(df.columns)[:6]}..."
        )

    y_binary = df["label"].astype(np.int64).to_numpy()
    class_names = sorted(df["type"].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_multi = df["type"].astype(str).map(class_to_idx).astype(np.int64).to_numpy()

    feat = df.drop(columns=["label", "type"])
    feat = feat.drop(columns=[c for c in _DROP_COLUMNS if c in feat.columns], errors="ignore")

    onehot_cols = [c for c in _ONEHOT_COLUMNS if c in feat.columns]
    if onehot_cols:
        for c in onehot_cols:
            feat[c] = feat[c].astype(str).replace({"-": "missing", "": "missing"})
        feat = pd.get_dummies(feat, columns=onehot_cols, dummy_na=False, dtype=np.float32)

    feat = feat.replace({"-": 0, "": 0})
    feat = feat.apply(pd.to_numeric, errors="coerce")
    feat = feat.fillna(0.0)
    x = feat.to_numpy(dtype=np.float32)
    return x, y_binary, y_multi, class_names


class TonIotDataset(Dataset):
    """Slice of TON_IoT (network flows) as a torch Dataset."""

    def __init__(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        self._x = x.astype(np.float32)
        self._y = y.astype(np.int64)
        if mean is not None and std is not None:
            self._x = ((self._x - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)
        self.in_features = int(self._x.shape[1])

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._x[idx]),
            torch.tensor(int(self._y[idx]), dtype=torch.long),
        )


def load_toniot(
    *,
    data_dir: Path | str,
    mode: TonIotMode = "binary",
    num_clients: int = 9,
    max_samples: int | None = None,
    test_fraction: float = 0.2,
    seed: int = 0,
    normalise: bool = True,
) -> tuple[list[TonIotDataset], TonIotDataset, dict[str, object]]:
    """Load TON_IoT into ``num_clients`` train slices + a shared test set.

    Raises :class:`TonIotMissing` if the cache is absent.
    """
    if not (0.0 < test_fraction < 0.5):
        raise ValueError(f"test_fraction must be in (0, 0.5), got {test_fraction}")
    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0, got {num_clients}")

    data_dir = Path(data_dir)
    csv_path = _resolve_csv(data_dir)

    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    x_all, yb_all, ym_all, class_names = _preprocess_dataframe(df)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(x_all.shape[0])
    if max_samples is not None:
        idx = idx[:max_samples]
    x = x_all[idx]
    y = (yb_all if mode == "binary" else ym_all)[idx]

    n = x.shape[0]
    n_test = int(n * test_fraction)
    x_test, y_test = x[:n_test], y[:n_test]
    x_train, y_train = x[n_test:], y[n_test:]

    mean = std = None
    if normalise:
        mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)

    per_client_idx = np.array_split(np.arange(len(x_train)), num_clients)
    client_datasets = [
        TonIotDataset(x=x_train[i], y=y_train[i], mean=mean, std=std) for i in per_client_idx
    ]
    test_set = TonIotDataset(x=x_test, y=y_test, mean=mean, std=std)

    meta: dict[str, object] = {
        "csv_path": str(csv_path),
        "num_clients": num_clients,
        "mode": mode,
        "num_classes": 2 if mode == "binary" else len(class_names),
        "class_names": class_names if mode == "multi" else ["benign", "attack"],
        "in_features": int(test_set.in_features),
        "test_fraction": test_fraction,
    }
    return client_datasets, test_set, meta
