"""Dataset entry points for the IoT benchmarks the paper evaluates on.

* `nbaiot.load_nbaiot_per_device` -- primary benchmark, 9 real devices.
* `edge_iiotset.load_edge_iiotset` -- secondary benchmark, ~1.9M flows.
* `tabular.SyntheticTabularDataset` -- offline fixture used by unit tests.

See the individual modules for the per-dataset details. The single
`load_dataset` shim below dispatches by name; callers usually prefer the
explicit loaders for finer control.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from safel_dt.data.edge_iiotset import EdgeIIoTsetDataset, load_edge_iiotset
from safel_dt.data.nbaiot import NBaIoTDataset, load_nbaiot_per_device
from safel_dt.data.toniot import TonIotDataset, load_toniot

SUPPORTED_DATASETS: tuple[str, ...] = ("nbaiot", "edge_iiotset", "toniot")


def load_dataset(
    name: str,
    data_dir: Path | str,
    **kwargs: Any,
) -> tuple[Sequence[Any], Any, dict[str, object]]:
    """Dispatcher: name -> per-client train datasets, shared test set, meta.

    Returns a ``(client_train_sets, test_set, meta)`` triple. ``meta`` is a
    free-form dictionary containing things like ``num_classes`` and
    ``in_features`` that the simulator's model factory consumes.
    """
    name = name.lower()
    if name == "nbaiot":
        return load_nbaiot_per_device(data_dir=data_dir, **kwargs)
    if name == "edge_iiotset":
        return load_edge_iiotset(data_dir=data_dir, **kwargs)
    if name == "toniot":
        return load_toniot(data_dir=data_dir, **kwargs)
    raise ValueError(
        f"Unknown dataset {name!r}; expected one of {SUPPORTED_DATASETS}."
    )


__all__ = [
    "SUPPORTED_DATASETS",
    "EdgeIIoTsetDataset",
    "NBaIoTDataset",
    "TonIotDataset",
    "load_dataset",
    "load_edge_iiotset",
    "load_nbaiot_per_device",
    "load_toniot",
]
